# -*- coding: utf-8 -*-
# FLAVA style text-only tri-modal attention baseline for Instagram
# Session InfoNCE training + NN Hit/MRR selection
#
# Choice A
#   Use ONE dataset (MultiViewPOIDataset) for session_dict only.
#   Do NOT generate pt here.
#   Require raw pt already exists in CACHE_DIR:
#     {meta_raw, review_raw, photo_raw}.pt
#
# Output:
#   ../instagram/embeddings/embedding_flava_text3_meta-review-photo_raw_4096_2048_ins.csv.gz
#   ./hp_search_flava_text3_ins.csv

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import yaml
import math
import time
import random
import itertools
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from representation.dataset import MultiViewPOIDataset
from representation.utils import save_poi_embeddings


# =========================
# Config
# =========================
CONFIG_PATH = "../configs/representation_config.yaml"

GLOBAL_SEED = 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODALITIES = ["meta", "review", "photo"]

# >>> MOD 1: raw pt cache dir and names (same as weighted concat)
CACHE_DIR = "../cache/poi_embeddings"

OUT_DIR = "../yelp/embeddings"
OUT_NAME_BEST = "embedding_flava_text3_meta-review-photo_raw_4096_2048.csv.gz"
RESULTS_CSV = "./hp_search_flava_text3.csv"

EPOCHS = 10
BATCH_SIZE_PAIRS = 4096
EMB_BATCH_EXPORT = 2048

RECALL_K = 20
EVAL_MAX_ANCHORS = 20000
SIM_CHUNK = 4096

RAW_DIM = 4096
PCA_SEED = 2025  # keep for reproducibility if you later add PCA


# =========================
# Search space
# =========================
SEARCH_SPACE = {
    "target_dim": [2048],
    "lr": [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay": [1e-5, 5e-5, 1e-4, 3e-4, 1e-3],
    "temperature": [0.03, 0.04, 0.05, 0.07, 0.1],
    "num_layers": [2, 3, 4],
    "num_heads": [4, 8, 16],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "ff_mult": [2, 3, 4],
}
MAX_TRIALS = 20


# =========================
# Repro
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_search_space(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def sample_combos(combos: List[Dict[str, Any]], max_trials: int, seed: int) -> List[Dict[str, Any]]:
    if len(combos) <= max_trials:
        return combos
    rng = random.Random(seed)
    return rng.sample(combos, k=max_trials)


# =========================
# >>> MOD 2: Raw pt loading (same as weighted concat)
# =========================
def raw_pt_path(cache_dir: str, modality: str) -> str:
    return os.path.join(cache_dir, f"{modality}_raw.pt")


def load_raw_embeddings(
    cache_dir: str,
    modalities: List[str],
    expect_dim: int,
) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
    raw_data: Dict[str, torch.Tensor] = {}
    base_ids: Optional[List[Any]] = None

    for m in modalities:
        p = raw_pt_path(cache_dir, m)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing raw pt: {p}")

        data = torch.load(p, map_location="cpu")
        if not isinstance(data, dict) or "item_id" not in data or "embedding" not in data:
            raise RuntimeError(f"Invalid pt structure: {p}. Expect keys: item_id, embedding")

        item_ids = list(data["item_id"])
        emb = data["embedding"]

        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(f"Invalid embedding type in {p}: {type(emb)}")

        if emb.ndim != 2 or emb.size(1) != int(expect_dim):
            raise RuntimeError(f"Dim mismatch in {p}: got {tuple(emb.shape)}, expect [N,{expect_dim}]")

        if len(item_ids) != emb.size(0):
            raise RuntimeError(f"Row mismatch in {p}: len(item_id)={len(item_ids)} but emb.shape[0]={emb.size(0)}")

        if base_ids is None:
            base_ids = item_ids
        else:
            if item_ids != base_ids:
                raise RuntimeError(f"item_id order mismatch across modalities, problem at modality={m}")

        raw_data[m] = emb

    assert base_ids is not None
    return raw_data, base_ids


# =========================
# NN metrics
# =========================
def build_pos_map_from_sessions(session_dict: Dict[Any, List[Any]]) -> Dict[Any, set]:
    pos_map: Dict[Any, set] = {}
    for sid, seq in session_dict.items():
        if not seq or len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            if a is None or b is None or a == b:
                continue
            pos_map.setdefault(a, set()).add(b)
    return pos_map


@torch.no_grad()
def nn_metrics_at_k(
    embeddings: Dict[Any, np.ndarray],
    session_dict: Dict[Any, List[Any]],
    k: int = 20,
    max_anchors: int = 20000,
    device: Optional[torch.device] = None,
    chunk: int = 4096,
) -> Tuple[float, float]:
    if device is None:
        device = DEVICE

    pos_map = build_pos_map_from_sessions(session_dict)

    poi_ids = list(embeddings.keys())
    if not poi_ids:
        return 0.0, 0.0

    id2idx = {pid: i for i, pid in enumerate(poi_ids)}
    M = torch.tensor(np.stack([embeddings[pid] for pid in poi_ids]), device=device, dtype=torch.float32)
    M = F.normalize(M, dim=1)

    anchors = [a for a in pos_map.keys() if a in id2idx and len(pos_map[a]) > 0]
    if not anchors:
        return 0.0, 0.0

    if len(anchors) > max_anchors:
        rng = np.random.default_rng(GLOBAL_SEED)
        anchors = list(rng.choice(anchors, size=max_anchors, replace=False))

    hit_anchors = 0
    mrr_sum = 0.0
    total_anchors = 0

    for a in tqdm(anchors, desc=f"NN Hit/MRR@{k}", leave=False):
        a_idx = id2idx[a]
        q = M[a_idx:a_idx + 1]

        best_scores = None
        best_indices = None

        for start in range(0, M.size(0), chunk):
            end = min(start + chunk, M.size(0))
            sims = (q @ M[start:end].T).squeeze(0)

            if start <= a_idx < end:
                sims[a_idx - start] = -1e9

            topk_s, topk_i = torch.topk(sims, min(k, sims.numel()))
            topk_i += start

            if best_scores is None:
                best_scores = topk_s
                best_indices = topk_i
            else:
                all_s = torch.cat([best_scores, topk_s])
                all_i = torch.cat([best_indices, topk_i])
                keep_s, keep_idx = torch.topk(all_s, min(k, all_s.numel()))
                best_scores = keep_s
                best_indices = all_i[keep_idx]

        nn_ids = [poi_ids[i] for i in best_indices.tolist()]
        positives = pos_map.get(a, set())

        first_hit_rank = None
        for rank, b in enumerate(nn_ids, start=1):
            if b in positives:
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            hit_anchors += 1
            mrr_sum += 1.0 / float(first_hit_rank)

        total_anchors += 1

    hit_at_k = hit_anchors / total_anchors if total_anchors > 0 else 0.0
    mrr_at_k = mrr_sum / total_anchors if total_anchors > 0 else 0.0
    return hit_at_k, mrr_at_k


# =========================
# Pair dataset
# =========================
class SessionNextPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


def build_pairs_from_sessions(session_dict: Dict[Any, List[Any]], id2row: Dict[Any, int]) -> List[Tuple[int, int]]:
    pairs = []
    for sid, seq in session_dict.items():
        if not seq or len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            if a is None or b is None or a == b:
                continue
            if a in id2row and b in id2row:
                pairs.append((id2row[a], id2row[b]))
    return pairs


# =========================
# FLAVA style token mixer for 3 text modalities
# =========================
class FlavaText3Fusion(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        target_dim: int,
        modalities: List[str],
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_mult: int,
    ):
        super().__init__()
        self.modalities = modalities
        self.target_dim = target_dim

        self.proj = nn.ModuleDict({
            m: nn.Linear(in_dims[m], target_dim, bias=False)
            for m in modalities
        })

        self.cls = nn.Parameter(torch.zeros(1, 1, target_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + len(modalities), target_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=target_dim,
            nhead=int(num_heads),
            dim_feedforward=int(target_dim * int(ff_mult)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(target_dim)

        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        toks = []
        for m in self.modalities:
            toks.append(self.proj[m](x[m]))
        toks = torch.stack(toks, dim=1)

        B = toks.size(0)
        cls = self.cls.expand(B, 1, self.target_dim)
        X = torch.cat([cls, toks], dim=1)
        X = X + self.pos[:, :X.size(1), :]

        X = self.encoder(X)
        z = X[:, 0, :]
        z = self.out_norm(z)
        z = F.normalize(z, dim=-1)
        return z


# =========================
# Session InfoNCE
# =========================
def info_nce_session(anchor: torch.Tensor, pos: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    pos = F.normalize(pos, dim=-1)
    logits = (anchor @ pos.t()) / float(temperature)
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def export_all_embeddings(
    fusion: nn.Module,
    all_embeddings: Dict[str, torch.Tensor],
    poi_ids: List[Any],
    modalities: List[str],
    device: torch.device,
    batch_size: int,
) -> Dict[Any, np.ndarray]:
    fusion.eval()
    N = len(poi_ids)
    out: Dict[Any, np.ndarray] = {}

    for start in tqdm(range(0, N, batch_size), desc="Export fused", leave=False):
        end = min(start + batch_size, N)
        x = {m: all_embeddings[m][start:end].to(device) for m in modalities}
        z = fusion(x).detach().cpu().numpy()
        for i in range(end - start):
            out[poi_ids[start + i]] = z[i]
    return out


# =========================
# Main train + eval for one hp
# =========================
def train_one_run_and_eval(
    base_config: Dict[str, Any],
    hp: Dict[str, Any],
    run_seed: int,
) -> Tuple[float, float, Dict[str, Any], Dict[Any, np.ndarray]]:
    set_all_seeds(run_seed)
    config = copy.deepcopy(base_config)

    dataset = MultiViewPOIDataset(
        session_df_path=config["paths"]["sequence"],
        product_csv_path=config["paths"]["product"],
        review_csv_path=config["paths"]["review"],
        photo_csv_path=config["paths"]["photo"],
        itemid_map_path=config["paths"]["itemid_map_path"],
        subset_ratio=config.get("subset_ratio", 0.05),
        subset_size=config.get("subset_size", None),
        seed=run_seed,
    )

    raw_embeddings, poi_ids = load_raw_embeddings(
        cache_dir=CACHE_DIR,
        modalities=MODALITIES,
        expect_dim=RAW_DIM,
    )

    all_embeddings = {m: raw_embeddings[m].float().cpu().contiguous() for m in MODALITIES}

    id2row = {pid: i for i, pid in enumerate(poi_ids)}
    pairs = build_pairs_from_sessions(dataset.session_dict, id2row)
    if len(pairs) == 0:
        raise RuntimeError("No valid (anchor,next) pairs found after intersecting with cached POI ids.")

    pair_ds = SessionNextPairDataset(pairs)
    pair_loader = DataLoader(
        pair_ds,
        batch_size=int(BATCH_SIZE_PAIRS),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    in_dims = {m: int(all_embeddings[m].shape[1]) for m in MODALITIES}
    fusion = FlavaText3Fusion(
        in_dims=in_dims,
        target_dim=int(hp["target_dim"]),
        modalities=MODALITIES,
        num_layers=int(hp["num_layers"]),
        num_heads=int(hp["num_heads"]),
        dropout=float(hp["dropout"]),
        ff_mult=int(hp["ff_mult"]),
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        fusion.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )

    temperature = float(hp["temperature"])

    for epoch in range(1, EPOCHS + 1):
        fusion.train()
        pbar = tqdm(pair_loader, desc=f"Train ep{epoch}", leave=False)
        for a_idx, b_idx in pbar:
            a_idx = a_idx.long()
            b_idx = b_idx.long()

            x_a = {m: all_embeddings[m][a_idx].to(DEVICE) for m in MODALITIES}
            x_b = {m: all_embeddings[m][b_idx].to(DEVICE) for m in MODALITIES}

            z_a = fusion(x_a)
            z_b = fusion(x_b)

            loss = info_nce_session(z_a, z_b, temperature)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optim.step()

    fused_all = export_all_embeddings(
        fusion=fusion,
        all_embeddings=all_embeddings,
        poi_ids=poi_ids,
        modalities=MODALITIES,
        device=DEVICE,
        batch_size=EMB_BATCH_EXPORT,
    )

    hit_k, mrr_k = nn_metrics_at_k(
        embeddings=fused_all,
        session_dict=dataset.session_dict,
        k=RECALL_K,
        max_anchors=EVAL_MAX_ANCHORS,
        device=DEVICE,
        chunk=SIM_CHUNK,
    )

    extra = {
        "n_pairs": int(len(pairs)),
        "n_pois": int(len(poi_ids)),
        "raw_dim": int(RAW_DIM),
        "target_dim": int(hp["target_dim"]),
        "num_layers": int(hp["num_layers"]),
        "num_heads": int(hp["num_heads"]),
        "dropout": float(hp["dropout"]),
        "ff_mult": int(hp["ff_mult"]),
    }
    return hit_k, mrr_k, extra, fused_all


def main():
    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    base_config = copy.deepcopy(base_config)
    base_config["seed"] = int(base_config.get("seed", GLOBAL_SEED))

    combos = flatten_search_space(SEARCH_SPACE)
    combos = sample_combos(combos, MAX_TRIALS, seed=GLOBAL_SEED)

    rows = []
    best = {"mrr_k": -1.0, "hit_k": -1.0, "hp": None, "emb": None}

    for t, hp in enumerate(combos, 1):
        run_seed = GLOBAL_SEED

        print("\n==============================")
        print(f"Trial {t}/{len(combos)} seed={run_seed}")
        print(json.dumps(hp, ensure_ascii=False, indent=2))

        start = time.time()
        try:
            hit_k, mrr_k, extra, fused_all = train_one_run_and_eval(
                base_config=base_config,
                hp=hp,
                run_seed=run_seed,
            )
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            hit_k = float("nan")
            mrr_k = float("nan")
            extra = {"error": str(e)}
            fused_all = None

        elapsed = time.time() - start
        print(f"Hit@{RECALL_K} = {hit_k:.6f}   MRR@{RECALL_K} = {mrr_k:.6f}   time={elapsed:.1f}s")

        out = {
            "trial": t,
            "seed": run_seed,
            "hit_k": hit_k,
            "mrr_k": mrr_k,
            "k": int(RECALL_K),
            "time_sec": elapsed,
            **hp,
            **extra,
        }
        rows.append(out)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

        if not math.isnan(mrr_k):
            eps = 1e-12
            better = (mrr_k > best["mrr_k"] + eps) or (
                abs(mrr_k - best["mrr_k"]) <= eps
                and (not math.isnan(hit_k))
                and hit_k > best["hit_k"]
            )
            if better:
                best["mrr_k"] = mrr_k
                best["hit_k"] = hit_k if not math.isnan(hit_k) else best["hit_k"]
                best["hp"] = hp
                best["emb"] = fused_all

                print("✅ New best found, saving embedding...")
                save_poi_embeddings(
                    poi_embeddings=best["emb"],
                    save_dir=OUT_DIR,
                    filename=OUT_NAME_BEST,
                )

    print("\n==============================")
    print(f"Best by MRR@{RECALL_K} then Hit@{RECALL_K}")
    print(f"Best mrr_k: {best['mrr_k']:.6f}")
    print(f"Best hit_k: {best['hit_k']:.6f}")
    print(json.dumps(best["hp"], ensure_ascii=False, indent=2))
    print(f"Saved best embedding: {os.path.join(OUT_DIR, OUT_NAME_BEST)}")
    print(f"Saved results: {RESULTS_CSV}")


if __name__ == "__main__":
    set_all_seeds(GLOBAL_SEED)
    main()