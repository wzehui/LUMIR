# -*- coding: utf-8 -*-
# Weighted concat (fixed best HP) for Instagram
# Session InfoNCE training + Proxy NN Hit/MRR@K (printed) + Save final embedding
#
# Requires raw pt cache:
#   {CACHE_DIR}/{meta_raw,review_raw,photo_raw}.pt
#
# Output:
#   {OUT_DIR}/{OUT_NAME_BEST}

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import yaml
import time
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
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

CACHE_DIR = "../cache/poi_embeddings"  # contains *_raw.pt
OUT_DIR = "../yelp/embeddings"
OUT_NAME_BEST = "embedding_weighted_concat_meta-review-photo_raw_4096_2048.csv.gz"

EPOCHS = 10
BATCH_SIZE_PAIRS = 4096
EMB_BATCH_EXPORT = 2048

RECALL_K = 20
EVAL_MAX_ANCHORS = 20000
SIM_CHUNK = 4096

RAW_DIM = 4096
TARGET_DIM = 2048
PCA_SEED = 2025

# =========================
# Fixed best hyper-params (Trial 3)
# =========================
BEST_HP = {
    "lr": 5e-4,
    "weight_decay": 5e-5,
    "temperature": 0.12,
    "use_softmax_weights": True,
}   # 0.03736526946107784


# =========================
# Repro
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Raw pt loading
# =========================
def raw_pt_path(cache_dir: str, modality: str) -> str:
    return os.path.join(cache_dir, f"{modality}_raw.pt")


def load_raw_embeddings(
    cache_dir: str,
    modalities: List[str],
    expect_dim: int
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
# NN metrics (proxy)
# =========================
def build_pos_map_from_sessions(session_dict: Dict[Any, List[Any]]) -> Dict[Any, set]:
    pos_map: Dict[Any, set] = {}
    for _, seq in session_dict.items():
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
    for _, seq in session_dict.items():
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
# Weighted concat model
# =========================
class WeightedConcat(nn.Module):
    def __init__(self, modalities: List[str], use_softmax_weights: bool = True):
        super().__init__()
        self.modalities = modalities
        self.use_softmax_weights = use_softmax_weights
        self.logits = nn.Parameter(torch.zeros(len(modalities), dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        if self.use_softmax_weights:
            w = F.softmax(self.logits, dim=0)
        else:
            w = torch.sigmoid(self.logits)
            w = w / (w.sum() + 1e-12)
        return w

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.weights()
        blocks = []
        for i, m in enumerate(self.modalities):
            blocks.append(x[m] * w[i])
        out = torch.cat(blocks, dim=1)  # [B, 4096*3]
        out = F.normalize(out, dim=1)
        return out


def info_nce_session(anchor: torch.Tensor, pos: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    pos = F.normalize(pos, dim=-1)
    logits = (anchor @ pos.t()) / float(temperature)
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def export_weighted_X_numpy(
    model: nn.Module,
    all_embeddings: Dict[str, torch.Tensor],
    poi_ids: List[Any],
    modalities: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    N = len(poi_ids)
    chunks = []
    for start in tqdm(range(0, N, batch_size), desc="Export weighted X", leave=False):
        end = min(start + batch_size, N)
        x = {m: all_embeddings[m][start:end].to(device) for m in modalities}
        Xb = model(x).detach().cpu().numpy()
        chunks.append(Xb)
    return np.concatenate(chunks, axis=0)


def main():
    set_all_seeds(GLOBAL_SEED)

    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)
    base_config = copy.deepcopy(base_config)

    # dataset for session_dict only (full dataset if subset_ratio/subset_size are None)
    dataset = MultiViewPOIDataset(
        session_df_path=base_config["paths"]["sequence"],
        product_csv_path=base_config["paths"]["product"],
        review_csv_path=base_config["paths"]["review"],
        photo_csv_path=base_config["paths"]["photo"],
        itemid_map_path=base_config["paths"]["itemid_map_path"],
        subset_ratio=base_config.get("subset_ratio", None),
        subset_size=base_config.get("subset_size", None),
        seed=GLOBAL_SEED,
    )

    print("cwd =", os.getcwd())
    print("CACHE_DIR (abs) =", os.path.abspath(CACHE_DIR))
    print("OUT =", os.path.join(OUT_DIR, OUT_NAME_BEST))
    print("BEST_HP =", json.dumps(BEST_HP, ensure_ascii=False))

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
    print(f"[DATA] n_pois={len(poi_ids)}  n_pairs={len(pairs)}")

    pair_ds = SessionNextPairDataset(pairs)
    pair_loader = DataLoader(
        pair_ds,
        batch_size=int(BATCH_SIZE_PAIRS),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = WeightedConcat(
        modalities=MODALITIES,
        use_softmax_weights=bool(BEST_HP["use_softmax_weights"]),
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(BEST_HP["lr"]),
        weight_decay=float(BEST_HP["weight_decay"]),
    )
    temperature = float(BEST_HP["temperature"])

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(pair_loader, desc=f"Train ep{epoch}", leave=False)
        for a_idx, b_idx in pbar:
            a_idx = a_idx.long()
            b_idx = b_idx.long()

            x_a = {m: all_embeddings[m][a_idx].to(DEVICE) for m in MODALITIES}
            x_b = {m: all_embeddings[m][b_idx].to(DEVICE) for m in MODALITIES}

            z_a = model(x_a)
            z_b = model(x_b)

            loss = info_nce_session(z_a, z_b, temperature)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

    print(f"[DONE] training finished, time_sec={time.time() - t0:.1f}")
    print("[WEIGHTS]", {m: float(w) for m, w in zip(MODALITIES, model.weights().detach().cpu().numpy().tolist())})

    # export weighted X (still 4096*3)
    X_np = export_weighted_X_numpy(
        model=model,
        all_embeddings=all_embeddings,
        poi_ids=poi_ids,
        modalities=MODALITIES,
        device=DEVICE,
        batch_size=EMB_BATCH_EXPORT,
    )

    # PCA must match your simple concat baseline
    X = torch.from_numpy(X_np).float()
    X = F.normalize(X, dim=1)

    in_dim = X.size(1)
    if TARGET_DIM > in_dim:
        raise ValueError(f"TARGET_DIM={TARGET_DIM} must be <= in_dim={in_dim} for PCA.")

    X_np2 = X.numpy()
    pca = PCA(
        n_components=TARGET_DIM,
        svd_solver="randomized",
        random_state=PCA_SEED,
        whiten=False,
    )
    Z_np = pca.fit_transform(X_np2)

    Z = torch.from_numpy(Z_np).float()
    Z = F.normalize(Z, dim=1)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[PCA] in_dim={in_dim} -> out_dim={TARGET_DIM}, explained_variance_sum={explained:.6f}")

    poi_embeddings = {iid: Z[i].numpy() for i, iid in enumerate(poi_ids)}

    # print proxy result (do not save)
    hit_k, mrr_k = nn_metrics_at_k(
        embeddings=poi_embeddings,
        session_dict=dataset.session_dict,
        k=RECALL_K,
        max_anchors=EVAL_MAX_ANCHORS,
        device=DEVICE,
        chunk=SIM_CHUNK,
    )
    print(f"[PROXY] NN Hit@{RECALL_K} = {hit_k:.6f}   MRR@{RECALL_K} = {mrr_k:.6f}")

    # save final embedding
    save_poi_embeddings(
        poi_embeddings=poi_embeddings,
        save_dir=OUT_DIR,
        filename=OUT_NAME_BEST,
    )


if __name__ == "__main__":
    main()