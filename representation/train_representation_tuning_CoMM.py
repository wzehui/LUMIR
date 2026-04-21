# -*- coding: utf-8 -*-
# Hyperparameter search using NN Recall@10 as intrinsic selection metric

# MUST be set before importing torch
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
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from representation.dataset import MultiViewPOIDataset
from representation.encoder_ling import MultisourceTextFusionEncoder
from representation.comm_fusion import CoMMFusionModule
from modality_loss import CoMMLoss


# =========================
# Config
# =========================
CONFIG_PATH = "../configs/representation_config.yaml"

EPOCHS = 15
RECALL_K = 20

# 训练时使用的视图数量
EXPORT_K_VIEWS = 8
EXPORT_BS = 8

# 评估采样数量, 太大就慢
EVAL_MAX_ANCHORS = 20000

# 相似度计算的 chunk
SIM_CHUNK = 4096

# 固定随机性
GLOBAL_SEED = 2025


# =========================
# Search space
# 你按需填写即可
# =========================
SEARCH_SPACE = {
    # loss weights
    "lambda_modality": [1.0],

    # LCompress
    "tau_modality": [0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
    "use_symmetric_global": [True],
    "use_inbatch_for_global": [True],
    "use_inbatch_for_modalities": [False],

}

# 如果你不想跑全排列, 就设一个上限
MAX_TRIALS = 30
EVAL_POS_STRATEGY = "next"
EVAL_POS_WINDOW = 1

RESULTS_CSV = "./hp_search_CoMM_nn_hit_mrr.csv"


# =========================
# Utils
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
    picked = rng.sample(combos, k=max_trials)
    return picked


# =========================
# Positives builder for NN Recall
# =========================
def build_pos_map_from_sessions(
    session_dict: Dict[Any, List[Any]],
    pos_strategy: str = "next",
    pos_window: int = 1,
) -> Dict[Any, Set[Any]]:
    """
    Build positive POI set for each anchor POI from sessions.
    session_dict should be {session_id: [poi_id_1, poi_id_2, ...]}

    pos_strategy:
      - "next": anchor -> the immediate next POI in the same session
      - "window": anchor -> POIs within +/- pos_window around occurrences
    """
    assert pos_strategy in {"next", "window"}

    pos_map: Dict[Any, Set[Any]] = {}

    for sid, seq in session_dict.items():
        if not seq or len(seq) < 2:
            continue

        L = len(seq)
        for i in range(L):
            a = seq[i]
            if a is None:
                continue

            if a not in pos_map:
                pos_map[a] = set()

            if pos_strategy == "next":
                if i + 1 < L:
                    b = seq[i + 1]
                    if b is not None and b != a:
                        pos_map[a].add(b)
            else:
                lo = max(0, i - pos_window)
                hi = min(L, i + pos_window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    b = seq[j]
                    if b is not None and b != a:
                        pos_map[a].add(b)

    return pos_map

def build_session_pos_index(session_dict):
    """
    Build:
      poi -> { session_id -> [positions] }
    """
    poi2sesspos = {}

    for sid, seq in session_dict.items():
        for idx, pid in enumerate(seq):
            if pid is None:
                continue
            if pid not in poi2sesspos:
                poi2sesspos[pid] = {}
            poi2sesspos[pid].setdefault(sid, []).append(idx)

    return poi2sesspos

# =========================
# NN metrics at K
# =========================
@torch.no_grad()
def nn_metrics_at_k(
    embeddings: dict,              # {poi_id: np.ndarray}
    session_dict: dict,            # {session_id: [poi_seq]}
    k: int = 10,
    max_anchors: int = 20000,
    device=None,
    chunk: int = 4096,
) -> Tuple[float, float]:
    """
    Anchor-level Hit@K and MRR@K using immediate-next positives only.

    Positives:
      For each anchor a, positives are the POIs that appear as the immediate next item of a
      in any session.

    Hit@K:
      For each anchor, hit is 1 if any topK neighbor is in positives(a).

    MRR@K:
      Reciprocal rank of the first topK neighbor that is in positives(a).
      If none, contribute 0.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build immediate-next positives
    pos_map = build_pos_map_from_sessions(
        session_dict=session_dict,
        pos_strategy=EVAL_POS_STRATEGY,
        pos_window=EVAL_POS_WINDOW,
    )

    # Embedding matrix
    poi_ids = list(embeddings.keys())
    if not poi_ids:
        return 0.0, 0.0

    id2idx = {pid: i for i, pid in enumerate(poi_ids)}
    M = torch.tensor(
        np.stack([embeddings[pid] for pid in poi_ids]),
        device=device,
        dtype=torch.float32
    )
    M = F.normalize(M, dim=1)

    # Anchors must have at least one next-positive and have embeddings
    anchors = [a for a in pos_map.keys() if a in id2idx and len(pos_map[a]) > 0]
    if not anchors:
        return 0.0, 0.0

    if len(anchors) > max_anchors:
        rng = np.random.default_rng(2025)
        anchors = list(rng.choice(anchors, size=max_anchors, replace=False))

    hit_anchors = 0
    mrr_sum = 0.0
    total_anchors = 0

    for a in tqdm(anchors, desc=f"NN Hit/MRR@{k}", leave=False):
        a_idx = id2idx[a]
        q = M[a_idx:a_idx + 1]

        # Chunked topK
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
# Train and export embeddings
# =========================
def build_dataloader(dataset: MultiViewPOIDataset, config: Dict[str, Any], device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=lambda b: b,
        persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )


def build_models(config: Dict[str, Any], device: torch.device):
    encoder = MultisourceTextFusionEncoder(config=config).to(device)

    fusion_module = CoMMFusionModule(
        embedding_dim=config["model"]["embedding_dim"],
        fusion_dim=config["model"]["fusion_dim"],
        projection_dim=config["model"]["projection_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        modality_token_per_modality=config["modality_token_per_modality"],
        fusion_schedule=config["fusion_schedule"],
        use_gating=False
    ).to(device)

    return encoder, fusion_module


def build_losses(config: Dict[str, Any], dataset: MultiViewPOIDataset):
    comm_loss_fn = CoMMLoss(
        tau=float(config["loss"]["tau_modality"]),
        memory_size=4096,
        embedding_dim=int(config["model"]["projection_dim"]),
        use_symmetric_global=bool(config["loss"]["use_symmetric_global"]),
        use_inbatch_for_modalities=bool(config["loss"]["use_inbatch_for_modalities"]),
        use_inbatch_for_global=bool(config["loss"]["use_inbatch_for_global"]),
    )

    return comm_loss_fn


def train_one_run_and_eval(
    base_config: Dict[str, Any],
    hp: Dict[str, Any],
    run_seed: int,
    device: torch.device,
) -> Tuple[float, float, Dict[str, Any], Dict[Any, np.ndarray]]:
    """
    Train for EPOCHS then compute anchor-level Hit@K and MRR@K on session positives.
    Returns (hit_k, mrr_k, extra_stats, embeddings)
    """
    config = copy.deepcopy(base_config)

    # inject hparams
    config["loss"]["lambda_modality"] = float(hp["lambda_modality"])
    config["loss"]["tau_modality"] = float(hp["tau_modality"])
    config["loss"]["use_symmetric_global"] = bool(hp["use_symmetric_global"])
    config["loss"]["use_inbatch_for_global"] = bool(hp["use_inbatch_for_global"])
    config["loss"]["use_inbatch_for_modalities"] = bool(hp["use_inbatch_for_modalities"])

    set_all_seeds(run_seed)

    dataset = MultiViewPOIDataset(
        session_df_path=config["paths"]["sequence"],
        product_csv_path=config["paths"]["product"],
        review_csv_path=config["paths"]["review"],
        photo_csv_path=config["paths"]["photo"],
        itemid_map_path=config["paths"]["itemid_map_path"],
        subset_ratio=config.get("subset_ratio", 0.05),
        subset_size=config.get("subset_size", None),
        seed=run_seed
    )

    dataloader = build_dataloader(dataset, config, device)
    encoder, fusion_module = build_models(config, device)

    comm_loss_fn = build_losses(config, dataset)
    comm_loss_fn = comm_loss_fn.to(device)

    optimizer = torch.optim.AdamW(
        list(fusion_module.parameters()) + list(encoder.get_trainable_parameters()),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"])
    )
    scaler = torch.amp.GradScaler(enabled=True)

    used_modalities = list(config["used_modalities"])
    lambda_modality = float(config["loss"]["lambda_modality"])

    # train
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        fusion_module.train()
        fusion_module.current_epoch = epoch

        pbar = tqdm(dataloader, desc=f"Train ep{epoch}", leave=False)
        for batch in pbar:
            batch = [item for item in batch if item is not None]
            if not batch:
                continue

            item_ids = [item["ItemId"] for item in batch]
            text_dict = {k: [item[k] for item in batch] for k in used_modalities}

            modality_embeddings = encoder(text_dict, item_ids=item_ids)
            modality_list = [modality_embeddings[k] for k in used_modalities]
            B = modality_list[0].shape[0]

            mask_full = [[True] * len(used_modalities) for _ in range(B)]
            mask_drop1 = fusion_module.gen_random_mask(B, min_keep=1, max_keep=len(used_modalities))
            mask_drop2 = fusion_module.gen_random_mask(B, min_keep=1, max_keep=len(used_modalities))

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                Z_full, _, _ = fusion_module(modality_list, mask_full)
                Z_v1, _, _ = fusion_module(modality_list, mask_drop1)
                Z_v2, _, _ = fusion_module(modality_list, mask_drop2)

                Z_modalities_view = [
                    fusion_module.project_modality(m, name)
                    for m, name in zip(modality_list, fusion_module.modalities)
                ]

                outputs = {
                    "aug1_embed": [Z_v1] + Z_modalities_view,
                    "aug2_embed": [Z_v2] + Z_modalities_view,
                }

                comm_loss_dict = comm_loss_fn(outputs)
                comm_loss_value = comm_loss_dict["loss"]

                total_loss = lambda_modality * comm_loss_value

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                comm_loss_fn.update_memory(Z_v2)

    # export embeddings using K random drop views
    encoder.eval()
    fusion_module.eval()

    export_loader = DataLoader(
        dataset,
        batch_size=EXPORT_BS,
        shuffle=False,
        collate_fn=lambda b: b,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    poi_embeddings_mean: Dict[Any, np.ndarray] = {}
    n_mod = len(used_modalities)

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(export_loader, desc="Export", leave=False)):
            batch = [x for x in batch if x is not None]
            if not batch:
                continue

            item_ids = [x["ItemId"] for x in batch]
            text_dict = {k: [x[k] for x in batch] for k in used_modalities}

            modality_embeddings = encoder(text_dict, item_ids=item_ids)
            modality_list = [modality_embeddings[k] for k in used_modalities]
            B = modality_list[0].size(0)

            Z_accum = 0.0
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(run_seed + bidx)
                for _ in range(EXPORT_K_VIEWS):
                    mask = fusion_module.gen_random_mask(B, min_keep=1, max_keep=n_mod)
                    Z_k, _, _ = fusion_module(modality_list, mask)
                    Z_accum = Z_accum + Z_k

            Z_mean = F.normalize(Z_accum / float(EXPORT_K_VIEWS), dim=-1)

            for i, iid in enumerate(item_ids):
                poi_embeddings_mean[iid] = Z_mean[i].detach().cpu().numpy()

    # build positives for NN Recall
    pos_map = build_pos_map_from_sessions(
        session_dict=dataset.session_dict,
        pos_strategy=EVAL_POS_STRATEGY,
        pos_window=EVAL_POS_WINDOW,
    )

    hit_k, mrr_k = nn_metrics_at_k(
        embeddings=poi_embeddings_mean,
        session_dict=dataset.session_dict,
        k=RECALL_K,
        max_anchors=EVAL_MAX_ANCHORS,
        device=device,
        chunk=SIM_CHUNK,
    )

    extra = {
        "n_embeddings": int(len(poi_embeddings_mean)),
        "n_pos_anchors": int(sum(1 for a in pos_map.keys() if a in poi_embeddings_mean and len(pos_map[a]) > 0)),
    }
    return hit_k, mrr_k, extra, poi_embeddings_mean

# =========================
# Main search
# =========================
def main():
    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    base_config = copy.deepcopy(base_config)
    base_config["seed"] = int(base_config.get("seed", GLOBAL_SEED))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    combos = flatten_search_space(SEARCH_SPACE)
    combos = sample_combos(combos, MAX_TRIALS, seed=GLOBAL_SEED)
    print(f"Trials: {len(combos)}")

    rows = []
    best = {"mrr_k": -1.0, "hit_k": -1.0, "hp": None}

    for t, hp in enumerate(combos, 1):
        run_seed = GLOBAL_SEED
        print("\n==============================")
        print(f"Trial {t}/{len(combos)} seed={run_seed}")
        print(json.dumps(hp, ensure_ascii=False, indent=2))

        start = time.time()
        try:
            hit_k, mrr_k, extra, embeddings = train_one_run_and_eval(
                base_config=base_config,
                hp=hp,
                run_seed=run_seed,
                device=device
            )
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            hit_k = float("nan")
            mrr_k = float("nan")
            extra = {"error": str(e)}

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

        if not math.isnan(mrr_k):
            better = False

            # Primary metric: MRR@K
            eps = 1e-12
            if mrr_k > best["mrr_k"] + eps:
                better = True
            # Secondary metric: Hit@K
            elif abs(mrr_k - best["mrr_k"]) <= eps and (
            not math.isnan(hit_k)) and hit_k > best["hit_k"]:
                better = True

            if better:
                best["mrr_k"] = mrr_k
                best["hit_k"] = hit_k if not math.isnan(hit_k) else best["hit_k"]
                best["hp"] = hp

                # Only save embeddings when the best changes
                save_dir = "./best_embedding"
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(
                    save_dir,
                    f"best_embedding_trial{t}_mrr{mrr_k:.4f}_hit{hit_k:.4f}.csv.gz"
                )

                records = [
                    {"ItemId": iid, "embedding": json.dumps(vec.tolist())}
                    for iid, vec in embeddings.items()
                ]
                pd.DataFrame(records).to_csv(save_path, index=False,
                                             compression="gzip")
                print(f"✅ Saved new best embedding to: {save_path}")

        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    print("\n==============================")
    print(f"Best by MRR@{RECALL_K} then Hit@{RECALL_K}")
    print(f"Best mrr_k: {best['mrr_k']:.6f}")
    print(f"Best hit_k: {best['hit_k']:.6f}")
    print(json.dumps(best["hp"], ensure_ascii=False, indent=2))
    print(f"Saved: {RESULTS_CSV}")


if __name__ == "__main__":
    set_all_seeds(GLOBAL_SEED)
    main()