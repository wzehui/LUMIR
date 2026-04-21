# -*- coding: utf-8 -*-
# Reweighted concatenation for Yelp
# Grid search weights (no NN) + fixed PCA + proxy NN Hit/MRR selection
#
# Requirements:
#   {CACHE_DIR}/{meta_raw,review_raw,photo_raw}.pt
#
# Output:
#   ./hp_search_reweighted_concat_yelp.csv
#   (optional) best embedding saved once at the end to {OUT_DIR}/{OUT_NAME_BEST}

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

CACHE_DIR = "../cache/poi_embeddings"
OUT_DIR = "../yelp/embeddings"

RESULTS_CSV = "./hp_search_reweighted_concat_yelp.csv"

RECALL_K = 20
EVAL_MAX_ANCHORS = 20000
SIM_CHUNK = 4096

RAW_DIM = 4096
PCA_FIT_POI = 5000
TARGET_DIM = 2048
PCA_SEED = 2025

# ===== ADDED =====
BEST_W = (0.32, 0.00, 0.68)

# weight search
WEIGHT_STEP = 0.1
MAX_TRIALS = None

# normalization
DO_STANDARDIZE = True
EPS = 1e-6

SAVE_BEST_EMB_AT_END = False
OUT_NAME_BEST = "embedding_reweighted_concat_meta-review-photo_fixedpca_4096_2048.csv.gz"


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

        item_ids = list(data["item_id"])
        emb = data["embedding"]

        if base_ids is None:
            base_ids = item_ids
        else:
            if item_ids != base_ids:
                raise RuntimeError("item order mismatch")

        raw_data[m] = emb.float().cpu().contiguous()

    return raw_data, base_ids


# =========================
# Normalization helpers
# =========================
def l2_only(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def standardize_then_l2(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    x = (x - mu) / torch.sqrt(var + eps)
    return F.normalize(x, dim=1)


# =========================
# Proxy NN metrics
# =========================
def build_pos_map_from_sessions(session_dict: Dict[Any, List[Any]]) -> Dict[Any, set]:
    pos_map: Dict[Any, set] = {}
    for _, seq in session_dict.items():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            if a != b:
                pos_map.setdefault(a, set()).add(b)
    return pos_map


@torch.no_grad()
def nn_metrics_at_k(
    embeddings: Dict[Any, np.ndarray],
    session_dict: Dict[Any, List[Any]],
    k: int = 20,
    max_anchors: int = 20000,
    device=None,
    chunk=4096,
):

    pos_map = build_pos_map_from_sessions(session_dict)

    poi_ids = list(embeddings.keys())
    id2idx = {pid: i for i, pid in enumerate(poi_ids)}

    M = torch.tensor(
        np.stack([embeddings[pid] for pid in poi_ids]),
        device=device,
        dtype=torch.float32
    )

    M = F.normalize(M, dim=1)

    anchors = [a for a in pos_map.keys() if a in id2idx]

    hit = 0
    mrr = 0
    total = 0

    for a in tqdm(anchors, desc="NN eval", leave=False):

        a_idx = id2idx[a]
        q = M[a_idx:a_idx+1]

        sims = torch.matmul(q, M.T).squeeze(0)
        sims[a_idx] = -1e9

        topk = torch.topk(sims, k).indices

        positives = pos_map[a]

        rank = None
        for r, idx in enumerate(topk.tolist(), start=1):
            if poi_ids[idx] in positives:
                rank = r
                break

        if rank is not None:
            hit += 1
            mrr += 1/rank

        total += 1

    return hit/total, mrr/total


# =========================
# Main
# =========================
def main():

    set_all_seeds(GLOBAL_SEED)

    import yaml

    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    base_config = copy.deepcopy(base_config)

    # ===== MODIFIED =====
    # 使用全体数据集
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

    raw, poi_ids = load_raw_embeddings(
        cache_dir=CACHE_DIR,
        modalities=MODALITIES,
        expect_dim=RAW_DIM,
    )

    emb = {}

    for m in MODALITIES:

        x = raw[m]

        if DO_STANDARDIZE:
            x = standardize_then_l2(x, eps=EPS)
        else:
            x = l2_only(x)

        emb[m] = x

    # =========================
    # PCA
    # =========================

    X0 = torch.cat(
        [
            emb["meta"],
            emb["review"],
            emb["photo"],
        ],
        dim=1
    )

    X0 = F.normalize(X0, dim=1)

    X0_np = X0.numpy()

    # ===== MODIFIED =====
    # 不使用 PCA 子采样
    pca = PCA(
        n_components=TARGET_DIM,
        svd_solver="randomized",
        random_state=PCA_SEED,
        whiten=False,
    )

    pca.fit(X0_np)

    print("PCA explained variance:", np.sum(pca.explained_variance_ratio_))

    # =========================
    # weight search
    # =========================

    # ===== DISABLED =====
    # weights = build_local_weight_grid(...)
    # for t, (w_meta, w_review, w_photo) in enumerate(weights, 1):

    # ===== ADDED =====
    w_meta, w_review, w_photo = BEST_W

    print("Using fixed weights:", BEST_W)

    Xw = torch.cat(
        [
            emb["meta"] * w_meta,
            emb["review"] * w_review,
            emb["photo"] * w_photo,
        ],
        dim=1
    )

    Xw = F.normalize(Xw, dim=1)

    Zw_np = pca.transform(Xw.numpy())

    Zw = torch.from_numpy(Zw_np).float()

    Zw = F.normalize(Zw, dim=1)

    poi_embeddings = {
        iid: Zw[i].numpy()
        for i, iid in enumerate(poi_ids)
    }

    # ===== ADDED =====
    # 最终评估一次
    hit_k, mrr_k = nn_metrics_at_k(
        embeddings=poi_embeddings,
        session_dict=dataset.session_dict,
        k=RECALL_K,
        max_anchors=EVAL_MAX_ANCHORS,
        device=DEVICE,
        chunk=SIM_CHUNK,
    )

    print("FINAL RESULT")
    print("Hit@", RECALL_K, "=", hit_k)
    print("MRR@", RECALL_K, "=", mrr_k)

    # ===== ADDED =====
    # 保存 embedding
    os.makedirs(OUT_DIR, exist_ok=True)

    save_poi_embeddings(
        poi_embeddings=poi_embeddings,
        save_dir=OUT_DIR,
        filename=OUT_NAME_BEST,
    )

    print("Embedding saved:", os.path.join(OUT_DIR, OUT_NAME_BEST))


if __name__ == "__main__":
    main()