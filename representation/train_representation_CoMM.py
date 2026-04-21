# -*- coding: utf-8 -*-
# CoMM (modality compression only, fixed best HP) for Yelp
# Train CoMM + Proxy NN Hit/MRR@K (printed) + Save final embedding
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
from representation.utils import save_poi_embeddings


# =========================
# Config
# =========================
CONFIG_PATH = "../configs/representation_config.yaml"

GLOBAL_SEED = 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RECALL_K = 20
EVAL_MAX_ANCHORS = 20000
SIM_CHUNK = 4096

EPOCHS = 15
EXPORT_K_VIEWS = 8
EXPORT_BS = 8

# 输出路径与命名方式模仿 weighted concat 脚本
OUT_DIR = "../yelp/embeddings"
OUT_NAME_BEST = "embedding_comm_2048.csv.gz"

# =========================
# Fixed best hyper-params (from your hp_search_CoMM result)
# Best by MRR@20: tau_modality=0.03
# =========================
BEST_HP = {
    "lambda_modality": 1.0,
    "tau_modality": 0.03,
    "use_symmetric_global": True,
    "use_inbatch_for_global": True,
    "use_inbatch_for_modalities": False,
}


# =========================
# Repro
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# Train and export
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
        use_gating=False,
    ).to(device)

    return encoder, fusion_module


def build_loss(config: Dict[str, Any]) -> CoMMLoss:
    return CoMMLoss(
        tau=float(config["loss"]["tau_modality"]),
        memory_size=4096,
        embedding_dim=int(config["model"]["projection_dim"]),
        use_symmetric_global=bool(config["loss"]["use_symmetric_global"]),
        use_inbatch_for_modalities=bool(config["loss"]["use_inbatch_for_modalities"]),
        use_inbatch_for_global=bool(config["loss"]["use_inbatch_for_global"]),
    )


def main():
    set_all_seeds(GLOBAL_SEED)

    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)
    config = copy.deepcopy(base_config)

    # inject best hp
    config["loss"]["lambda_modality"] = float(BEST_HP["lambda_modality"])
    config["loss"]["tau_modality"] = float(BEST_HP["tau_modality"])
    config["loss"]["use_symmetric_global"] = bool(BEST_HP["use_symmetric_global"])
    config["loss"]["use_inbatch_for_global"] = bool(BEST_HP["use_inbatch_for_global"])
    config["loss"]["use_inbatch_for_modalities"] = bool(BEST_HP["use_inbatch_for_modalities"])

    dataset = MultiViewPOIDataset(
        session_df_path=config["paths"]["sequence"],
        product_csv_path=config["paths"]["product"],
        review_csv_path=config["paths"]["review"],
        photo_csv_path=config["paths"]["photo"],
        itemid_map_path=config["paths"]["itemid_map_path"],
        subset_ratio=config.get("subset_ratio", None),
        subset_size=config.get("subset_size", None),
        seed=GLOBAL_SEED,
    )

    print("cwd =", os.getcwd())
    print("OUT =", os.path.join(OUT_DIR, OUT_NAME_BEST))
    print("BEST_HP =", json.dumps(BEST_HP, ensure_ascii=False))
    print("Device =", DEVICE)

    dataloader = build_dataloader(dataset, config, DEVICE)
    encoder, fusion_module = build_models(config, DEVICE)

    comm_loss_fn = build_loss(config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(fusion_module.parameters()) + list(encoder.get_trainable_parameters()),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=True)

    used_modalities = list(config["used_modalities"])
    lambda_modality = float(config["loss"]["lambda_modality"])

    t0 = time.time()
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

            mask_drop1 = fusion_module.gen_random_mask(B, min_keep=1, max_keep=len(used_modalities))
            mask_drop2 = fusion_module.gen_random_mask(B, min_keep=1, max_keep=len(used_modalities))

            with torch.amp.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu"):
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

    print(f"[DONE] training finished, time_sec={time.time() - t0:.1f}")

    # export embeddings by averaging K random drop views
    encoder.eval()
    fusion_module.eval()

    export_loader = DataLoader(
        dataset,
        batch_size=EXPORT_BS,
        shuffle=False,
        collate_fn=lambda b: b,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    poi_embeddings: Dict[Any, np.ndarray] = {}
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
            with torch.random.fork_rng(devices=[DEVICE] if DEVICE.type == "cuda" else []):
                torch.manual_seed(GLOBAL_SEED + bidx)
                for _ in range(EXPORT_K_VIEWS):
                    mask = fusion_module.gen_random_mask(B, min_keep=1, max_keep=n_mod)
                    Z_k, _, _ = fusion_module(modality_list, mask)
                    Z_accum = Z_accum + Z_k

            Z_mean = F.normalize(Z_accum / float(EXPORT_K_VIEWS), dim=-1)
            for i, iid in enumerate(item_ids):
                poi_embeddings[iid] = Z_mean[i].detach().cpu().numpy()

    hit_k, mrr_k = nn_metrics_at_k(
        embeddings=poi_embeddings,
        session_dict=dataset.session_dict,
        k=RECALL_K,
        max_anchors=EVAL_MAX_ANCHORS,
        device=DEVICE,
        chunk=SIM_CHUNK,
    )
    print(f"[PROXY] NN Hit@{RECALL_K} = {hit_k:.6f}   MRR@{RECALL_K} = {mrr_k:.6f}")

    save_poi_embeddings(
        poi_embeddings=poi_embeddings,
        save_dir=OUT_DIR,
        filename=OUT_NAME_BEST,
    )
    print("[SAVE]", os.path.join(OUT_DIR, OUT_NAME_BEST))


if __name__ == "__main__":
    main()