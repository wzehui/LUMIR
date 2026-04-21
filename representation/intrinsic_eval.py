# -*- coding: utf-8 -*-
"""
Grid search for LUMIR hyperparameters using intrinsic metric NN Recall@10.

It follows the same NN Recall definition as your evaluation script:
- build poi_to_sessions from dataset.session_dict
- anchors are sampled_poi_ids
- topK neighbors are from the sampled pool
- hit if neighbor shares at least one session with anchor
- NN Recall@K = hits / (K * num_anchors)

Outputs:
- results CSV under config['paths']['results']
"""

# MUST be set before importing torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import yaml
import time
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from representation.dataset import MultiViewPOIDataset
from representation.encoder_ling import MultisourceTextFusionEncoder
from representation.comm_fusion import CoMMFusionModule
from modality_loss import CoMMLoss
from align_loss import MemoryBankAlignLoss


CONFIG_PATH = "../configs/representation_config.yaml"


def load_base_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset_and_loader(config, device):
    dataset = MultiViewPOIDataset(
        session_df_path=config["paths"]["sequence"],
        product_csv_path=config["paths"]["product"],
        review_csv_path=config["paths"]["review"],
        photo_csv_path=config["paths"]["photo"],
        itemid_map_path=config["paths"]["itemid_map_path"],
        subset_ratio=config.get("subset_ratio", None),
        subset_size=config.get("subset_size", None),
        seed=config.get("seed", 2025)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=lambda b: b,
        persistent_workers=True,
        pin_memory=(device.type == "cuda")
    )
    return dataset, dataloader


def build_model_and_losses(config, dataset, device):
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

    comm_loss_fn = CoMMLoss(
        tau=config["loss"]["tau_modality"],
        memory_size=4096,
        embedding_dim=config["model"]["projection_dim"],
        use_symmetric_global=bool(config["loss"].get("use_symmetric_global", True)),
        use_inbatch_for_modalities=bool(config["loss"].get("use_inbatch_for_modalities", False)),
        use_inbatch_for_global=bool(config["loss"].get("use_inbatch_for_global", True))
    ).to(device)

    align_memory_bank = {}
    seqalign_loss_fn = MemoryBankAlignLoss(
        memory_bank=align_memory_bank,
        session_dict=dataset.session_dict,
        poi_to_sessions=dataset.poi_to_sessions,
        temperature=config["loss"]["tau_align"],
        neg_sample_size=int(config["loss"].get("neg_sample_size", 1000)),
        pos_window=int(config["loss"].get("pos_window", 1)),
        pos_strategy=str(config["loss"].get("pos_strategy", "next")),
        normalize_store=True,
        update_bank=True
    )

    optimizer = torch.optim.AdamW(
        list(fusion_module.parameters()) + list(encoder.get_trainable_parameters()),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"])
    )

    scaler = torch.amp.GradScaler(enabled=True)

    return encoder, fusion_module, comm_loss_fn, seqalign_loss_fn, align_memory_bank, optimizer, scaler


def warmup_align_memory_bank(dataloader, encoder, fusion_module, align_memory_bank, config, device):
    print("Warming up align memory bank...")
    encoder_was_training = encoder.training
    fusion_was_training = fusion_module.training
    encoder.eval()
    fusion_module.eval()

    used_modalities = config["used_modalities"]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Warmup Memory Bank"):
            batch = [x for x in batch if x is not None]
            if not batch:
                continue

            item_ids = [x["ItemId"] for x in batch]
            text_dict = {k: [x[k] for x in batch] for k in used_modalities}

            modality_embeddings = encoder(text_dict, item_ids=item_ids)
            modality_list = [modality_embeddings[k] for k in used_modalities]

            mask_full = [[True] * len(used_modalities) for _ in item_ids]
            Z_full, _, _ = fusion_module(modality_list, mask_full)

            for i, iid in enumerate(item_ids):
                align_memory_bank[iid] = Z_full[i].detach().cpu()

    if encoder_was_training:
        encoder.train()
    if fusion_was_training:
        fusion_module.train()

    print(f"Warmup done. bank size: {len(align_memory_bank)}")


def train_for_fixed_epochs(config, dataset, dataloader, encoder, fusion_module,
                           comm_loss_fn, seqalign_loss_fn, align_memory_bank,
                           optimizer, scaler, device, epochs: int):
    lambda_modality = float(config["loss"]["lambda_modality"])
    lambda_align = float(config["loss"]["lambda_align"])
    used_modalities = config["used_modalities"]

    if lambda_align != 0.0 and len(align_memory_bank) == 0:
        warmup_align_memory_bank(dataloader, encoder, fusion_module, align_memory_bank, config, device)

    for epoch in range(1, epochs + 1):
        encoder.train()
        fusion_module.train()
        fusion_module.current_epoch = epoch

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Train epoch {epoch}")
        for batch in pbar:
            batch = [x for x in batch if x is not None]
            if not batch:
                continue

            item_ids = [x["ItemId"] for x in batch]
            text_dict = {k: [x[k] for x in batch] for k in used_modalities}

            modality_embeddings = encoder(text_dict, item_ids=item_ids)
            modality_list = [modality_embeddings[k] for k in used_modalities]
            batch_size = modality_list[0].shape[0]

            mask_full = [[True] * len(used_modalities) for _ in range(batch_size)]
            mask_drop1 = fusion_module.gen_random_mask(batch_size, min_keep=1, max_keep=len(used_modalities))
            mask_drop2 = fusion_module.gen_random_mask(batch_size, min_keep=1, max_keep=len(used_modalities))

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

                align_loss_value = seqalign_loss_fn(anchor_ids=item_ids, anchor_embeddings=Z_full)

                total_loss = (lambda_modality * comm_loss_value) + (lambda_align * align_loss_value)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                comm_loss_fn.update_memory(Z_v2)

            epoch_loss += float(total_loss.item())
            pbar.set_postfix(loss=f"{float(total_loss.item()):.4f}")

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")


def export_embeddings_meanK(config, dataset, encoder, fusion_module, device, epochs: int):
    print("Exporting POI embeddings (mean of K random drop masks)...")

    export_bs = int(config.get("export_bs", 8))
    K_views = int(config.get("export_K", 8))
    used_modalities = config["used_modalities"]
    n_mod = len(used_modalities)
    seed = int(config.get("seed", 2025))

    export_loader = DataLoader(
        dataset,
        batch_size=export_bs,
        shuffle=False,
        collate_fn=lambda b: b,
        num_workers=config.get("num_workers", 4),
        pin_memory=(device.type == "cuda")
    )

    encoder.eval()
    fusion_module.eval()

    poi_embeddings_mean = {}

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(export_loader, desc="Export")):
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
                torch.manual_seed(seed + bidx)
                for _ in range(K_views):
                    mask = fusion_module.gen_random_mask(B, min_keep=1, max_keep=n_mod)
                    Z_k, _, _ = fusion_module(modality_list, mask)
                    Z_accum = Z_accum + Z_k

            Z_mean = F.normalize(Z_accum / float(K_views), dim=-1)

            for i, iid in enumerate(item_ids):
                poi_embeddings_mean[int(iid)] = Z_mean[i].detach().cpu().numpy()

    save_path = config["paths"]["representation"]
    os.makedirs(save_path, exist_ok=True)

    records = [{"ItemId": iid, "embedding": json.dumps(vec.tolist())} for iid, vec in poi_embeddings_mean.items()]
    df = pd.DataFrame.from_records(records)

    out_name = f"lumir_grid_epoch{epochs}.csv.gz"
    out_path = os.path.join(save_path, out_name)
    df.to_csv(out_path, index=False, compression="gzip")

    print(f"Saved embeddings: {out_path}")
    return out_path


def load_embeddings_csv_gz(embedding_path: str):
    df = pd.read_csv(embedding_path)
    poi_ids = df["ItemId"].astype(int).tolist()
    X = np.stack([np.array(json.loads(r), dtype=np.float32) for r in df["embedding"].tolist()])
    return poi_ids, X


def nn_recall_at_k_from_embeddings(dataset, poi_ids, X, K=10, num_poi_sample=1000, seed=2025):
    """
    This matches your NN Recall definition:
    - sample anchors from available poi_ids
    - neighbors are from the sampled pool
    - hit if neighbor shares at least one session with anchor
    - return hits / (K * num_anchors)
    """
    set_seed(seed)

    X_t = torch.tensor(X, dtype=torch.float32)
    X_t = F.normalize(X_t, dim=-1)

    N = len(poi_ids)
    S = min(int(num_poi_sample), N)
    sampled_indices = random.sample(range(N), S)
    sampled_poi_ids = [poi_ids[i] for i in sampled_indices]
    sampled_X = X_t[sampled_indices, :]

    all_sampled_indices = {pid: idx for idx, pid in enumerate(sampled_poi_ids)}

    poi_to_sessions = {}
    for sid, seq in dataset.session_dict.items():
        for pid in seq:
            if pid in all_sampled_indices:
                if pid not in poi_to_sessions:
                    poi_to_sessions[pid] = set()
                poi_to_sessions[pid].add(sid)

    nn_recall_hits = 0
    nn_recall_total = 0

    for pid in tqdm(sampled_poi_ids, desc="NN Recall"):
        e_anchor = sampled_X[all_sampled_indices[pid]].unsqueeze(0)
        sims = F.cosine_similarity(e_anchor, sampled_X).cpu().numpy()

        topk_idx = np.argsort(-sims)[1:K + 1]
        topk_pids = [sampled_poi_ids[i] for i in topk_idx]

        anchor_sessions = poi_to_sessions.get(pid, set())
        hit = 0
        for nbr_pid in topk_pids:
            nbr_sessions = poi_to_sessions.get(nbr_pid, set())
            if len(anchor_sessions & nbr_sessions) > 0:
                hit += 1

        nn_recall_hits += hit
        nn_recall_total += K

    if nn_recall_total == 0:
        return 0.0

    return float(nn_recall_hits) / float(nn_recall_total)


def apply_overrides(config, overrides: dict):
    c = copy.deepcopy(config)
    for k, v in overrides.items():
        if k in {"lambda_modality", "lambda_align", "tau_modality", "tau_align",
                 "use_symmetric_global", "use_inbatch_for_global", "use_inbatch_for_modalities",
                 "pos_strategy", "pos_window"}:
            if "loss" not in c:
                c["loss"] = {}
            if k == "lambda_modality":
                c["loss"]["lambda_modality"] = float(v)
            elif k == "lambda_align":
                c["loss"]["lambda_align"] = float(v)
            elif k == "tau_modality":
                c["loss"]["tau_modality"] = float(v)
            elif k == "tau_align":
                c["loss"]["tau_align"] = float(v)
            elif k == "use_symmetric_global":
                c["loss"]["use_symmetric_global"] = bool(v)
            elif k == "use_inbatch_for_global":
                c["loss"]["use_inbatch_for_global"] = bool(v)
            elif k == "use_inbatch_for_modalities":
                c["loss"]["use_inbatch_for_modalities"] = bool(v)
            elif k == "pos_strategy":
                c["loss"]["pos_strategy"] = str(v)
            elif k == "pos_window":
                c["loss"]["pos_window"] = int(v)
        else:
            c[k] = v
    return c


def main():
    base_config = load_base_config()
    seed = int(base_config.get("seed", 2025))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(base_config["paths"]["results"], exist_ok=True)

    EPOCHS = 15
    NN_K = 10
    NUM_POI_SAMPLE = int(base_config.get("intrinsic_eval", {}).get("num_poi_sample", 1000))

    grid = {
        "lambda_modality": [0.5, 1.0, 2.0],
        "lambda_align": [0.0, 0.25, 0.5, 1.0],
        "tau_modality": [0.05, 0.07, 0.1],
        "use_symmetric_global": [True, False],
        "use_inbatch_for_global": [True, False],
        "use_inbatch_for_modalities": [False, True],
        "tau_align": [0.05, 0.07, 0.1],
        "pos_strategy": ["next", "window"],
        "pos_window": [1, 2],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total combos: {len(combos)}")

    results = []

    for run_id, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values))
        run_config = apply_overrides(base_config, overrides)

        run_tag = (
            f"run{run_id:04d}_lm{run_config['loss']['lambda_modality']}"
            f"_la{run_config['loss']['lambda_align']}"
            f"_tm{run_config['loss']['tau_modality']}"
            f"_ta{run_config['loss']['tau_align']}"
            f"_sg{int(run_config['loss'].get('use_symmetric_global', True))}"
            f"_ibg{int(run_config['loss'].get('use_inbatch_for_global', True))}"
            f"_ibm{int(run_config['loss'].get('use_inbatch_for_modalities', False))}"
            f"_ps{run_config['loss'].get('pos_strategy', 'next')}"
            f"_pw{int(run_config['loss'].get('pos_window', 1))}"
        )

        print("")
        print("=" * 80)
        print(f"Grid run {run_id}/{len(combos)}")
        print(run_tag)
        print(overrides)

        set_seed(seed)

        dataset, dataloader = build_dataset_and_loader(run_config, device)
        encoder, fusion_module, comm_loss_fn, seqalign_loss_fn, align_memory_bank, optimizer, scaler = \
            build_model_and_losses(run_config, dataset, device)

        t0 = time.time()
        train_for_fixed_epochs(
            config=run_config,
            dataset=dataset,
            dataloader=dataloader,
            encoder=encoder,
            fusion_module=fusion_module,
            comm_loss_fn=comm_loss_fn,
            seqalign_loss_fn=seqalign_loss_fn,
            align_memory_bank=align_memory_bank,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epochs=EPOCHS
        )

        emb_path = export_embeddings_meanK(
            config=run_config,
            dataset=dataset,
            encoder=encoder,
            fusion_module=fusion_module,
            device=device,
            epochs=EPOCHS
        )

        poi_ids, X = load_embeddings_csv_gz(emb_path)

        nn_recall = nn_recall_at_k_from_embeddings(
            dataset=dataset,
            poi_ids=poi_ids,
            X=X,
            K=NN_K,
            num_poi_sample=NUM_POI_SAMPLE,
            seed=seed
        )

        dt = time.time() - t0
        print(f"NN Recall@{NN_K}: {nn_recall:.6f}")
        print(f"Time: {dt:.1f}s")

        row = {
            "run_id": run_id,
            "tag": run_tag,
            "epochs": EPOCHS,
            "nn_k": NN_K,
            "num_poi_sample": NUM_POI_SAMPLE,
            "nn_recall": nn_recall,
            "seconds": dt,
        }
        for k in keys:
            row[k] = overrides[k]
        results.append(row)

        del encoder, fusion_module, comm_loss_fn, seqalign_loss_fn
        torch.cuda.empty_cache()

        out_csv = os.path.join(run_config["paths"]["results"], f"grid_nnrecall_epoch{EPOCHS}.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved intermediate results: {out_csv}")

    df = pd.DataFrame(results).sort_values("nn_recall", ascending=False)
    out_csv = os.path.join(base_config["paths"]["results"], f"grid_nnrecall_epoch{EPOCHS}_final.csv")
    df.to_csv(out_csv, index=False)
    print("")
    print("Top 10 configs by NN Recall:")
    print(df.head(10)[["run_id", "nn_recall", "lambda_modality", "lambda_align",
                      "tau_modality", "tau_align", "use_symmetric_global",
                      "use_inbatch_for_global", "use_inbatch_for_modalities",
                      "pos_strategy", "pos_window", "seconds"]])
    print(f"Saved final results: {out_csv}")


if __name__ == "__main__":
    main()