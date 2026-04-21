# MUST be set before importing torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

from representation.dataset import MultiViewPOIDataset
from representation.encoder_ling import MultisourceTextFusionEncoder
from representation.comm_fusion import CoMMFusionModule
from modality_loss import CoMMLoss
from align_loss import MemoryBankAlignLoss

# ======== CONFIG ========
CONFIG_PATH = "../configs/representation_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])

# ======== Dataset & Dataloader ========
dataset = MultiViewPOIDataset(
    session_df_path=config['paths']['sequence'],
    product_csv_path=config['paths']['product'],
    review_csv_path=config['paths']['review'],
    photo_csv_path=config['paths']['photo'],
    itemid_map_path=config['paths']['itemid_map_path'],
    subset_ratio=config.get('subset_ratio', None),
    subset_size=config.get('subset_size', None),
    seed=config.get('seed', 2025)
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

# ======== Model ========
encoder = MultisourceTextFusionEncoder(config=config).to(device)
fusion_module = CoMMFusionModule(
    embedding_dim=config['model']['embedding_dim'],
    fusion_dim=config['model']['fusion_dim'],
    projection_dim=config['model']['projection_dim'],
    num_layers=config['model']['num_layers'],
    num_heads=config['model']['num_heads'],
    modality_token_per_modality=config['modality_token_per_modality'],
    fusion_schedule=config['fusion_schedule'],
    use_gating=False
).to(device)

# ======== Loss ========
comm_loss_fn = CoMMLoss(
    tau=config['loss']['tau_modality'],
    memory_size=4096,
    embedding_dim=config['model']['projection_dim'],
    use_symmetric_global=True,
    use_inbatch_for_modalities=False,  # modal terms in-batch → save VRAM
    use_inbatch_for_global=True
).to(device)

align_memory_bank = {}
seqalign_loss_fn = MemoryBankAlignLoss(
    memory_bank=align_memory_bank,
    session_dict=dataset.session_dict,
    poi_to_sessions=dataset.poi_to_sessions,
    temperature=config['loss']['tau_align'],
    neg_sample_size=1000,
    pos_window=1,
    pos_strategy="window", #"next",
    normalize_store=True,
    update_bank=True
)

lambda_modality = config['loss']['lambda_modality']
lambda_align = config['loss']['lambda_align']

# ======== Optimizer & Scaler ========
optimizer = torch.optim.AdamW(
    list(fusion_module.parameters()) + list(encoder.get_trainable_parameters()),
    lr=float(config['optimizer']['lr']),
    weight_decay=float(config['optimizer']['weight_decay'])
)
scaler = torch.amp.GradScaler(enabled=True)

# ======== Warm up Align memory bank (reuse training dataloader) ========
def warmup_align_memory_bank_with_train_dl():
    """
    Pre-fill the alignment memory bank with Z_full (full-modality embeddings)
    by making one forward-only pass over the dataset using the *training* dataloader.

    Behavior:
      - Switches encoder/fusion to eval mode and disables grad.
      - Computes Z_full with an all-True mask (full modalities).
      - Writes each item’s embedding into the CPU-side align_memory_bank.
      - Restores the original train/eval states afterward.
    """
    print("🚀 Warming up Align Loss memory bank...")

    # Remember current train/eval state and switch to eval for forward-only pass
    encoder_was_training = encoder.training
    fusion_was_training = fusion_module.training
    encoder.eval()
    fusion_module.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Warmup Memory Bank"):
            batch = [x for x in batch if x is not None]
            if len(batch) == 0:
                continue

            item_ids  = [x["ItemId"] for x in batch]
            text_dict = {k: [x[k] for x in batch] for k in config['used_modalities']}

            # Encode per-modality
            modality_embeddings = encoder(text_dict, item_ids=item_ids)
            modality_list = [modality_embeddings[k] for k in config['used_modalities']]

            # Full-modality mask (all True)
            mask_full = [[True] * len(config['used_modalities']) for _ in item_ids]

            # Fused full-view embedding
            Z_full, _, _ = fusion_module(modality_list, mask_full)

            # Write CPU copies into the bank
            for i, iid in enumerate(item_ids):
                align_memory_bank[iid] = Z_full[i].detach().cpu()

    print(f"✅ Warmup done. Memory bank size: {len(align_memory_bank)}")

    # Restore original train/eval states
    if encoder_was_training:
        encoder.train()
    if fusion_was_training:
        fusion_module.train()

# ======== Warm up Align memory bank (reuse training dataloader) ========
if lambda_align != 0:
    if len(align_memory_bank) == 0:
        warmup_align_memory_bank_with_train_dl()
else:
    print("⚠️ Skipping Align Loss memory bank warm-up because lambda_align = 0.")

# ======== Logging ========
history = {"epoch": [], "total_loss": [], "comm_loss": [], "align_loss": []}
os.makedirs(config['paths']['results'], exist_ok=True)

# ======== Training Loop ========
for epoch in range(1, config['epochs'] + 1):
    encoder.train()
    fusion_module.train()
    fusion_module.current_epoch = epoch

    epoch_loss = epoch_comm_loss = epoch_align_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = [item for item in batch if item is not None]
        item_ids = [item["ItemId"] for item in batch]
        text_dict = {k: [item[k] for item in batch] for k in config['used_modalities']}

        # Encode all modalities
        modality_embeddings = encoder(text_dict, item_ids=item_ids)
        modality_list = [modality_embeddings[k] for k in config['used_modalities']]
        batch_size = modality_list[0].shape[0]

        # === Build three views: full (align), two drops (CoMM) ===
        mask_full  = [[True] * len(config['used_modalities']) for _ in range(batch_size)]
        mask_drop1 = fusion_module.gen_random_mask(batch_size, min_keep=1, max_keep=len(config['used_modalities']))
        mask_drop2 = fusion_module.gen_random_mask(batch_size, min_keep=1, max_keep=len(config['used_modalities']))

        # # --- DEBUG: check the two augmented masks are actually different ---
        # if epoch == 1:  # 只在第1个epoch看下
        #     n_mod = len(fusion_module.modalities)
        #     total_slots = batch_size * n_mod
        #
        #     # 计算两路增广在模态维上的不同比例
        #     diff_cnt = 0
        #     for r1, r2 in zip(mask_drop1, mask_drop2):
        #         for m1, m2 in zip(r1, r2):
        #             diff_cnt += int(m1 != m2)
        #     diff_rate = diff_cnt / total_slots if total_slots > 0 else 0.0
        #
        #     # 也顺便看看每个视图平均保留了多少模态
        #     keep1 = sum(sum(row) for row in mask_drop1) / total_slots
        #     keep2 = sum(sum(row) for row in mask_drop2) / total_slots
        #
        #     print(
        #         f"[dbg] mask difference rate: {diff_rate:.2f} | keep ratio v1={keep1:.2f}, v2={keep2:.2f}")

        with torch.amp.autocast(device_type='cuda'):
        #     import torch.nn.functional as F
        #     # Full-view (Align only)
            Z_full, _, gate_loss_full = fusion_module(modality_list, mask_full)
        #
        #     B = Z_full.size(0)
        #     Zf = F.normalize(Z_full, dim=1)  # 关键：先做 L2 归一化
        #     sim_full = Zf @ Zf.T  # [B,B] 余弦相似度矩阵；对角线=1
        #     # 非对角线的平均余弦
        #     off_mean = ((sim_full.sum() - sim_full.diag().sum()) / (
        #                 B * (B - 1))).item()
        #     # 非对角线的最大余弦
        #     mask_off = ~torch.eye(B, dtype=torch.bool, device=sim_full.device)
        #     off_max = sim_full[mask_off].max().item()
        #     # 每维方差（看有没有几乎全 0 方差）
        #     per_dim_std_mean = Zf.std(dim=0).mean().item()
        #
        #     print(
        #         f"[dbg-Zfull] off_mean={off_mean:.3f}, off_max={off_max:.3f}, "
        #         f"per-dim-std-mean={per_dim_std_mean:.4f},")

            # Two augmented views (CoMM)
            Z_v1, _, gate_loss_v1 = fusion_module(modality_list, mask_drop1)
            Z_v2, _, gate_loss_v2 = fusion_module(modality_list, mask_drop2)

            # # 在 fusion_module(modality_list, mask_*) 返回后立即打印
            # with torch.no_grad():
            #     # 强制使用 float32 计算，避免 fp16 累加误差
            #     q = F.normalize(Z_v1.float(), dim=1)
            #     k = F.normalize(Z_v2.float(), dim=1)
            #
            #     # 正样本：逐行余弦后再取平均
            #     pos_cos = (q * k).sum(dim=1).mean().item()
            #
            #     # 批内负样本：矩阵相乘后去掉对角线再取平均
            #     sim_mat = q @ k.T  # [B,B], 每个值都在 [-1,1]
            #     B = sim_mat.size(0)
            #     off_diag_sum = sim_mat.sum() - sim_mat.diag().sum()
            #     neg_cos = (off_diag_sum / (B * (B - 1))).item()
            #
            #     # 也顺便看下每个向量的 L2 范数，确认是否≈1
            #     q_norm = Z_v1.norm(dim=1).mean().item()
            #     k_norm = Z_v2.norm(dim=1).mean().item()
            #
            #     print(f"[dbg] cos(pos)={pos_cos:.3f}, cos(neg)={neg_cos:.3f}, "
            #           f"||Z_v1||≈{q_norm:.3f}, ||Z_v2||≈{k_norm:.3f}")


                # 看门控是否把一切压平
                # retain_weights_v1/v2 你目前没返回，先看 Z_v* 的 std 即可；或者临时在 forward 里把 retain_weights 也返回

            # Modality-specific reps for CoMM modality terms
            Z_modalities_view = [
                fusion_module.project_modality(m, name)
                for m, name in zip(modality_list, fusion_module.modalities)
            ]
            # Z_modalities_view2 = [
            #     fusion_module.project_modality(m, name)
            #     for m, name in zip(modality_list, fusion_module.modalities)
            # ]

            outputs = {
                "aug1_embed": [Z_v1] + Z_modalities_view,
                "aug2_embed": [Z_v2] + Z_modalities_view,
            }

            # Losses
            comm_loss_dict = comm_loss_fn(outputs)
            comm_loss_value = comm_loss_dict['loss']

            align_loss_value = seqalign_loss_fn(anchor_ids=item_ids, anchor_embeddings=Z_full)

            # Optional: gate loss if you want to regularize
            # gate_loss = gate_loss_full + gate_loss_v1 + gate_loss_v2

            total_loss = (
                lambda_modality * comm_loss_value
                + lambda_align   * align_loss_value
                # + gate_loss
            )

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        # update the CoMM memory bank
        with torch.no_grad():
            comm_loss_fn.update_memory(Z_v2)

        # Accumulate
        epoch_loss += total_loss.item()
        epoch_comm_loss += comm_loss_value.item()
        epoch_align_loss += align_loss_value.item()

        pbar.set_postfix(
            total=f"{total_loss.item():.4f}",
            comm=f"{comm_loss_value.item():.4f}",
            align=f"{align_loss_value.item():.4f}",
        )

    avg_loss = epoch_loss / len(dataloader)
    avg_comm = epoch_comm_loss / len(dataloader)
    avg_align = epoch_align_loss / len(dataloader)
    print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}, CoMM: {avg_comm:.4f}, Align: {avg_align:.4f}")

    # === Plot ===
    history["epoch"].append(int(epoch))
    history["total_loss"].append(avg_loss)
    history["comm_loss"].append(avg_comm)
    history["align_loss"].append(avg_align)

    plt.figure()
    plt.plot(history["epoch"], history["total_loss"], label="Total Loss")
    plt.plot(history["epoch"], history["comm_loss"], label="CoMM Loss")
    plt.plot(history["epoch"], history["align_loss"], label="Align Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.xticks(range(1, int(max(history["epoch"])) + 1))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(config['paths']['results'], "loss_curve.png"))
    plt.close()

# ======== Save final POI embeddings (multi-view mean only) ========
print("Extracting POI embeddings (multi-view mean)...")
export_bs = 8         # Batch size for exporting (tune for your GPU)
K = 8                  # Number of random modality-drop views per POI to average
n_mod = len(config['used_modalities'])
seed = config.get("seed", 2025)

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
        # Skip empty samples that may slip through the collate
        batch = [x for x in batch if x is not None]
        if not batch:
            continue

        # Gather IDs and texts
        item_ids = [x["ItemId"] for x in batch]
        text_dict = {k: [x[k] for x in batch] for k in config['used_modalities']}

        # Encode per-modality features (already on the right device in your encoder)
        modality_embeddings = encoder(text_dict, item_ids=item_ids)
        modality_list = [modality_embeddings[k] for k in config['used_modalities']]
        B = modality_list[0].size(0)

        # Average K random masks per item to get a robust “drop-modality” representation
        # We force at least 1 modality to be dropped (min_keep=1) to match training distribution
        Z_accum = 0
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed + bidx)  # deterministic masks per batch
            for _ in range(K):
                mask = fusion_module.gen_random_mask(B, min_keep=1, max_keep=n_mod)
                # fusion_module returns normalized projected embeddings already
                Z_k, _, _ = fusion_module(modality_list, mask)
                Z_accum = Z_accum + Z_k

        # Mean of K views, then L2-normalize again just to be safe
        Z_mean = F.normalize(Z_accum / K, dim=-1)

        # Store per-POI vectors
        for i, iid in enumerate(item_ids):
            poi_embeddings_mean[iid] = Z_mean[i].detach().cpu().numpy()

# Save
save_path = config['paths']['representation']; os.makedirs(save_path, exist_ok=True)
import json, pandas as pd
records = [{"ItemId": iid, "embedding": json.dumps(vec.tolist())} for iid,
vec in poi_embeddings_mean.items()]
df = pd.DataFrame.from_records(records)
output_filename = (f"lumir_epoch{config['epochs']}.csv.gz")
output_path = os.path.join(save_path, output_filename)
df.to_csv(output_path, index=False, compression="gzip")
print(f"✅ Final POI embeddings saved to {output_path}")