# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from representation.encoder_ling import MultisourceTextFusionEncoder
from preprocessing.preparation_embedding_ins import InstagramPOITextDataset
from representation.utils import save_poi_embeddings


# >>> MOD 1: raw pt 的文件名规范
def raw_pt_path(cache_dir: str, modality: str) -> str:
    return os.path.join(cache_dir, f"{modality}_raw.pt")


# >>> MOD 2: 校验 raw pt 是否存在且维度正确
def raw_pt_ok(pt_path: str, expect_dim: int) -> bool:
    if not os.path.exists(pt_path):
        return False
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "item_id" not in data or "embedding" not in data:
        return False
    emb = data["embedding"]
    if not isinstance(emb, torch.Tensor):
        return False
    if emb.ndim != 2 or emb.size(1) != int(expect_dim):
        return False
    if len(data["item_id"]) != emb.size(0):
        return False
    return True


# >>> MOD 3: 生成 raw pt（调用 encoder.save_static_embedding）
def ensure_raw_pt_cache(
    encoder,
    dataset,
    cache_dir,
    modalities,
    device,
    expect_dim=4096,
    batch_size=8,
    num_workers=0,
):
    os.makedirs(cache_dir, exist_ok=True)

    # save_static_embedding 需要 dataloader 输出 list[dict]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda xs: xs,  # 保持 batch 为 list
    )

    # 如果有任何一个 raw 不存在或维度不对，就调用一次 save_static_embedding
    need_build = False
    for m in modalities:
        p = raw_pt_path(cache_dir, m)
        if not raw_pt_ok(p, expect_dim):
            need_build = True
            break

    if need_build:
        # >>> 关键: 让 encoder.save_static_embedding 写到 cache_dir
        print(f"[GEN] raw pt cache missing or invalid. Generating raw embeddings into {cache_dir}")
        encoder = encoder.to(device)
        encoder.eval()
        encoder.save_static_embedding(dataloader=loader, save_dir=cache_dir)

    # 生成后逐个校验
    for m in modalities:
        p = raw_pt_path(cache_dir, m)
        if not raw_pt_ok(p, expect_dim):
            # 尝试打印实际 shape 方便定位
            data = torch.load(p, map_location="cpu") if os.path.exists(p) else None
            shape = None
            if isinstance(data, dict) and isinstance(data.get("embedding", None), torch.Tensor):
                shape = tuple(data["embedding"].shape)
            raise RuntimeError(f"Raw cache invalid for {m}: {p}, got {shape}, expect [N,{expect_dim}].")

        data = torch.load(p, map_location="cpu")
        print(f"[OK] {m}_raw.pt: item={len(data['item_id'])}, emb_shape={tuple(data['embedding'].shape)}")


# >>> MOD 4: 从 raw pt 读取 embeddings，并保证 item_id 对齐
def load_raw_embeddings(cache_dir: str, modalities):
    raw_data = {}
    base_ids = None

    for m in modalities:
        p = raw_pt_path(cache_dir, m)
        data = torch.load(p, map_location="cpu")

        item_ids = data["item_id"]
        emb = data["embedding"]

        if base_ids is None:
            base_ids = item_ids
        else:
            # 要求三份 raw 的 item_id 顺序完全一致
            if list(item_ids) != list(base_ids):
                raise RuntimeError(f"item_id order mismatch between modalities. modality={m}")

        raw_data[m] = emb

    return raw_data, base_ids


def main():
    # =========================
    # Paths
    # =========================
    CONFIG_PATH = "../configs/representation_config_ins.yaml"
    CACHE_DIR = "../cache/poi_embeddings_ins"
    OUT_DIR = "../instagram/embeddings"
    OUT_NAME = "embedding_meta-review-photo_raw_4096_2048_ins.csv.gz"

    # =========================
    # Settings
    # =========================
    MODALITIES = ["meta", "review", "photo"]
    RAW_DIM = 4096         # >>> MOD 5: raw pt 目标维度
    TARGET_DIM = 2048
    PCA_SEED = 2025

    CACHE_BATCH_SIZE = 8   # raw embedding 生成时的 batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    poi_dataset = InstagramPOITextDataset(
        business_csv_path=config["paths"]["product"],
        caption_summary_csv_path=config["paths"]["review"],
        image_summary_csv_path=config["paths"]["photo"],
        h3_resolutions=(9,),
    )

    encoder = MultisourceTextFusionEncoder(config=config).to(device)
    encoder.eval()

    # =========================
    # Step 1: Ensure raw pt exists, otherwise generate via save_static_embedding()
    # =========================
    ensure_raw_pt_cache(
        encoder=encoder,
        dataset=poi_dataset,
        cache_dir=CACHE_DIR,
        modalities=MODALITIES,
        device=device,
        expect_dim=RAW_DIM,
        batch_size=CACHE_BATCH_SIZE,
        num_workers=0,
    )

    # =========================
    # Step 2: Load raw embeddings from pt and concat
    # =========================
    embeddings, poi_ids = load_raw_embeddings(CACHE_DIR, MODALITIES)

    emb_list = []
    for m in MODALITIES:
        emb = embeddings[m]
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"Unexpected embedding type for {m}: {type(emb)}")
        emb_list.append(emb.detach().cpu())

    X = torch.cat(emb_list, dim=1)  # [N, 4096*3]
    X = F.normalize(X, dim=1)

    in_dim = X.size(1)
    if TARGET_DIM > in_dim:
        raise ValueError(f"TARGET_DIM={TARGET_DIM} must be <= in_dim={in_dim} for PCA.")

    # =========================
    # Step 3: PCA projection to TARGET_DIM (keep your setting unchanged)
    # =========================
    X_np = X.numpy()
    pca = PCA(
        n_components=TARGET_DIM,
        svd_solver="randomized",
        random_state=PCA_SEED,
        whiten=False,
    )
    Z_np = pca.fit_transform(X_np)

    Z = torch.from_numpy(Z_np).float()
    Z = F.normalize(Z, dim=1)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[PCA] in_dim={in_dim} -> out_dim={TARGET_DIM}, explained_variance_sum={explained:.6f}")

    poi_embeddings = {iid: Z[i].numpy() for i, iid in enumerate(poi_ids)}

    save_poi_embeddings(
        poi_embeddings=poi_embeddings,
        save_dir=OUT_DIR,
        filename=OUT_NAME,
    )


if __name__ == "__main__":
    main()