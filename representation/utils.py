import pandas as pd
import h3
import ast
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

# Compute H3 codes (Resolution 5 to 10) and include actual neighboring hexagons
# def generate_h3_features(lat, lon, resolutions=(5, 7, 9)):
def generate_h3_features(lat, lon, resolutions=(9,)):
    if pd.isna(lat) or pd.isna(lon):
        return {}
    result = {}
    for res in resolutions:
        h3_code = h3.latlng_to_cell(lat, lon, res)
        # h3_center = h3.cell_to_latlng(h3_code)
        neighbors = list(set(h3.grid_disk(h3_code, 1)) - {h3_code})
        result[f"H3-{res}"] = {
            "center": h3_code,
            "neighbors": neighbors,
        }
    return result


def combine_attributes(row, keys=None):
    if keys is None:
        keys = ['summary', 'keywords', 'indoor_color_tone', 'venue_style',
                'food_style', 'drink_style', 'target_audience',
                'special_features']
    parts = [f"{key.replace('_', ' ').title()}: {row[key]}"
             for key in keys if pd.notna(row.get(key))]
    return " | ".join(parts)

def safe_eval(value):
    try:
        if isinstance(value, str):
            return ast.literal_eval(value)
        return value if isinstance(value, dict) else {}
    except Exception as e:
        print(f"[safe_eval] Warning: failed to eval {value} with error {e}")
        return {}

def process_attributes(row_dict, flatten_nested=True):
    result = {}
    for key, value in row_dict.items():
        parsed = safe_eval(value)
        if isinstance(parsed, dict) and flatten_nested:
            true_keys = [k for k, v in parsed.items() if v is True]
            result[key] = ", ".join(true_keys) if true_keys else None
        else:
            result[key] = parsed
    return result

def combine_normalized_attributes(row):
    parts = []
    for col, value in row.items():
        if pd.isna(value):
            continue
        elif isinstance(value, bool):
            parts.append(f"{col}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

def collate_fn(batch):
    """
    Args:
        batch: List of sessions, each session is a list of POI dicts
    Returns:
        flat_pois: List[dict], all POIs flattened
        session_indices: List[List[int]], mapping each session to flat_pois indices
    """
    flat_pois = []
    session_indices = []
    idx = 0
    for session in batch:
        indices = []
        for poi in session:
            flat_pois.append(poi)
            indices.append(idx)
            idx += 1
        if len(indices) > 1:  # only keep sessions with >1 POI
            session_indices.append(indices)
    return flat_pois, session_indices

@torch.no_grad()
def load_or_generate_poi_embeddings(
    encoder,
    dataset,
    save_dir="../cache/poi_embeddings",
    modalities=None,
    device=None,
    batch_size=128,
    mode="static",  # "static" or "dynamic"
):
    """
    Load or generate POI embeddings with automatic projection dim inference.

    Args:
        encoder (nn.Module): Text encoder (e.g. with optional PEFT / LoRA).
        dataset (Dataset): POI dataset yielding dicts with ItemId and modality text.
        save_dir (str): Directory to cache embeddings.
        modalities (List[str], optional): Which modalities to use. Defaults to all except 'ItemId'.
        device (torch.device, optional): Compute device.
        batch_size (int): Encoding batch size.
        mode (str): 'static' (frozen) or 'dynamic' (trainable).

    Returns:
        embeddings (Dict[str, Tensor]), poi_ids (List[str])
    """
    assert mode in ["static", "dynamic"]
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # === Determine modalities ===
    sample = dataset[0]
    if isinstance(sample, tuple): sample = sample[0]
    if modalities is None:
        modalities = [k for k in sample.keys() if k != "ItemId"]

    # === Check if all embeddings are cached ===
    all_exist = all(
        os.path.exists(os.path.join(save_dir, f"{m}.pt")) for m in
        modalities)
    if all_exist:
        print("✅ Found cached embeddings. Loading from disk...")
        all_embeddings = {}
        for m in modalities:
            data = torch.load(os.path.join(save_dir, f"{m}.pt"),
                              map_location=device)
            all_embeddings[m] = data["embedding"]
        poi_ids = data["item_id"]
        return all_embeddings, poi_ids

    # === No cache: regenerate all ===
    print(f"No cache found. Generating {mode} embeddings...")
    encoder.eval()
    encoder = encoder.to(device)

    text_data = {m: [] for m in modalities}
    token_stats = {m: 0 for m in modalities}
    poi_ids = []
    tokenizer = encoder.tokenizer

    for poi in tqdm(dataset, desc="Collecting POI texts"):
        if isinstance(poi, tuple): poi = poi[0]
        for m in modalities:
            text = str(poi[m])
            text_data[m].append(text)
            token_stats[m] += len(tokenizer.tokenize(text))
        poi_ids.append(poi["ItemId"])

    # === Estimate dimension and set projection ===
    num_samples = len(poi_ids)
    avg_tokens = {m: token_stats[m] / num_samples for m in modalities}
    D_total = 6144
    token_sum = sum(avg_tokens.values())
    modality_dims = {
        m: max(1, round(D_total * avg_tokens[m] / token_sum))
        for m in modalities
    }

    print("Modality-wise avg token lengths:", avg_tokens)
    print("Auto-calculated projection dims:", modality_dims)

    # encoder.set_projection_dims(modality_dims)
    if hasattr(encoder, "set_projection_dims"):
        encoder.set_projection_dims(modality_dims)
    else:
        print(
            "[WARN] encoder has no set_projection_dims, skip dynamic dim allocation.")

    # === Encode & Save ===
    embeddings = {}
    for m in modalities:
        print(f"Encoding modality: {m}")
        modality_embeddings = []
        texts = text_data[m]

        for i in tqdm(range(0, len(texts), batch_size), desc=f"{m}"):
            batch_texts = texts[i:i + batch_size]
            batch_emb = encoder({m: batch_texts})[m].cpu()
            modality_embeddings.append(batch_emb)

        emb = torch.cat(modality_embeddings, dim=0)
        out_path = os.path.join(save_dir, f"{m}.pt")
        torch.save({"item_id": poi_ids, "embedding": emb}, out_path)
        embeddings[m] = emb

    print("✅ Embeddings saved to disk.")
    return embeddings, poi_ids

def save_poi_embeddings(poi_embeddings, save_dir, filename):
    """
    Save POI embeddings to CSV or CSV.GZ.

    Output columns:
      ItemId: POI id
      embedding: JSON string of the embedding vector

    Args:
        poi_embeddings (dict): {ItemId: np.ndarray or torch.Tensor [dim]}
        save_dir (str): directory to save
        filename (str): file name, can end with .csv or .csv.gz
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    records = []
    for item_id, emb in poi_embeddings.items():
        if emb is None:
            continue

        if isinstance(emb, torch.Tensor):
            vec = emb.detach().cpu().numpy().reshape(-1)
        else:
            vec = np.asarray(emb).reshape(-1)

        records.append(
            {
                "ItemId": item_id,
                "embedding": json.dumps(vec.tolist(), ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)

    if filename.endswith(".gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)

    print(f"Saved POI embeddings to {path}")

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