import os
import torch
import pandas as pd
import json

def modality2embedding(config, filename="baseline_concatenation.csv.gz"):
    """
    Merge all modality-specific .pt files into a single baseline concatenated embedding file,
    and save it as a .csv.gz under the directory config['paths']['representation'].

    Args:
        config (dict): Configuration dictionary with a 'paths' section.
        filename (str): Output filename (default: 'baseline_concatenation.csv.gz').
    """
    pt_dir = config['paths']['embedding']
    output_dir = config['paths']['representation']
    output_path = os.path.join(output_dir, filename)

    # Collect all .pt files in sorted order
    pt_files = sorted([os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")])
    assert pt_files, f"No .pt files found in directory: {pt_dir}"

    print(f"Found {len(pt_files)} embedding files. Starting merge...")

    item_ids = None
    all_embeddings = []

    for path in pt_files:
        data = torch.load(path, map_location="cpu")
        ids = data["item_id"]
        emb = data["embedding"]  # Tensor [N, D]

        # Ensure consistent order of ItemIds across modalities
        if item_ids is None:
            item_ids = ids
        else:
            assert ids == item_ids, f"ItemId mismatch in {path}"

        all_embeddings.append(emb)

    # Concatenate all modality embeddings into a single tensor
    merged_embedding = torch.cat(all_embeddings, dim=1)
    print(f"✅ Merged embedding shape: {merged_embedding.shape}")

    # Prepare records for saving as DataFrame
    records = [{"ItemId": id, "embedding": json.dumps(vec.tolist())}
               for id, vec in zip(item_ids, merged_embedding)]

    df = pd.DataFrame.from_records(records)

    # Create output directory and save as .csv.gz
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"✅ Saved to {output_path}")