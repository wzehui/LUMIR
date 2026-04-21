import os
import torch

CACHE_DIR = "../cache/poi_embeddings"


def inspect_pt(path):
    print("=" * 80)
    print(f"File: {path}")

    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        print("Unexpected structure:", type(obj))
        return

    print("Keys:", list(obj.keys()))

    item_ids = obj.get("item_id", None)
    embeddings = obj.get("embedding", None)

    if item_ids is None or embeddings is None:
        print("Missing expected keys.")
        return

    print("Number of items:", len(item_ids))

    # 如果是 list
    if isinstance(embeddings, list):
        print("Embedding type: list")

        if len(embeddings) > 0:
            first = embeddings[0]

            if isinstance(first, list):
                print("Embedding dimension:", len(first))
            elif isinstance(first, torch.Tensor):
                print("Embedding dimension:", first.shape[0])
            else:
                print("Unknown embedding element type:", type(first))

    # 如果已经是 Tensor
    elif isinstance(embeddings, torch.Tensor):
        print("Embedding tensor shape:", embeddings.shape)

    else:
        print("Unknown embedding type:", type(embeddings))

    print("=" * 80)
    print()


def main():
    abs_dir = os.path.abspath(CACHE_DIR)
    print("Inspecting:", abs_dir)

    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".pt"):
            inspect_pt(os.path.join(CACHE_DIR, f))


if __name__ == "__main__":
    main()