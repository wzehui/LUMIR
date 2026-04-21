import os
import json
import pandas as pd
import numpy as np

# FILE_PATH = "../instagram/embeddings/embedding_meta-review-photo_768_2048_ins.csv.gz"
FILE_PATH = "../yelp/embeddings/embedding_meta-review-photo_768_2048.csv.gz"

def main():
    if not os.path.exists(FILE_PATH):
        print("File not found:", FILE_PATH)
        return

    print("Loading file:", FILE_PATH)
    df = pd.read_csv(FILE_PATH, compression="gzip")

    print("Number of rows:", len(df))

    if "embedding" not in df.columns:
        print("No 'embedding' column found.")
        return

    dims = []
    for i, row in df.iterrows():
        vec = json.loads(row["embedding"])
        dims.append(len(vec))

        # 只打印第一条向量的前5维
        if i == 0:
            print("First embedding first 5 dims:", vec[:5])

    dims = np.array(dims)
    unique_dims = np.unique(dims)

    print("Unique embedding dimensions found:", unique_dims)

    if len(unique_dims) == 1:
        print("All embeddings have consistent dimension:", unique_dims[0])
    else:
        print("Dimension mismatch detected.")

if __name__ == "__main__":
    main()