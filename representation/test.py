import pandas as pd
import json

# 设置两个文件路径
file1 = "../yelp/embeddings/total_no_neighbor_embeddings_embeddings_openai.csv.gz"
file2 = "../yelp/embeddings/comm_epoch5.csv.gz"

print("📂 Reading:", file1)
df1 = pd.read_csv(file1, compression="gzip")

print("📂 Reading:", file2)
df2 = pd.read_csv(file2, compression="gzip")

# 打印列名
print("\n🔍 Columns in file1:", list(df1.columns))
print("🔍 Columns in file2:", list(df2.columns))

# 检查是否包含 "embedding" 和 "ItemId"
if "embedding" not in df1.columns or "embedding" not in df2.columns:
    print("❌ One or both files do not contain an 'embedding' column.")
else:
    # 尝试解析第一个 embedding 样本
    try:
        emb1 = json.loads(df1["embedding"].iloc[0])
        emb2 = json.loads(df2["embedding"].iloc[0])
        dim1 = len(emb1)
        dim2 = len(emb2)

        print("\n✅ Sample embedding loaded.")
        print(f"📐 Embedding dimension in file1: {dim1}")
        print(f"📐 Embedding dimension in file2: {dim2}")

        if dim1 == dim2:
            print("✅ Embedding dimensions match.")
        else:
            print("⚠️ Embedding dimensions differ!")

    except Exception as e:
        print("❌ Failed to parse embeddings:", e)

# 可选：比较前几个 ItemId 是否一致
if "ItemId" in df1.columns and "ItemId" in df2.columns:
    id_set_1 = set(df1["ItemId"].head(5))
    id_set_2 = set(df2["ItemId"].head(5))
    print("\n🆔 First 5 ItemIds in file1:", id_set_1)
    print("🆔 First 5 ItemIds in file2:", id_set_2)

    if id_set_1 == id_set_2:
        print("✅ First few ItemIds match.")
    else:
        print("⚠️ First few ItemIds do NOT match.")