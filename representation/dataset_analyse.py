# inspect_n_from_pickle.py

import os
import numpy as np
from backbone.data.session_dataset import SessionDataset

DATASET_FILENAME = "../yelp/dataset/dataset.pickle"


def round_to_5(x: int) -> int:
    return int(5 * round(x / 5))


def recommend_N(lengths: np.ndarray, cap: int = 200) -> dict:
    p50 = int(np.percentile(lengths, 50))
    p75 = int(np.percentile(lengths, 75))
    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    mx = int(lengths.max())

    # 推荐策略：
    # 优先 p95，再做一个上限控制（避免 attention 爆炸）
    suggested = min(round_to_5(p95), cap)

    return {
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "suggested_N": suggested,
    }


def main():
    if not os.path.exists(DATASET_FILENAME):
        raise FileNotFoundError(DATASET_FILENAME)

    dataset = SessionDataset.from_pickle(DATASET_FILENAME)

    # 确保 input_data 已经存在
    if getattr(dataset, "input_data", None) is None:
        if hasattr(dataset, "load"):
            dataset.load()

    # 使用类内接口统计 session 长度
    counts = dataset.get_sample_counts()  # {session_id: length}
    lengths = np.array(list(counts.values()), dtype=np.int32)

    print("===== Session Length Statistics =====")
    print("Total sessions:", int(lengths.shape[0]))
    print("Min:", int(lengths.min()))
    print("Mean:", float(lengths.mean()))
    print("Median:", int(np.median(lengths)))
    print("Max:", int(lengths.max()))

    rec = recommend_N(lengths, cap=200)

    print("\n===== Percentiles & Recommendation =====")
    print(rec)

    N = rec["suggested_N"]
    frac_not_truncated = float((lengths <= N).mean())
    print(f"\nIf N = {N}, fraction of sessions NOT truncated: {frac_not_truncated:.4f}")


if __name__ == "__main__":
    main()