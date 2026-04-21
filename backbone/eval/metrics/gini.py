from backbone.eval.metrics.metric import RankingMetric, MetricDependency
import numpy as np
from typing import List, Dict, Optional


class CatalogGiniIndex(RankingMetric):
    """
    Gini index over the item exposure distribution induced by the recommender.

    For each evaluation, we count how many times each item appears in predictions
    (across all samples), then compute the Gini coefficient on that count vector.

    Range:
      0.0 -> perfectly uniform exposure
      ~1.0 -> extremely unequal exposure
    """

    def __init__(self):
        super().__init__()
        self.num_items_total: Optional[int] = None

    def get_required_dependencies(self) -> List[MetricDependency]:
        # Needed to know how many items exist in the catalog so we can include
        # items that never appear (count = 0) in the Gini computation.
        return super().get_required_dependencies() + [MetricDependency.NUM_ITEMS]

    def state_init(self) -> Dict[int, int]:
        # Total catalog size (all recommendable items)
        self.num_items_total = int(self.get_dependency(MetricDependency.NUM_ITEMS))
        # exposure counts: item_id -> count
        return {}

    def eval_sample(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        intersect: np.ndarray,
        sample_id: int,
    ) -> Dict[int, int]:
        cnt: Dict[int, int] = {}
        for it in predictions.tolist():
            cnt[it] = cnt.get(it, 0) + 1
        return cnt

    def eval_bulk(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        intersect: List[np.ndarray],
        sample_ids: np.ndarray,
    ) -> Dict[int, int]:
        cnt: Dict[int, int] = {}
        if len(predictions) == 0:
            return cnt
        all_pred = np.concatenate(predictions).tolist()
        for it in all_pred:
            cnt[it] = cnt.get(it, 0) + 1
        return cnt

    def state_merge(self, current: Dict[int, int], to_add: Dict[int, int]) -> Dict[int, int]:
        for k, v in to_add.items():
            current[k] = current.get(k, 0) + int(v)
        return current

    def state_merge_bulk(self, current: Dict[int, int], to_add: Dict[int, int]) -> Dict[int, int]:
        return self.state_merge(current, to_add)

    @staticmethod
    def _gini_from_counts(counts: np.ndarray) -> float:
        """
        Compute Gini coefficient for a non-negative vector.
        Uses the common sorted-formula:
          G = (2 * sum_i i*x_i) / (n * sum_x) - (n + 1) / n
        where i is 1..n after sorting ascending.
        """
        x = np.asarray(counts, dtype=np.float64)
        if x.size == 0:
            return 0.0
        if np.any(x < 0):
            raise ValueError("Gini requires non-negative counts.")
        s = x.sum()
        if s <= 0:
            return 0.0

        x_sorted = np.sort(x)  # ascending
        n = x_sorted.size
        i = np.arange(1, n + 1, dtype=np.float64)
        g = (2.0 * np.sum(i * x_sorted)) / (n * s) - (n + 1.0) / n
        # numerical safety
        if g < 0:
            g = 0.0
        if g > 1:
            g = 1.0
        return float(g)

    def state_finalize(self, current: Dict[int, int]) -> float:
        # If the framework already set self.num_items, prefer it; else use dependency.
        n = int(self.num_items_total or getattr(self, "num_items", 0) or 0)
        if n <= 0:
            return 0.0

        # Build full exposure vector including items never recommended (count=0).
        # Note: we don't know item IDs for missing ones, but for Gini we only need
        # the multiset of counts, so we can just append zeros to reach length n.
        observed_counts = np.fromiter(current.values(), dtype=np.int64)
        missing = n - observed_counts.size
        if missing > 0:
            counts = np.concatenate([observed_counts, np.zeros(missing, dtype=np.int64)])
        else:
            counts = observed_counts[:n]  # safety if somehow bigger

        return self._gini_from_counts(counts)

    def per_sample(self) -> bool:
        return False

    def name(self) -> str:
        return f"GiniIndex@{self.top_k}"