from backbone.eval.metrics.mrr import MeanReciprocalRank
from backbone.eval.metrics.hitrate import HitRate
from backbone.eval.metrics.serendipity import Serendipity
from backbone.eval.metrics.novelty import Novelty
from backbone.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from backbone.eval.metrics.catalog_coverage import CatalogCoverage
from backbone.eval.metrics.gini import CatalogGiniIndex
from backbone.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)

# =========================
# Default metric sets
# =========================

ALL_DEFAULT = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
    CatalogCoverage(),
    CatalogGiniIndex(),
    Serendipity(),
    Novelty(),
]

BEYOND_ACCURACY = [
    Serendipity(),
    Novelty(),
    CatalogCoverage(),
    CatalogGiniIndex(),
]

ALL_RANKING = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
]

ALL = [
    NormalizedDiscountedCumulativeGain(),
    HitRate(),
    MeanReciprocalRank(),
    CatalogCoverage(),
    CatalogGiniIndex(),
    Serendipity(),
    Novelty(),
]