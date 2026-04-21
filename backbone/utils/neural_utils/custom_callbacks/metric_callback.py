from typing import Union, Any
import numpy as np

from backbone.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)
from backbone.abstract_model import Model
import logging


class MetricCallback:
    def __init__(
        self,
        main_model: Model,
        metric_cls: type[RankingMetric],
        predict_data: Union[np.ndarray, dict[int, np.ndarray], None],
        ground_truths: dict[int, np.ndarray],
        top_k: int,
        prefix: str = "",
        dependencies: dict[MetricDependency, Any] = {},
        cores: int = 1,
    ):
        """
        PyTorch version of the metric callback used for validation metric tracking.

        This version preserves TensorFlow/Keras usage structure and parameter naming.
        """
        self.main_model = main_model
        self.metric_cls = metric_cls
        self.predict_data = predict_data
        self.ground_truths = ground_truths
        self.top_k = top_k
        self.prefix = prefix
        self.dependencies = dependencies
        self.cores = cores

        self.latest = None  # Most recent evaluation result
        self.main_model.is_trained = True  # Allow .predict() to work

    def on_epoch_end(self, epoch: int, logs: dict[str, float] = None):
        if logs is None:
            logs = {}

        predictions = self.main_model.predict(self.predict_data, self.top_k)

        result = self.metric_cls.eval(
            predictions=predictions,
            ground_truths=self.ground_truths,
            top_k=self.top_k,
            dependencies=self.dependencies,
            cores=self.cores,
        )

        # Log and store result
        temp_metric = self.metric_cls()
        temp_metric.top_k = self.top_k
        metric_name = f"{self.prefix}{temp_metric.name()}"

        logs[metric_name] = result
        self.latest = result

        if result > self.main_model.best_ndcg:
            self.main_model.best_ndcg = result
            self.best_epoch = epoch
            logging.info(f"✅ Metric improved at epoch {epoch + 1}. New best: {result:.4f}")