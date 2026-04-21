import logging
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import torch

from backbone.neural_model import NeuralModel
from backbone.utils.config_util import extract_config
from backbone.utils.id_reducer import IDReducer
from backbone.utils.neural_utils.custom_preprocessors.data_description import (
    DataDescription,
    get_data_description,
)
from backbone.utils.top_k_computer import TopKComputer
from backbone.utils.utils import INT_INF
from backbone.utils.utils import to_dense_encoding


class Transformer(NeuralModel):
    """The base Transformer class."""

    def __init__(
        self,
        N: int = None,
        L: int = 2,
        h: int = 2,
        emb_dim: int = 64,
        trans_dim_scale: int = 4,
        transformer_layer_kwargs: dict = {},
        drop_rate: float = 0.2,
        optimizer_kwargs: dict = {},
        infer_N_percentile: int = 95,
        pred_seen: bool = False,
        **neural_model_kwargs,
    ) -> None:
        super().__init__(**neural_model_kwargs)

        self.N = N
        self.L = L
        self.h = h
        self.emb_dim = emb_dim
        self.trans_dim_scale = trans_dim_scale
        self.transformer_layer_kwargs = transformer_layer_kwargs
        self.drop_rate = drop_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.infer_N_percentile = infer_N_percentile
        self.pred_seen = pred_seen

        self.id_reducer = None
        self.data_description = None
        self.model = None

        logging.info(
            f"Instantiating {self.name()} with configuration: {extract_config(self)}"
        )

    def train(self, input_data: pd.DataFrame) -> None:
        self.data_description: DataDescription = get_data_description(input_data)
        if self.N is None:
            self.N = int(
                self.data_description["session_length_description"][
                    f"{self.infer_N_percentile}%"
                ]
            )

        self.id_reducer = IDReducer(input_data, "ItemId")
        input_data = self.id_reducer.to_reduced(input_data)
        super().train(input_data)

    def get_recommendations_batched(
        self, predict_data: Dict[int, np.ndarray], top_k: int = 10
    ) -> Dict[int, np.ndarray]:
        predict_data = self.id_reducer.to_reduced(predict_data)

        test_data_tensor = self.get_test_generator(
            predict_data, for_prediction=True, batch_size=self.pred_batch_size
        )[0][0].to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(test_data_tensor)
            predictions = predictions_tensor.cpu().numpy()

        dense_sessions = to_dense_encoding(
            test_data_tensor.cpu().numpy(), predictions.shape[1],
            ignore_oob=True
        )
        allowed_items = 0 - dense_sessions * INT_INF
        if allowed_items.shape[1] != predictions.shape[1]:
            min_dim = min(allowed_items.shape[1], predictions.shape[1])
            allowed_items = allowed_items[:, :min_dim]
            predictions = predictions[:, :min_dim]

        predictions = np.add(predictions, allowed_items)

        predictions_top_k = TopKComputer.compute_top_k(predictions, top_k)
        key_to_predictions = dict(zip(predict_data.keys(), predictions_top_k))

        recommendations = self.id_reducer.to_original(key_to_predictions)
        return recommendations