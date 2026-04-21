import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from backbone.grurec.grurec_model import GRURecModel
from backbone.neural_model import NeuralModel
from backbone.utils.config_util import extract_config
from backbone.utils.id_reducer import IDReducer
from backbone.utils.neural_utils.custom_generators.next_item_test_generator import (
    NextItemTestGenerator,
)
from backbone.utils.neural_utils.custom_generators.next_item_train_generator import (
    NextItemTrainGenerator,
)
from backbone.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import (
    masked_sparse_categorical_crossentropy,
)
from backbone.utils.neural_utils.custom_preprocessors.data_description import *
from backbone.utils.top_k_computer import TopKComputer
from backbone.utils.utils import INT_INF
from backbone.utils.utils import to_dense_encoding


class GRURec(NeuralModel):
    def __init__(
        self,
        N: int = None,
        infer_N_percentile: int = 95,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        drop_rate: float = 0.2,
        optimizer_kwargs: dict = {},
        pred_seen: bool = False,
        **neural_model_kwargs: dict,
    ):
        """The GRU4Rec model.

        The implementation mostly follows the original GRU4Rec paper.
        """
        super().__init__(**neural_model_kwargs)

        self.N = N
        self.infer_N_percentile = infer_N_percentile
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.pred_seen = pred_seen

        logging.info(
            f"Instantiating GRU4Rec with configuration: {extract_config(self)}"
        )

    def train(self, input_data: pd.DataFrame) -> None:
        """Trains the GRU4Rec model given the training data."""
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
        """Generates predictions for the test sessions given."""
        predict_data = self.id_reducer.to_reduced(predict_data)

        test_generator = self.get_test_generator(
            predict_data, for_prediction=True, batch_size=self.pred_batch_size
        )

        # In our NextItemTestGenerator(for_prediction=True) we guarantee a single batch
        assert len(test_generator) == 1
        test_data_tensor = test_generator[0][0]

        # --- NEW: put model in eval mode for deterministic inference ---
        self.model.to(self.device)
        self.model.eval()

        # --- Ensure correct dtype/device for embedding lookup ---
        test_data_tensor = test_data_tensor.to(self.device).long()

        with torch.no_grad():
            outputs = self.model(test_data_tensor, training=False)  # [B, V]
            predictions = (
                outputs.detach().cpu().numpy()
                if isinstance(outputs, torch.Tensor)
                else np.asarray(outputs)
            )

        # Build dense “seen-items” mask from the ORIGINAL input ids (padding intact)
        space_size = self.data_description["num_items"]
        dense_sessions = to_dense_encoding(
            test_data_tensor.detach().cpu().numpy(),
            # move back to CPU for numpy ops
            space_size,
            ignore_oob=True
        )

        if not self.pred_seen:
            allowed_items = 0 - dense_sessions * INT_INF
            predictions = np.add(predictions, allowed_items)

        predictions_top_k = TopKComputer.compute_top_k(predictions, top_k)

        # --- NEW: map back using a stable key order that matches the generator output ---
        # safest is to reuse the order the generator emitted; if your generator does not
        # return the keys, fall back to insertion order but assert row counts match.
        key_list = list(predict_data.keys())
        if len(key_list) != predictions_top_k.shape[0]:
            # This can happen if generator dropped empty sessions. Filter keys accordingly.
            # Keep only sessions with length > 0 like the generator did.
            # If your NextItemTestGenerator already exposes valid indices/keys, use that instead.
            # Minimal fallback: truncate to min length to avoid misalignment.
            n = min(len(key_list), predictions_top_k.shape[0])
            key_list = key_list[:n]
            predictions_top_k = predictions_top_k[:n]

        key_to_predictions = dict(zip(key_list, predictions_top_k))
        recommendations = self.id_reducer.to_original(key_to_predictions)
        return recommendations

    # === Changed only this function name and implementation ===
    def get_model(self, data: dict[int, np.ndarray]) -> torch.nn.Module:
        """Build and return the PyTorch version of GRU4Rec model."""
        num_items = self.data_description["num_items"]
        return GRURecModel(
            num_items,
            self.emb_dim,
            self.hidden_dim,
            self.drop_rate,
            self.optimizer_kwargs,
            self.activation,
        )

    def get_train_generator(self, data: pd.DataFrame, batch_size: int):
        return NextItemTrainGenerator(data, self.N, batch_size)

    def get_test_generator(self, data: pd.DataFrame, for_prediction: bool, batch_size: int):
        return NextItemTestGenerator(
            data, self.N, batch_size, for_prediction=for_prediction
        )

    def name(self) -> str:
        return "GRU4Rec"