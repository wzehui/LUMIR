import itertools
import logging
import math
from abc import abstractmethod
from typing import Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from tenacity import retry, stop_after_attempt, wait_random
from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory)
from backbone.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import masked_sparse_categorical_crossentropy

from backbone.abstract_model import Model


class NeuralModel(Model):
    def __init__(
        self,
        num_epochs: int = 100,
        fit_batch_size: int = 256,
        pred_batch_size: int = 1024,
        train_val_fraction: float = 0.1,
        early_stopping_patience: int = 2,
        activation: str = "gelu",
        filepath_weights=None,
        **model_kwargs,
    ) -> None:
        self.grad_clip_norm = model_kwargs.pop("clipnorm", None)
        super().__init__(**model_kwargs)

        self.num_epochs = num_epochs
        self.fit_batch_size = fit_batch_size
        self.pred_batch_size = pred_batch_size
        self.train_val_fraction = train_val_fraction
        self.early_stopping_patience = early_stopping_patience
        self.activation = activation
        self.filepath_weights = filepath_weights
        self.input_data = None
        self.history: dict[str, float] = {}
        self.active_callbacks = []
        self.best_ndcg = float("-inf")
        self.grad_clip_norm = model_kwargs.pop("clipnorm", None)
        self.optimizer_kwargs = model_kwargs.pop("optimizer_kwargs", {})


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(
            f"Initializing neural model. The following device will be used by PyTorch: {self.device}"
        )

    def train(self, input_data: Union[csr_matrix, pd.DataFrame]) -> None:
        self.input_data: Union[csr_matrix, pd.DataFrame] = input_data
        if isinstance(self.input_data, csr_matrix):
            self.num_samples = input_data.shape[0]
        else:
            self.num_samples = len(input_data)
        if self.fit_batch_size > self.num_samples:
            logging.warning(
                f"Batch size {self.fit_batch_size} larger than the number of samples "
                f"{self.num_samples}. Using number of samples as batch size instead."
            )
            self.fit_batch_size = self.num_samples
        self.model = self.get_model(self.input_data)
        self.model.to(self.device)

        # for name, param in self.model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        if self.filepath_weights is not None:
            self.model.load_state_dict(torch.load(self.filepath_weights))
            self.is_trained = True
            return
        # np.random.seed(2025)
        train_data, val_data = self.split_session_data(self.input_data, self.train_val_fraction)
        train_generator = self.get_train_generator(train_data, self.fit_batch_size)

        # Prepare session-wise evaluation data
        ground_truths = (
            val_data.copy()
            .drop_duplicates(subset=["SessionId"], keep="last")
            .groupby("SessionId")["ItemId"]
            .apply(np.array)
            .to_dict()
        )
        predict_data = (
            val_data.copy()[val_data.duplicated(["SessionId"], keep="last")]
            .groupby("SessionId")["ItemId"]
            .apply(np.array)
            .to_dict()
        )
        ground_truths = self.id_reducer.to_original(ground_truths)
        predict_data = self.id_reducer.to_original(predict_data)

        from backbone.eval.metrics.ndcg import (
            NormalizedDiscountedCumulativeGain,
        )
        from backbone.utils.neural_utils.custom_callbacks.metric_callback import (
            MetricCallback,
        )
        from backbone.utils.neural_utils.custom_earlystop.early_stop import (
            EarlyStopping
        )
        # Setup callbacks
        metric_callback = MetricCallback(
            self,
            NormalizedDiscountedCumulativeGain,
            predict_data,
            ground_truths,
            top_k=20,
            prefix="inner_val_",
        )
        early_stop = EarlyStopping(
            monitor="inner_val_NDCG@20",
            min_delta=0.0005,
            patience=self.early_stopping_patience,
            verbose=self.is_verbose,
            mode="max",
            restore_best_weights=True,
        )

        def fit_model(x, epochs, callbacks, verbose):
            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         **self.optimizer_kwargs)
            loss_fn = masked_sparse_categorical_crossentropy

            # Ensure model and tensors are placed on the same device (safe and idempotent)
            self.model.to(self.device)

            metric_callback, early_stop, *extra_callbacks = callbacks

            for epoch in range(epochs):
                self.model.train()
                epoch_losses = []

                from tqdm import tqdm
                pbar = tqdm(x, desc=f"Epoch {epoch + 1}/{epochs}")

                for batch in pbar:
                    # The generator yields (x: [B, T], y: [B, T]) — identical to TensorFlow behavior
                    input_tensor, target_tensor = batch
                    input_tensor = input_tensor.to(self.device)
                    target_tensor = target_tensor.to(self.device)

                    optimizer.zero_grad()

                    # Force GRU models into “training branch” (flatten all non-padding positions)
                    # BERT models do not use 'training' flag in forward()
                    try:
                        preds = self.model(input_tensor,
                                           training=True)  # [N, V]
                    except TypeError:
                        preds = self.model(input_tensor)  # [N, V] for BERT

                    # # === 🔍 DEBUG INSPECTION BLOCK (insert here) ===
                    # if epoch == 0 and len(
                    #         epoch_losses) < 5:  # only print first few batches for sanity check
                    #     pad = (input_tensor == TensorFactory.PADDING_TARGET)
                    #     n_nonpad = int((~pad).sum().item())
                    #     n_mask = int((input_tensor == getattr(self.model,
                    #                                           "mask_target_used",
                    #                                           -999)).sum().item())
                    #     print(
                    #         f"[DBG] preds_rows={preds.shape[0]}  n_nonpad={n_nonpad}  n_mask={n_mask}")

                    # ====== Auto-select label alignment strategy ======
                    # Candidate 1 (BERT-style): labels at masked positions only
                    y_masked = None
                    if hasattr(self.model, "mask_target_used"):
                        mask_id = self.model.mask_target_used
                        rel_bert = (
                                    input_tensor == mask_id)  # [B, T] boolean mask
                        y_masked = target_tensor[rel_bert]  # [N₁]

                    # Candidate 2 (GRU-style): labels at all non-padding positions
                    pad_id = TensorFactory.PADDING_TARGET
                    rel_gru = (input_tensor != pad_id)  # [B, T] boolean mask
                    y_all = target_tensor[rel_gru]  # [N₂]

                    # Select whichever matches the number of predictions; otherwise skip the batch
                    y = None
                    if y_masked is not None and y_masked.numel() == preds.shape[
                        0]:
                        y = y_masked
                    elif y_all.numel() == preds.shape[0]:
                        y = y_all

                    # Skip invalid or empty batches (e.g., all-padding sequences)
                    if y is None or y.numel() == 0 or preds.shape[0] == 0:
                        continue

                    # Compute loss — preds: [N, V], y: [N]
                    loss = loss_fn(preds, y)
                    loss.backward()

                    # Gradient clipping (optional, if grad_clip_norm is set)
                    if getattr(self, "grad_clip_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.grad_clip_norm)

                    optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss.item():.2f}")

                # ====== Validation and early stopping ======
                logs = {}
                metric_callback.on_epoch_end(epoch, logs=logs)
                metric_value = metric_callback.latest

                if verbose:
                    logging.info(
                        f"[Epoch {epoch + 1}] {metric_callback.prefix}NDCG@{metric_callback.top_k}: {metric_value:.4f}"
                    )

                if early_stop.on_epoch_end(metric_value, model=self.model):
                    break

            logging.info(f"Training complete. Best NDCG@20: {self.best_ndcg}")

        fit_model(
            x=train_generator,
            epochs=self.num_epochs,
            callbacks=[metric_callback, early_stop, *self.active_callbacks],
            verbose=2 if self.is_verbose else 0,
        )
        self.is_trained = True

    def predict(
        self,
        predict_data: Union[np.ndarray, dict[int, np.ndarray]] = None,
        top_k: int = 10,
    ) -> dict[Any, np.ndarray]:
        assert self.is_trained
        if predict_data is None:
            predict_data = np.arange(self.num_samples)

        # Get number of samples to predict.
        num_samples = len(predict_data)

        num_batches = math.ceil(num_samples / self.pred_batch_size)

        # Create batches
        i = itertools.cycle(range(num_batches))
        batches = [dict() for _ in range(num_batches)]
        for k, v in predict_data.items():
            batches[next(i)][k] = v

        recommendations = {}

        # Calculate recommendations for batches.
        for i, cur_batch in enumerate(batches):
            # Log if verbose and enough time has passed since last log.
            logging.info(f"Predicting batch {i} out of {len(batches)}")
            recommendations_batch = self.get_recommendations_batched(cur_batch, top_k)
            recommendations.update(recommendations_batch)

        return recommendations

    @abstractmethod
    def get_recommendations_batched(
        self, user_batch: np.ndarray, top_k
    ) -> dict[int, np.ndarray]:
        pass

    @abstractmethod
    def get_model(self,
                  data: Union[csr_matrix, pd.DataFrame]) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_train_generator(
            self, data: Union[csr_matrix, pd.DataFrame], batch_size: int
    ) -> DataLoader:
        pass

    def split_session_data(
        self, data: pd.DataFrame, train_val_fraction: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits session train data into a train DataFrame and a validation DataFrame
        based on a train-validation split conveyed as a fraction of the overall train
        data.

        We use a random subset of rows of the data for validation.

        Args:
            data (pd.DataFrame): The training data.
            train_val_fraction (float): Fraction of train data to be used for validation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, val_data).
        """
        unique_sessions = data["SessionId"].unique()

        num_sessions_val = min(
            math.ceil(train_val_fraction * len(unique_sessions)), 500
        )

        sessions_val = set(
            np.random.choice(unique_sessions, size=num_sessions_val, replace=False)
        )
        sessions_train = set([id for id in unique_sessions if id not in sessions_val])

        data_1 = data[data["SessionId"].isin(sessions_train)]
        data_2 = data[data["SessionId"].isin(sessions_val)]

        return (data_1, data_2)