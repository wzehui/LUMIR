import math
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch

from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import TensorFactory


class NextItemTrainGenerator:
    def __init__(self, train_data: Union[pd.DataFrame, dict], N: int, batch_size: int) -> None:
        """
        PyTorch replica of the TensorFlow NextItemTrainGenerator.

        - Builds a fixed-length [S, N+1] sequence tensor with left padding.
        - Splits into train_input = [:, :-1] and train_true = [:, 1:].
        - Critically: mirrors TF behavior by setting labels to PADDING_TARGET
          where the corresponding input positions are padding.
        - Returns (x, y) as [B, T] tensors (no flattening here).
        """
        self.train_data = train_data
        self.N = N
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self) -> int:
        # Number of batches per epoch
        return math.ceil(self.train_input.shape[0] / self.batch_size)

    def __getitem__(self, batch_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a batch (train_input [B, T], train_true [B, T]).
        Uses torch.index_select to keep device/dtype consistent.
        """
        start = batch_index * self.batch_size
        end = (batch_index + 1) * self.batch_size
        idx_np = self.indices[start:end]
        idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=self.train_input.device)

        x = self.train_input.index_select(0, idx_t)  # [B, T]
        y = self.train_true.index_select(0, idx_t)   # [B, T]
        return x, y

    def __iter__(self):
        """Explicit iterator to make iteration robust and finite."""
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self) -> None:
        """
        Rebuild tensors and shuffle row indices.

        TF parity:
        - Build sequence tensor with length N+1
        - train_input = seq[:, :-1], train_true = seq[:, 1:]
        - Overwrite labels with PADDING_TARGET where train_input is padding
          (exactly what TF's tf.where does).
        """
        # [S, N+1] numpy -> torch.LongTensor (CPU by default)
        seq = TensorFactory.to_sequence_tensor(
            sessions=self.train_data,
            sequence_length=self.N + 1,
        )
        seq = torch.as_tensor(seq, dtype=torch.long)

        # Split into input/labels
        self.train_input = seq[:, :-1].contiguous()   # [S, N]
        self.train_true  = seq[:,  1:].contiguous()   # [S, N]

        # Mirror TF: set labels to padding where input is padding
        pad_id = TensorFactory.PADDING_TARGET
        pad_t  = torch.as_tensor(pad_id, dtype=self.train_input.dtype, device=self.train_input.device)
        padding_mask = self.train_input.eq(pad_t)     # [S, N] bool
        self.train_true = torch.where(padding_mask, pad_t, self.train_true)

        # Shuffle along the batch dimension
        self.indices = np.arange(self.train_input.shape[0])
        np.random.shuffle(self.indices)