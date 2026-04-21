import math
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch

from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class NextItemTestGenerator:
    def __init__(
        self,
        sessions: Union[pd.DataFrame, dict[int, np.ndarray]],
        N: int,
        batch_size: int,
        for_prediction: bool,
    ) -> None:
        """
        PyTorch replica of the TensorFlow NextItemTestGenerator.

        TF parity:
        - Prediction: test_input = seq[:, 1:], test_true = None
        - Validation: test_input = seq[:, :-1], test_true = concat(padding=-1, last_col)
          where test_true has the SAME SHAPE as test_input: [S, N]
        - __len__ = ceil(S / batch_size) in BOTH modes (do not force single batch).
        """
        self.N = N
        self.batch_size = batch_size
        self.for_prediction = for_prediction

        self._generate_test_data(sessions)

    def __len__(self) -> int:
        # Same batching rule as TF: ceil(num_rows / batch_size)
        return math.ceil(self.test_input.shape[0] / self.batch_size)

    def __getitem__(self, batch_index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return (cur_test_input, cur_test_true) slice by batch_index.
        - cur_test_true is None in prediction mode, a [B, N] LongTensor in validation mode.
        """
        start_index = batch_index * self.batch_size
        end_index = (batch_index + 1) * self.batch_size

        cur_test_input = self.test_input[start_index:end_index]

        if self.test_true is not None:
            cur_test_true = self.test_true[start_index:end_index]
        else:
            cur_test_true = None

        return (cur_test_input, cur_test_true)

    def __iter__(self):
        """Explicit iterator to make iteration robust and finite."""
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self) -> None:
        # No shuffling in TF version; keep parity.
        pass

    def _generate_test_data(
        self, sessions: Union[pd.DataFrame, dict[int, np.ndarray]]
    ) -> None:
        """
        Build self.test_input and self.test_true to mirror the TF implementation:
        - Build sequence tensor of length N+1 using TensorFactory.to_sequence_tensor
        - Prediction:
            test_input = seq[:, 1:]        # [S, N]
            test_true  = None
        - Validation:
            test_input       = seq[:, :-1] # [S, N]
            test_true_column = seq[:, -1:] # [S, 1]
            test_true_padding = full([S, N-1], fill_value=-1)
            test_true = concat([test_true_padding, test_true_column], dim=-1)  # [S, N]
        """
        # TensorFactory returns a numpy/int array; convert to torch.LongTensor
        seq_np = TensorFactory.to_sequence_tensor(
            sessions=sessions,
            sequence_length=self.N + 1,
        )
        sequence_tensor = torch.as_tensor(seq_np, dtype=torch.long)

        if self.for_prediction:
            # Prediction phase: take last N items; labels unused.
            self.test_input = sequence_tensor[:, 1:]   # [S, N]
            self.test_true = None
        else:
            # Validation phase: leave out last item from input.
            self.test_input = sequence_tensor[:, :-1]  # [S, N]

            # Labels: only last position holds the target; the others are -1.
            # Match TF's tf.fill and tf.concat behavior.
            S, N = self.test_input.shape
            device = self.test_input.device
            dtype = self.test_input.dtype

            test_true_column = sequence_tensor[:, -1:].to(device=device, dtype=dtype)  # [S, 1]
            if N - 1 > 0:
                test_true_padding = torch.full(
                    (S, N - 1), fill_value=-1, dtype=dtype, device=device
                )
                self.test_true = torch.cat([test_true_padding, test_true_column], dim=-1)  # [S, N]
            else:
                # Edge case: N==1 -> padding part has width 0, so labels are exactly the last column
                self.test_true = test_true_column  # [S, 1] which equals [S, N] when N==1