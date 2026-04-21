import numpy as np
import torch
from typing import Tuple

from backbone.utils.neural_utils import (
    TensorFactory)


class Cloze:
    """Cloze handles masking sequences for training, either randomly or at the last item."""

    def __init__(self, mask_target: int) -> None:
        self.mask_target = mask_target

    def mask_random(self, data: torch.Tensor, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.from_numpy(np.random.binomial(1, mask_prob, data.shape)).to(data.device)
        return self.__convert_train(data, mask)

    def mask_last(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros_like(data)
        mask[:, -1] = 1
        return self.__convert_train(data, mask)

    def __convert_train(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        non_padding_entries = (data != TensorFactory.PADDING_TARGET).int()
        mask = mask * non_padding_entries

        new_train_targets = mask * self.mask_target
        new_true_targets = (1 - mask) * TensorFactory.PADDING_TARGET

        train_masked = data * (1 - mask) + new_train_targets
        true_masked = data * mask + new_true_targets
        return train_masked, true_masked