from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backbone.utils.neural_utils import Cloze
from backbone.utils.neural_utils import (
    DataDescription)
from backbone.utils.neural_utils import (
    TensorFactory)
from backbone.utils.utils import INT_INF


class TestGenerator(Dataset):
    def __init__(
        self,
        test_data: Union[pd.DataFrame, Dict[int, np.ndarray]],
        N: int,
        data_description: DataDescription,
        for_prediction: bool,
    ) -> None:
        self.test_data = test_data
        self.N = N
        self.data_description = data_description
        self.for_prediction = for_prediction

        self.__generate_test_data()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.test_input, self.test_true

    def __generate_test_data(self):
        test_sequences = TensorFactory.to_sequence_tensor(self.test_data, self.N)

        if self.for_prediction:
            pad = torch.full((test_sequences.size(0), 1), INT_INF, dtype=test_sequences.dtype)
            test_sequences = torch.cat([test_sequences[:, 1:], pad], dim=1)

        num_items = self.data_description["num_items"]
        self.cloze = Cloze(num_items)

        test_data = []
        true_data = []

        train_loo_data, true_loo_data = self.cloze.mask_last(test_sequences)
        test_data.append(train_loo_data)
        true_data.append(true_loo_data)

        self.test_input = torch.cat(test_data, dim=0)
        self.test_true = torch.cat(true_data, dim=0)