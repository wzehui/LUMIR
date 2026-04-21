import math
import numpy as np
import torch
from torch.utils.data import Dataset

from backbone.utils.neural_utils import TensorFactory
from backbone.utils.neural_utils import Cloze
from backbone.utils.neural_utils import DataDescription


class TrainGenerator(Dataset):
    def __init__(
        self,
        train_data,
        N: int,
        batch_size: int,
        mask_prob: float,
        data_description: DataDescription,
    ) -> None:
        self.train_data = train_data
        self.N = N
        self.batch_size = batch_size
        self.mask_prob = mask_prob

        num_items = data_description["num_items"]
        self.cloze = Cloze(num_items)

        self.on_epoch_end()

    def __len__(self):
        # use floor in pytorch
        return math.floor(len(self.train_input) / self.batch_size)

    def __getitem__(self, batch_index):
        start_index = batch_index * self.batch_size
        end_index = min((batch_index + 1) * self.batch_size, len(self.indices))  # 防止越界

        indices = self.indices[start_index:end_index]

        if len(indices) == 0:
            raise IndexError(f"[TrainGenerator] Requested empty batch at index {batch_index}")

        return self.train_input[indices], self.train_true[indices]

    def on_epoch_end(self):
        self.__generate_train_data()
        self.indices = np.arange(len(self.train_input))
        np.random.shuffle(self.indices)

    def __generate_train_data(self):
        sequences = TensorFactory.to_sequence_tensor(
            sessions=self.train_data,
            sequence_length=self.N,
        )

        train_data = []
        true_data = []

        # Add one last-item-masked version
        train_loo_data, true_loo_data = self.cloze.mask_last(sequences)
        train_data.append(train_loo_data)
        true_data.append(true_loo_data)

        # Add ten random-masked versions
        for _ in range(10):
            train_random_data, true_random_data = self.cloze.mask_random(
                sequences, self.mask_prob
            )
            train_data.append(train_random_data)
            true_data.append(true_random_data)

        self.train_input = torch.cat(train_data, dim=0)
        self.train_true = torch.cat(true_data, dim=0)