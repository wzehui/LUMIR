import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backbone.utils.top_k_computer import TopKComputer
from backbone.data.side_information import SideInformation


class SideEncoder(nn.Module):
    def __init__(
        self,
        side_information: SideInformation,
        encoder_dimension: int,
        l2: float = 0.0,
        optimizer_kwargs: dict = {},
        is_verbose: bool = False,
    ):
        super().__init__()
        self.side_information = side_information
        self.num_non_cat = side_information["num_non_categorical_features"]
        self.num_cat = side_information["num_categorical_features"]
        self.cat_sizes = side_information["category_sizes"]
        self.is_verbose = is_verbose

        self.cat_embedders = nn.ModuleList()
        total_embedding_size = 0
        for cat_size in self.cat_sizes:
            emb_size = math.ceil(math.log(cat_size, 2))
            total_embedding_size += emb_size
            emb_layer = nn.Embedding(cat_size, emb_size)
            self.cat_embedders.append(emb_layer)

        input_size = self.num_non_cat + total_embedding_size
        encoder_dims, decoder_dims = self.get_hidden_architecture(input_size, encoder_dimension)

        self.encoder = self._build_mlp(input_size, encoder_dims + [encoder_dimension], l2)
        self.decoder = self._build_mlp(encoder_dimension, decoder_dims + [self.num_non_cat + sum(self.cat_sizes)], l2)

    def _build_mlp(self, in_dim, layer_dims, l2):
        layers = []
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.pop()  # remove last ReLU
        return nn.Sequential(*layers)

    def forward(self, x, decode=True):
        encoding = self.get_encodings_tensor(x)
        return self.decoder(encoding) if decode else encoding

    def get_encodings_tensor(self, x):
        non_cat = x[:, :self.num_non_cat]
        cat_vars = [x[:, self.num_non_cat + i].long() for i in range(self.num_cat)]
        cat_embs = [embed(cat_vars[i]) for i, embed in enumerate(self.cat_embedders)]
        emb_cat = torch.cat(cat_embs, dim=1)
        enc_input = torch.cat([non_cat, emb_cat], dim=1)
        return self.encoder(enc_input)

    def pretrain(self, num_epochs=3, batch_size=512):
        features = torch.tensor(self.side_information["features"], dtype=torch.float32)
        dataset = TensorDataset(features, features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters())

        for epoch in range(num_epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                pred = self.forward(x_batch)
                loss = self.side_encoder_loss(y_batch, pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.is_verbose:
                print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def get_encodings(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(features, dtype=torch.float32)
        return self.get_encodings_tensor(tensor).detach().cpu().numpy()

    def get_decodings(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(features, dtype=torch.float32)
        pred = self.forward(tensor).detach().cpu().numpy()
        sizes_pred = [self.num_non_cat, *self.cat_sizes]
        indices_pred = [sum(sizes_pred[:i+1]) for i in range(len(sizes_pred)-1)]
        pred_split = np.split(pred, indices_pred, axis=1)

        non_cat = pred_split[0]
        cat_preds = [TopKComputer.compute_top_k(p, 1) for p in pred_split[1:]]
        return np.concatenate([non_cat, *cat_preds], axis=1)

    def side_encoder_loss(self, y_true, y_pred):
        sizes_true = [self.num_non_cat] + [1 for _ in self.cat_sizes]
        sizes_pred = [self.num_non_cat] + self.cat_sizes

        true_split = torch.split(y_true, sizes_true, dim=1)
        pred_split = torch.split(y_pred, sizes_pred, dim=1)

        loss = F.mse_loss(pred_split[0], true_split[0])
        for true_cat, pred_cat in zip(true_split[1:], pred_split[1:]):
            true_cat = true_cat.squeeze(1).long()
            loss += F.cross_entropy(pred_cat, true_cat)
        return loss

    @staticmethod
    def get_hidden_architecture(input_size: int, encoder_dim: int) -> Tuple[list[int], list[int]]:
        hidden = []
        cur_dim = encoder_dim * 2
        while cur_dim <= input_size:
            hidden.append(cur_dim)
            cur_dim *= 2
        return list(reversed(hidden)), hidden