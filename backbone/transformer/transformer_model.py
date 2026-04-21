import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from backbone.utils.config_util import extract_config
from backbone.transformer.custom_layers.embedding_layer import EmbeddingLayer
from backbone.utils.neural_utils.custom_layers.projection_head import ProjectionHead
from backbone.utils.neural_utils.custom_activations import to_activation
from backbone.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import masked_sparse_categorical_crossentropy


class TransformerModel(nn.Module, ABC):
    """The PyTorch base class for BERT4Rec-style transformer models."""

    def __init__(
        self,
        N: int,
        L: int,
        h: int,
        emb_dim: int,
        trans_dim_scale: int,
        drop_rate: float,
        activation: str,
        optimizer_kwargs: dict,
        transformer_layer_kwargs: dict,
        num_items: int,
    ) -> None:
        super().__init__()

        self.N = N
        self.L = L
        self.h = h
        self.emb_dim = emb_dim
        self.trans_dim_scale = trans_dim_scale
        self.drop_rate = drop_rate
        self.activation = to_activation(activation)
        self.optimizer_kwargs = optimizer_kwargs
        self.transformer_layer_kwargs = transformer_layer_kwargs
        self.num_items = num_items
        self.mask_target_used = self.num_items

        self.embedding_layer = self.get_embedding_layer()
        self.transformer = nn.Sequential(
            *[self.get_transformer_layer() for _ in range(self.L)]
        )
        self.projection_head = self.get_projection_head()

    def get_embedding_layer(self) -> nn.Module:
        return EmbeddingLayer(self.N, self.num_items, self.emb_dim)

    def get_projection_head(self) -> nn.Module:
        return ProjectionHead(
            self.emb_dim, self.embedding_layer.item_emb, self.activation
        )

    def get_optimizer(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)

    def compile_model(self) -> None:
        """Compiles the model."""
        optimizer = self.get_optimizer()
        self.compile(optimizer=optimizer, loss=masked_sparse_categorical_crossentropy)

    def get_config(self):
        return extract_config(self)

    @abstractmethod
    def get_transformer_layer(self) -> nn.Module:
        pass