import torch
import torch.nn as nn
from enum import Enum
from backbone.utils.neural_utils.custom_activations import to_activation


class TransformerEncoderLayerLayout(str, Enum):
    FDRN = 0
    NFDR = 1

    @staticmethod
    def from_str(label):
        label = label.lower()
        if label == "fdrn":
            return TransformerEncoderLayerLayout.FDRN
        elif label == "nfdr":
            return TransformerEncoderLayerLayout.NFDR
        else:
            raise ValueError(f"Unknown transformer encoder layout: {label}")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        intm_dim: int,
        num_heads: int,
        dropout: float,
        use_causal_mask: bool,
        activation: str,
        layer_norm_epsilon: float = 1e-5,
        layout: TransformerEncoderLayerLayout = TransformerEncoderLayerLayout.FDRN,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.intm_dim = intm_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_causal_mask = use_causal_mask
        self.activation = to_activation(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layout = (
            layout
            if isinstance(layout, TransformerEncoderLayerLayout)
            else TransformerEncoderLayerLayout.from_str(layout)
        )

        # Multi-head attention now uses correct embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.att_norm = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.ff_norm = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)

        self.att_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        # Feedforward layer: project to intm_dim, then back to embed_dim
        self.ff_intermediate = nn.Linear(embed_dim, intm_dim)
        self.ff_output = nn.Linear(intm_dim, embed_dim)

    def _mha(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        return attn_output

    def _ff(self, x):
        x = self.activation(self.ff_intermediate(x))
        return self.ff_output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == TransformerEncoderLayerLayout.FDRN:
            attn_out = self.att_dropout(self._mha(x))
            x = self.att_norm(x + attn_out)

            ff_out = self.ff_dropout(self._ff(x))
            x = self.ff_norm(x + ff_out)

        elif self.layout == TransformerEncoderLayerLayout.NFDR:
            x_norm = self.att_norm(x)
            attn_out = self.att_dropout(self._mha(x_norm))
            x = x + attn_out

            x_norm = self.ff_norm(x)
            ff_out = self.ff_dropout(self._ff(x_norm))
            x = x + ff_out

        return x