import torch
import torch.nn as nn


class BiasLayer(nn.Module):
    """
    PyTorch equivalent of the TensorFlow BiasLayer.
    Adds a trainable bias to the last dimension(s) of the input tensor,
    skipping batch and sequence dimensions.
    """

    def __init__(self, bias_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(bias_dim))

    def forward(self, x):
        return x + self.bias