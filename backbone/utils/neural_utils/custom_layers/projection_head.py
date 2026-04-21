import torch
import torch.nn as nn
from backbone.utils.neural_utils import to_activation
from backbone.utils.neural_utils import BiasLayer


class ProjectionHead(nn.Module):
    """
    PyTorch version of the BERT4Rec output projection head.
    Projects encoded sequences to logits over all items (including [MASK]).
    """

    def __init__(self, emb_dim, item_embedder: nn.Embedding, activation: str = "relu"):
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim)
        self.activation = to_activation(activation)
        self.item_embedder = item_embedder
        # self.bias = BiasLayer(item_embedder.num_embeddings)  # Include all items
        self.bias = BiasLayer(item_embedder.num_embeddings - 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_masked, emb_dim)

        Returns:
            Tensor of shape (num_masked, num_items)
        """
        x = self.dense(x)
        x = self.activation(x)

        # Use full item embedding (including mask token)
        # item_matrix = self.item_embedder.weight  # shape: (num_items, emb_dim)
        item_matrix = self.item_embedder.weight[:-1]  # exclude [MASK]
        logits = torch.matmul(x, item_matrix.T)  # shape: (num_masked, num_items)

        logits = self.bias(logits)
        return logits