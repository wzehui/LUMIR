import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    """
    PyTorch version matching TensorFlow's embedding layer initialization (uniform).
    Combines item and positional embeddings via element-wise sum.
    """

    def __init__(self, N: int, num_items: int, emb_dim: int):
        super().__init__()
        self.N = N
        self.item_emb = nn.Embedding(num_items + 1, emb_dim)  # +1 for [MASK]
        self.pos_emb = nn.Embedding(N, emb_dim)

        # ✅ Match Keras default: uniform(-limit, limit), limit = sqrt(1 / input_dim)
        item_limit = math.sqrt(1.0 / (num_items + 1))
        pos_limit = math.sqrt(1.0 / N)

        nn.init.uniform_(self.item_emb.weight, -item_limit, item_limit)
        nn.init.uniform_(self.pos_emb.weight, -pos_limit, pos_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()

        position_ids = torch.arange(self.N, dtype=torch.long, device=x.device)
        pos_embs = self.pos_emb(position_ids).unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, D)

        item_embs = self.item_emb(x)  # (B, L, D)
        if seq_len < self.N:
            pos_embs = pos_embs[:, :seq_len, :]

        return item_embs + pos_embs