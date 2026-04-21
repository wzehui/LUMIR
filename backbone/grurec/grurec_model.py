import torch
import torch.nn as nn
from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import TensorFactory
from backbone.utils.neural_utils.custom_layers.projection_head import ProjectionHead
from backbone.utils.neural_utils.custom_losses.masked_sparse_categorical_crossentropy import (
    masked_sparse_categorical_crossentropy,
)


class GRURecModel(nn.Module):
    def __init__(self, num_items, emb_dim, hidden_dim, drop_rate, optimizer_kwargs, activation):
        super().__init__()
        self.num_items = num_items
        self.mask_target_used = num_items  # identical to TF: the (V+1)-th id is the mask token
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.optimizer_kwargs = optimizer_kwargs

        # === Embedding (V+1, E) ===
        self.embedding_layer = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=emb_dim)
        self.embedding_dropout = nn.Dropout(p=drop_rate)

        # === GRU: [B, T, E] -> [B, T, H] ===
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gru_dropout = nn.Dropout(p=drop_rate)

        # === Optional adapter: H -> E (ProjectionHead expects E-dim features) ===
        self.adapt_to_emb = nn.Identity() if hidden_dim == emb_dim else nn.Linear(hidden_dim, emb_dim, bias=False)

        # === Projection head: maps [..., E] -> [..., V] using tied weights ===
        self.proj_head = ProjectionHead(emb_dim, self.embedding_layer, activation)

        # TF-compat "compile-like" members (not required by your loop but kept for parity)
        self.optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        self.criterion = masked_sparse_categorical_crossentropy

    def forward(self, inputs: torch.Tensor, training: bool = None):
        """
        TF-parity forward:
        - TRAIN: return logits for *all* non-padding positions -> [N_true, V]
                 (use boolean mask from ORIGINAL inputs and flatten inside the model)
        - EVAL : return logits for last timestep only         -> [B, V]
        """
        # Ensure long dtype
        inputs = inputs.long()

        # Follow module mode if 'training' is not explicitly passed
        if training is None:
            training = self.training

        # --- Device safety: move inputs to the same device as the embedding ---
        dev = self.embedding_layer.weight.device
        if inputs.device != dev:
            inputs = inputs.to(dev)

        # === Build padding mask from ORIGINAL inputs (before replacement), same as TF ===
        # padding == True where inputs are TensorFactory.PADDING_TARGET
        padding = inputs.eq(TensorFactory.PADDING_TARGET)  # [B, T] bool

        # === Replace padding ids with 'mask_target_used' for embedding lookup ===
        # (This mirrors TF's: tf.where(padding, mask_target_used, inputs))
        inputs = torch.where(
            padding,
            torch.as_tensor(self.mask_target_used, device=inputs.device, dtype=inputs.dtype),
            inputs,
        )

        # === Embedding + dropout ===
        x = self.embedding_layer(inputs)  # [B, T, E]
        x = self.embedding_dropout(x)

        # === GRU (+ dropout) ===
        x, _ = self.gru(x)                # [B, T, H]
        x = self.gru_dropout(x)

        # === H -> E (if needed) to feed ProjectionHead ===
        x = self.adapt_to_emb(x)          # [B, T, E]

        if training:
            # === TRAIN: boolean-mask like TF; flatten all valid (non-padding) positions ===
            valid = ~padding              # [B, T] bool
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                # Edge case: batch consists entirely of paddings; return empty logits
                # (your fit loop will skip this batch when labels are empty)
                return x.new_zeros((0, self.num_items))

            # Collapse B,T using boolean mask -> [N_true, E]
            x = x[valid]

            # Project to vocabulary logits -> [N_true, V]
            out = self.proj_head(x)
            return out
        else:
            # === EVAL/PRED: only the last timestep is used ===
            x_last = x[:, -1, :]          # [B, E]
            out = self.proj_head(x_last)  # [B, V]
            return out

    # def call(self, inputs: torch.Tensor, training: bool):
    #     """
    #     Keras-style alias for forward(), allowing external frameworks to call .call().
    #     """
    #     return self.forward(inputs, training)
    #
    # def compile_model(self):
    #     """
    #     Keras-style no-op method for API compatibility.
    #     The optimizer and loss have already been initialized in __init__.
    #     """
    #     return