import torch
from torch import nn

from backbone.transformer.custom_layers.transformer_encoder_layer import (
    TransformerEncoderLayer,
)
from backbone.transformer.transformer_model import (
    TransformerModel,
)
from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import (
    TensorFactory,
)


class SASRecModel(TransformerModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # SASRec additionally uses a dropout layer on the embeddings.
        self.embedding_dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines how the input tensor is propagated through the model
        to form the output tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length,)
                containing masked sessions.

        Returns:
            torch.Tensor: The model predictions.
        """
        # Determine whether we are in training or evaluation mode
        training = self.training

        # === 1. Identify padding positions ===
        padding = inputs.eq(TensorFactory.PADDING_TARGET)

        # === 2. Replace paddings with mask target token ===
        inputs = torch.where(
            padding,
            torch.as_tensor(
                self.mask_target_used, device=inputs.device, dtype=inputs.dtype
            ),
            inputs,
        )

        # === 3. Embedding + dropout ===
        embeddings = self.embedding_layer(inputs)
        embeddings = self.embedding_dropout(embeddings)

        # === 4. Transformer encoder ===
        transformations = self.transformer(embeddings)

        # === 5. Select relevant positions ===
        if training:
            # Use all non-padding positions for training
            relevant = ~padding  # boolean mask [B, T]
            relevant_transformations = transformations[relevant]  # [num_valid, D]
        else:
            # During prediction, only use the last timestep
            relevant_transformations = transformations[:, -1]

        # === 6. Projection head ===
        predictions = self.projection_head(relevant_transformations)
        return predictions

    def get_transformer_layer(self):
        """Build one SASRec-style TransformerEncoder layer."""
        return TransformerEncoderLayer(
            embed_dim=self.emb_dim,
            intm_dim=self.emb_dim * self.trans_dim_scale,
            num_heads=self.h,
            dropout=self.drop_rate,
            use_causal_mask=True,  # important for SASRec!
            activation=self.activation,
            **self.transformer_layer_kwargs,
        )