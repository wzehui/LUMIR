import torch
from backbone.transformer.custom_layers.transformer_encoder_layer import TransformerEncoderLayer
from backbone.transformer.transformer_model import TransformerModel
from backbone.utils.neural_utils.custom_preprocessors.tensor_factory import TensorFactory


class BERTModel(TransformerModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BERT4Rec model.

        Args:
            inputs (torch.Tensor): Shape (batch_size, sequence_length),
                containing masked session sequences.

        Returns:
            torch.Tensor: Predictions for masked positions.
        """
        # Determine which positions are masked (== mask_target_used)
        relevant = (inputs == self.mask_target_used)

        # Replace PADDING_TARGET positions with mask_target_used token
        inputs = torch.where(
            inputs == TensorFactory.PADDING_TARGET,
            torch.full_like(inputs, self.mask_target_used),
            inputs,
        )

        # Embedding → Transformer → extract masked positions → projection
        embeddings = self.embedding_layer(inputs)                     # (B, L, D)
        transformations = self.transformer(embeddings)                # (B, L, D)
        relevant_transformations = transformations[relevant]         # (num_masked, D)
        predictions = self.projection_head(relevant_transformations) # (num_masked, V)

        return predictions

    def get_transformer_layer(self):
        return TransformerEncoderLayer(
            embed_dim=self.emb_dim,
            intm_dim=self.emb_dim * self.trans_dim_scale,
            num_heads=self.h,
            dropout=self.drop_rate,
            use_causal_mask=False,  # important for BERT4Rec
            activation=self.activation,
            **self.transformer_layer_kwargs,
        )