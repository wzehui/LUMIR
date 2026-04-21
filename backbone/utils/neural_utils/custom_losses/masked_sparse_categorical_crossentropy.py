import torch
import torch.nn.functional as F


# def masked_sparse_categorical_crossentropy(y_true: torch.Tensor,
#                                            y_pred: torch.Tensor,
#                                            padding_target: int) -> torch.Tensor:
#     """
#     PyTorch version of masked sparse categorical crossentropy.
#
#     Args:
#         y_true (torch.Tensor): Tensor of shape (B, T) with true item indices, where
#             padding positions are marked with `padding_target`.
#         y_pred (torch.Tensor): Tensor of shape (N, num_items), containing the
#             logits/predicted probability distributions for masked positions.
#         padding_target (int): The special ID used to indicate padding in y_true.
#
#     Returns:
#         torch.Tensor: Scalar loss value (mean cross-entropy over non-padding positions).
#     """
#     # Flatten and mask y_true to get valid labels
#     y_true_masked = y_true[y_true != padding_target]
#
#     if y_true_masked.numel() == 0:
#         # No valid target to compute loss
#         return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)
#
#     # Assume y_pred is already masked (only predictions for masked positions)
#     # Its shape should be (num_masked_positions, num_items)
#     assert y_pred.size(0) == y_true_masked.size(0), \
#         f"y_pred has shape {y_pred.shape}, y_true_masked has shape {y_true_masked.shape}"
#
#     # Compute standard sparse categorical cross-entropy
#     return F.cross_entropy(y_pred, y_true_masked, reduction='mean')

def masked_sparse_categorical_crossentropy(y_pred, y_true, **kwargs):
    """
    Version that assumes y_true has already been filtered to masked positions.
    """
    return torch.nn.functional.cross_entropy(y_pred, y_true, reduction="mean")