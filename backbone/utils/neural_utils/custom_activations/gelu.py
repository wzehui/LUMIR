import torch
import torch.nn as nn

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output after applying GELU activation.
    """
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
    return x * cdf

class CustomGELU(nn.Module):
    def forward(self, x):
        return gelu(x)