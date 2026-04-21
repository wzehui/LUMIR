from .gelu import CustomGELU
import torch.nn as nn

__STR_TO_ACTIVATION = {
    "gelu": CustomGELU,
    "relu": nn.ReLU,
}

def to_activation(act) -> nn.Module:
    if isinstance(act, nn.Module):
        return act
    elif isinstance(act, str):
        if act in __STR_TO_ACTIVATION:
            return __STR_TO_ACTIVATION[act]()
        else:
            raise ValueError(f"Unsupported activation function string: {act}")
    else:
        raise ValueError(f"Unsupported activation type: {type(act)}")