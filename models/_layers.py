import torch
import torch.nn as nn


class ScaledLeakyRelu6(nn.Module):
    def __init__(self, neg_k=0.01):
        super().__init__()
        self.neg_k = neg_k

    def _forward(self, x):
        return torch.clamp(
            torch.where(x >= 0, x, x * self.neg_k),
            min=0.0,
            max=6.0,
        )

    def forward(self, x):
        return self._forward(x) / 6.0  # bound to [0, 1]

    def extra_repr(self):
        return f"neg_k={self.neg_k}"
