import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

# weight decay friendly norms
# Ohad Rubin - https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, x.shape[-1:], self.gamma + 1, eps = self.eps)

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma + 1, eps = self.eps)
