from __future__ import annotations

from torch import nn
from torch.nn import Linear, Module

from einops import rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# film

class FiLM(Module):
    def __init__(
        self,
        dim,
        cond_dim = None
    ):
        super().__init__()
        cond_dim = default(cond_dim, dim)

        self.to_gamma_beta = Linear(cond_dim, dim * 2)

        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        x,
        cond
    ):
        if x.ndim == 3 and cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)

        return x * (gamma + 1.) + beta
