from __future__ import annotations

from functools import partial

from torch import nn
from torch.nn import Module

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

LinearNoBias = partial(nn.Linear, bias = False)

# LoRA - https://arxiv.org/abs/2106.09685

class LoRA(Module):
    """
    low-rank projection, defaulting to identity at init (up projection zeroed)
    falls back to a full linear if rank is not less than min(dim, dim_out)
    """

    def __init__(
        self,
        dim,
        rank = 16,
        dim_out = None,
        activation: Module | None = None,
        zero_init_up = True,
        assert_rank_less_than_dim = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        min_dim = min(dim, dim_out)
        self.fallback = rank >= min_dim

        if self.fallback:
            if assert_rank_less_than_dim:
                raise AssertionError(
                    f'rank ({rank}) must be less than min(dim, dim_out) ({min_dim})'
                )

            self.linear = nn.Linear(dim, dim_out)

            if zero_init_up:
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)

            return

        self.down = LinearNoBias(dim, rank)
        self.up = LinearNoBias(rank, dim_out)

        if zero_init_up:
            nn.init.zeros_(self.up.weight)

        self.activation = default(activation, nn.Identity())

    def forward(self, x):
        if self.fallback:
            return self.linear(x)

        return self.up(self.activation(self.down(x)))
