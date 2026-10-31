import math

import torch
from torch import nn, einsum
from torch.nn import Module

from einops import rearrange
from torch_einops_utils.shape import shape

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main class

class GroupedLinear(Module):
    """
    x: (..., groups * dim_in) -> (..., groups * dim_out)

    each contiguous chunk of the input is projected by its own weight.
    """

    def __init__(
        self,
        dim,
        dim_out = None,
        groups = 1,
        bias = True,
        device = None,
        dtype = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        assert groups >= 1, 'groups must be at least 1'
        assert divisible_by(dim, groups), f'dim ({dim}) must be divisible by groups ({groups})'
        assert divisible_by(dim_out, groups), f'dim_out ({dim_out}) must be divisible by groups ({groups})'

        self.in_features = dim
        self.out_features = dim_out
        self.groups = groups
        self.dim_in_per_group = dim // groups
        self.dim_out_per_group = dim_out // groups

        # one weight matrix per group

        self.weight = nn.Parameter(torch.empty(groups, self.dim_out_per_group, self.dim_in_per_group, device = device, dtype = dtype))
        self.bias = nn.Parameter(torch.empty(dim_out, device = device, dtype = dtype)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.dim_in_per_group)
        nn.init.uniform_(self.weight, -bound, bound)

        if exists(self.bias):
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, groups={self.groups}, bias={exists(self.bias)}'

    def forward(self, x):
        # validate x is (..., groups * dim_in / groups)

        shape(x, '... (g i)', g = self.groups, i = self.dim_in_per_group)

        # x -> (..., groups, dim_in / groups)

        x = rearrange(x, '... (g i) -> ... g i', g = self.groups)

        # (..., g, i) @ (g, o, i) -> (..., g, o)

        out = einsum('... g i, g o i -> ... g o', x, self.weight)
        out = rearrange(out, '... g o -> ... (g o)')

        if exists(self.bias):
            out = out + self.bias

        return out
