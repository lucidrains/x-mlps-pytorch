from __future__ import annotations

from typing import Callable

from torch import nn, cat
from torch.nn import Module, ModuleList

from x_mlps_pytorch.weight_only_mlp import (
    PreRMSNorm,
    WeightOnlyLinear
)

# functions

def exists(v):
    return v is not None

# modules

class WeightOnlyFeedforward(Module):
    # the transformer style sandwich - optional pre-norm, up projection, activation, optional norm, down projection
    # every bias is a column of its weight, the input being padded with a constant 1
    # if use_bias is False, the projections are just regular bias free linears

    def __init__(
        self,
        dim,
        dim_hidden,
        activation: Module | Callable = nn.GELU(),
        use_rmsnorm = False,
        norm_after_activation = False,
        bias_left = False,
        use_bias = True,
        norm_eps = 1e-5
    ):
        super().__init__()
        self.to_hidden = WeightOnlyLinear(dim, dim_hidden, activation = activation, use_rmsnorm = use_rmsnorm, bias_left = bias_left, use_bias = use_bias, norm_eps = norm_eps)
        # the optional norm after the activation is simply the pre-norm of the down projection
        self.to_out = WeightOnlyLinear(dim_hidden, dim, use_rmsnorm = use_rmsnorm and norm_after_activation, bias_left = bias_left, use_bias = use_bias, norm_eps = norm_eps)

    def forward(self, x):
        return self.to_out(self.to_hidden(x))

class WeightOnlyFeedforwards(Module):
    """
    a stack of transformer style feedforwards, composed only of weight matrices
    the bias of every layer is a column of its weight, the input being padded with a constant 1
    the optional pre-rmsnorms are parameter free - any gamma belongs to the weight columns below them
    every parameter is thus a rank-2 matrix, directly adaptable by lora in evolutionary frameworks
    """

    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        activation: Module | Callable = nn.GELU(),
        expansion_factor = 4.,
        use_rmsnorm = False,
        norm_after_activation = False,
        final_norm = False,
        bias_left = False,
        use_bias = True,
        norm_eps = 1e-5
    ):
        super().__init__()

        dim_hidden = int(dim * expansion_factor)

        # layers

        self.layers = ModuleList([
            WeightOnlyFeedforward(
                dim,
                dim_hidden,
                activation = activation,
                use_rmsnorm = use_rmsnorm,
                norm_after_activation = norm_after_activation,
                bias_left = bias_left,
                use_bias = use_bias,
                norm_eps = norm_eps
            )
            for _ in range(depth)
        ])

        # maybe final norm

        self.norm = PreRMSNorm(dim, eps = norm_eps) if final_norm else nn.Identity()

        # proj in and out

        self.proj_in = WeightOnlyLinear(dim_in, dim, bias_left = bias_left, use_bias = use_bias, norm_eps = norm_eps) if exists(dim_in) else None
        self.proj_out = WeightOnlyLinear(dim, dim_out, bias_left = bias_left, use_bias = use_bias, norm_eps = norm_eps) if exists(dim_out) else None

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        if exists(self.proj_in):
            x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        x = self.norm(x)

        if exists(self.proj_out):
            x = self.proj_out(x)

        return x
