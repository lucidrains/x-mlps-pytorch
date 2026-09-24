from __future__ import annotations

from typing import Callable

from torch import nn, cat
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import pad_left_at_dim, pad_right_at_dim

# functions

def exists(v):
    return v is not None

# modules

class PreRMSNorm(Module):
    # parameter free pre-rmsnorm - there is no gamma
    # any per channel scale is simply part of the weight columns that read this output

    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), eps = self.eps)

class WeightOnlyLinear(Module):
    # a linear layer whose bias is a column of its weight, the input being padded with a constant 1
    # if use_bias is False, it is just a regular bias free linear
    # an optional parameter free pre-rmsnorm precedes it, an optional activation follows

    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Module | Callable | None = None,
        use_rmsnorm = False,
        bias_left = False,
        use_bias = True,
        norm_eps = 1e-5
    ):
        super().__init__()
        self.norm = PreRMSNorm(dim_in, eps = norm_eps) if use_rmsnorm else None
        self.linear = nn.Linear(dim_in + int(use_bias), dim_out, bias = False)
        self.activation = activation
        self.use_bias = use_bias
        self.pad_ones = pad_left_at_dim if bias_left else pad_right_at_dim

    def forward(self, x):
        if exists(self.norm):
            x = self.norm(x)

        # pad a constant 1, turning the bias into a weight column

        if self.use_bias:
            x = self.pad_ones(x, 1, value = 1.)

        x = self.linear(x)

        if exists(self.activation):
            x = self.activation(x)

        return x

class WeightOnlyMLP(Module):
    """
    an mlp composed only of weight matrices
    the bias of every layer is a column of its weight, the input being padded with a constant 1
    the optional pre-rmsnorm is parameter free - any gamma belongs to the weight columns below it
    every parameter is thus a rank-2 matrix, directly adaptable by lora in evolutionary frameworks
    """

    def __init__(
        self,
        *dims,
        activation: Module | Callable = nn.ReLU(),
        use_rmsnorm = False,
        bias_left = False,
        use_bias = True,
        norm_eps = 1e-5,
        activate_last = False,
        squeeze_out = False
    ):
        super().__init__()
        assert len(dims) > 1, 'must have more than 1 layer'
        self.squeeze_out = squeeze_out

        if squeeze_out:
            assert dims[-1] == 1, 'last dimension must be 1 to squeeze out'

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        layers = []

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)
            has_activation = not is_last or activate_last

            layer = WeightOnlyLinear(
                dim_in,
                dim_out,
                activation = activation if has_activation else None,
                use_rmsnorm = use_rmsnorm,
                bias_left = bias_left,
                use_bias = use_bias,
                norm_eps = norm_eps
            )
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        for layer in self.layers:
            x = layer(x)

        if self.squeeze_out:
            x = rearrange(x, '... 1 -> ...')

        return x
