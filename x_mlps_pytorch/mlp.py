import torch
from torch import nn, cat
from torch.nn import Linear, Module, ModuleList
from einops.layers.torch import Rearrange
from einops import rearrange
# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class MLP(Module):
    def __init__(
        self,
        *dims,
        activation = nn.ReLU(),
        bias = True,
        activate_last = False,
        squeeze_out = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'
        self.squeeze_out = squeeze_out

        if squeeze_out:
            assert dims[-1] == 1, 'last dimension must be 1 to squeeze out'

        layers = []

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = Linear(dim_in, dim_out, bias = bias)

            # if not last, add an activation after each linear layer

            if not is_last or activate_last:
                layer = nn.Sequential(layer, activation)

            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(
        self,
        x
    ):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        for layer in self.layers:
            x = layer(x)

        if self.squeeze_out:
            x = rearrange(x, '... 1 -> ...')

        return x

# factory function

def create_mlp(
    dim,
    depth,
    *,
    dim_in = None,
    dim_out = None,
    bias = True,
    squeeze_out = False,
    **mlp_kwargs
):
    no_depth = depth == 0
    requires_proj_in_out = exists(dim_in) or exists(dim_out)

    if no_depth and not requires_proj_in_out:
        return nn.Identity()
    elif no_depth and not squeeze_out:
        return Linear(default(dim_in, dim), default(dim_out, dim), bias = bias)

    dims = (dim,) * (depth + 1)

    if exists(dim_in):
        dims = (dim_in, *dims)

    if exists(dim_out):
        dims = (*dims, dim_out)

    return MLP(
        *dims,
        bias = bias,
        squeeze_out = squeeze_out,
        **mlp_kwargs
    )
