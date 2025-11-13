from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList, LayerNorm, RMSNorm

# functions

def exists(v):
    return v is not None

# main class

class MLP(Module):
    def __init__(
        self,
        *dims,
        activation = nn.ReLU(),
        bias = True,
        norm_fn: Module | None = None,
        use_rmsnorm = False,
        final_norm = False,
        activate_last = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'

        # layers

        layers = []

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # norm type

        if not exists(norm_fn):
            norm_fn = nn.RMSNorm if use_rmsnorm else partial(nn.LayerNorm, bias = False)

        *_, last_dim = dims

        self.final_norm = norm_fn(last_dim) if final_norm else nn.Identity()

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = nn.Linear(dim_in, dim_out, bias = bias)

            norm = norm_fn(dim_out)

            # if not last, add an activation after each linear layer

            if not is_last or activate_last:
                layer = nn.Sequential(layer, norm, activation)

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

        return self.final_norm(x)

# factory function

def create_mlp(
    dim,
    depth,
    *,
    dim_in = None,
    dim_out = None,
    **mlp_kwargs
):
    dims = (dim,) * depth

    if exists(dim_in):
        dims = (dim_in, *dims)

    if exists(dim_out):
        dims = (*dims, dim_out)

    return MLP(*dims, **mlp_kwargs)
