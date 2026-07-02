from __future__ import annotations
from typing import Callable
from functools import partial

import torch
from torch import nn, cat
from torch.nn import Linear, Module, ModuleList

from x_mlps_pytorch.norms import LayerNorm, RMSNorm
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
        norm_fn: Callable[[int], Module] | None = None,
        use_rmsnorm = False,
        norm_elementwise_affine = None,
        final_norm = False,
        activate_last = False,
        squeeze_out = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'
        self.squeeze_out = squeeze_out

        if squeeze_out:
            assert dims[-1] == 1, 'last dimension must be 1 to squeeze out'

        # layers

        layers = []

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # norm type

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

            if exists(norm_elementwise_affine):
                norm_fn = partial(norm_fn, elementwise_affine = norm_elementwise_affine)

        *_, last_dim = dims

        self.final_norm = norm_fn(last_dim) if final_norm else nn.Identity()

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = Linear(dim_in, dim_out, bias = bias)

            layer_modules = (layer,)

            # if not last, add a norm and an activation after each linear layer

            if not is_last or activate_last:
                assert dim_out > 1, f'should not layernorm dimension of 1'

                norm = norm_fn(dim_out)

                layer_modules = (*layer_modules, norm, activation)

            layers.append(nn.Sequential(*layer_modules))

        self.layers = ModuleList(layers)

    def forward(
        self,
        x
    ):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

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
        assert not squeeze_out, 'squeeze_out requires dim_out = 1'
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
