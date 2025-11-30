from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList, Identity

from x_mlps_pytorch.norms import RMSNorm, LayerNorm
from x_mlps_pytorch.normed_mlp import create_mlp as create_normed_mlp

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# main class

class ResidualNormedMLP(Module):
    # proposed by Wang et al. https://arxiv.org/abs/2503.14858 for scaling goal conditioned CRL

    def __init__(
        self,
        dim,
        depth = 32,
        residual_every = 4, # they used every 4
        dim_in = None,
        dim_out = None,
        activation = nn.SiLU(), # apparently silu was very important - nice ablation in paper
        bias = True,
        norm_fn: Module | None = None,
        use_rmsnorm = False,
        final_norm = True,
        skip_to_output = False # auto-compression network
    ):
        super().__init__()
        assert divisible_by(depth, residual_every), '`depth` must be divisible by `residual_every`'

        # proj in and out

        self.proj_in = nn.Linear(dim_in, dim) if exists(dim_in) else Identity()
        self.proj_out = nn.Linear(dim, dim_out) if exists(dim_out) else Identity()

        # layers

        layers = []

        # norm type

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

        self.final_norm = norm_fn(dim) if final_norm else Identity()

        # layers

        for _ in range(depth // residual_every):

            block = create_normed_mlp(
                dim = dim,
                depth = residual_every,
                norm_fn = norm_fn,
                activation = activation,
                bias = bias,
                activate_last = True
            )

            layers.append(block)

        self.layers = ModuleList(layers)

        self.skip_to_output = skip_to_output

    def forward(
        self,
        x
    ):
        skip_to_output = self.skip_to_output

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)

        layer_outs = []

        for layer in self.layers:
            out = layer(x)

            # traditional residual

            if not skip_to_output:
                x = x + out
                continue

            x = out
            layer_outs.append(out)

        # Dorovatas et al. https://openreview.net/forum?id=eIDa6pd9iQ

        if skip_to_output:
            x = sum(layer_outs)

        x = self.final_norm(x)

        return self.proj_out(x)
