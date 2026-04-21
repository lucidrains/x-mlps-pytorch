from __future__ import annotations
from functools import partial
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, cat
from torch.nn import Module, ModuleList, Identity

from x_mlps_pytorch.norms import RMSNorm, LayerNorm
from x_mlps_pytorch.normed_mlp import create_mlp as create_normed_mlp

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# modules

class ResidualUpdate(Module):
    def forward(self, x, residual):
        return x + residual

class OrthogonalResidualUpdate(Module):
    def __init__(
        self,
        double_precision = True
    ):
        super().__init__()
        self.double_precision = double_precision

    def forward(self, x, residual):
        use_double, dtype = self.double_precision, residual.dtype

        orig_residual = residual
        if use_double:
            residual, x = residual.double(), x.double()

        unit = F.normalize(residual, dim = -1)
        parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
        orthogonal = x - parallel

        if use_double:
            orthogonal = orthogonal.to(dtype)

        return orig_residual + orthogonal

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
        skip_to_output = False,     # auto-compression network
        use_orthogonal_residual = False,
        orthogonal_residual_double_precision = True,
        keel_post_ln = False,       # use the keel post-norm architecture proposed by Chen et al. from Bytedance - https://arxiv.org/abs/2601.19895
        keel_residual_scale = None  # defaults to number of blocks, for scaling the residual besides the first
    ):
        super().__init__()
        assert not (keel_post_ln and skip_to_output)

        assert divisible_by(depth, residual_every), '`depth` must be divisible by `residual_every`'

        # proj in and out

        self.proj_in = nn.Linear(dim_in, dim) if exists(dim_in) else Identity()
        self.proj_out = nn.Linear(dim, dim_out) if exists(dim_out) else Identity()

        # layers

        layers = []

        # norm type

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

        # layers

        num_blocks = depth // residual_every

        for _ in range(num_blocks):

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

        self.post_norms = ModuleList([norm_fn(dim) for _ in range(num_blocks - 1)]) if keel_post_ln else None

        self.keel_residual_scale = default(keel_residual_scale, num_blocks)

        self.skip_to_output = skip_to_output

        if use_orthogonal_residual:
            self.residual_update = OrthogonalResidualUpdate(double_precision = orthogonal_residual_double_precision)
        else:
            self.residual_update = ResidualUpdate()

        # final norm if not using keel post norm

        self.final_norm = norm_fn(dim) if final_norm and not keel_post_ln else Identity()

    def forward(
        self,
        x
    ):
        skip_to_output = self.skip_to_output

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)

        layer_outs = []

        for layer_index, layer in enumerate(self.layers):
            first_layer = layer_index == 0

            residual = x

            out = layer(x)

            # skip to output - auto compression paper

            if skip_to_output:
                x = out
                layer_outs.append(out)
                continue

            # handle residual with maybe post norm with keel stabilisation techniques

            has_post_norms = exists(self.post_norms)

            if first_layer or not has_post_norms:
                x = self.residual_update(out, residual)
                continue

            # handle post norm

            post_norm = self.post_norms[layer_index - 1]

            residual = residual * self.keel_residual_scale

            x = self.residual_update(out, residual)

            x = post_norm(x)

        # Dorovatas et al. https://openreview.net/forum?id=eIDa6pd9iQ

        if skip_to_output:
            x = sum(layer_outs)

        x = self.final_norm(x)

        return self.proj_out(x)
