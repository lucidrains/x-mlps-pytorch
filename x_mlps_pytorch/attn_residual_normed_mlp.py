from __future__ import annotations

import torch
from torch import nn, cat, stack, einsum, Tensor
from torch.nn import Module, ModuleList, Identity

from einops import repeat, pack, unpack

from x_mlps_pytorch.norms import RMSNorm, LayerNorm

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# lightweight attention for pooling

class AttentionPool(Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.query = nn.Parameter(torch.randn(dim) * 1e-2)

        self.to_q = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, context):
        batch = context.shape[0]

        q = repeat(self.query, 'd -> b d', b = batch)
        q = self.to_q(q)
        q = self.q_norm(q)

        k = self.k_norm(context)

        sim = einsum('b d, b j d -> b j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b j, b j d -> b d', attn, context)

        return out

# main class

class AttnResidualNormedMLP(Module):
    """
    ResidualNormedMLP variant where residual connections are replaced
    by attention-aggregated residuals over all layer hiddens.

    https://arxiv.org/abs/2601.21582
    https://arxiv.org/abs/2603.15031
    """

    def __init__(
        self,
        dim,
        depth = 32,
        dim_in = None,
        dim_out = None,
        activation = nn.SiLU(),
        bias = True,
        norm_fn: Module | None = None,
        use_rmsnorm = False,
        final_norm = True,
    ):
        super().__init__()

        self.proj_in = nn.Linear(dim_in, dim) if exists(dim_in) else Identity()
        self.proj_out = nn.Linear(dim, dim_out) if exists(dim_out) else Identity()

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

        self.layers = ModuleList([])

        for _ in range(depth):
            layer = nn.Sequential(
                nn.Linear(dim, dim, bias = bias),
                norm_fn(dim),
                activation,
            )

            attn_residual = AttentionPool(dim)

            self.layers.append(ModuleList([layer, attn_residual]))

        self.final_norm = norm_fn(dim) if final_norm else Identity()

    def forward(self, x):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)

        hiddens = [x]

        for layer, attn_residual in self.layers:
            out = layer(x)
            hiddens.append(out)

            stacked = stack(hiddens, dim = -2)
            stacked, ps = pack([stacked], '* l d')

            x = attn_residual(stacked)

            x, = unpack(x, ps, '* d')

        x = self.final_norm(x)

        return self.proj_out(x)

# quick test

if __name__ == '__main__':

    mlp = AttnResidualNormedMLP(
        dim = 256,
        depth = 64,
        dim_in = 77,
        dim_out = 64,
    )

    x = torch.randn(7, 3, 77)

    out = mlp(x)

    assert out.shape == (7, 3, 64)
