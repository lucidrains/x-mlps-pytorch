# catering to latent gene conditioned actor/critic from EPO
# https://web3.arxiv.org/abs/2503.19037

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList

from einops import rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# uses adaptive rmsnorm + ada-ln-zero from Peebles et al. of DiT

class AdaRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_cond = default(dim_cond, dim)

        self.to_gamma = Linear(dim_cond, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(
        self,
        x,
        *,
        cond
    ):
        normed = l2norm(x) * self.scale
        gamma = self.to_gamma(cond)
        return normed * (gamma + 1.)

class AdaLNZero(Module):
    def __init__(
        self,
        dim,
        dim_cond = None,
        init_bias_value = -2.
    ):
        super().__init__()
        dim_cond = default(dim_cond, dim)
        self.to_gamma = Linear(dim_cond, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(
        self,
        x,
        *,
        cond
    ):
        gamma = self.to_gamma(cond).sigmoid()
        return x * gamma

class AdaNormConfig(Module):
    def __init__(
        self,
        fn: Module,
        *,
        dim,
        dim_cond = None
    ):
        super().__init__()
        dim_cond = default(dim_cond, dim)

        self.ada_norm = AdaRMSNorm(dim = dim, dim_cond = dim_cond)

        self.fn = fn

        self.ada_ln_zero = AdaLNZero(dim = dim, dim_cond = dim_cond)

    def forward(
        self,
        t,
        *args,
        cond,
        **kwargs
    ):
        if t.ndim == 3:
            cond = rearrange(cond, 'b d -> b 1 d')

        t = self.ada_norm(t, cond = cond)

        out = self.fn(t, *args, **kwargs)

        return self.ada_ln_zero(out, cond = cond)

# main class

class LatentConditionedFeedforwards(Module):

    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_latent = None,
        latent_mlp = False,
        activation = nn.GELU(),
        bias = True,
        expansion_factor = 4.,
        final_norm = False
    ):
        super().__init__()

        layers = []

        dim_hidden = int(dim * expansion_factor)

        dim_cond = default(dim_latent, dim)

        # whether to take care of a latent mlp

        self.to_latent_cond = nn.Identity()

        if latent_mlp:
            self.to_latent_cond = nn.Sequential(
                Linear(dim_latent, dim * 2),
                nn.SiLU(),
            )

            dim_cond = dim * 2

        # layers

        for _ in range(depth):

            layer = nn.Sequential(
                nn.Linear(dim, dim_hidden, bias = bias),
                activation,
                nn.Linear(dim_hidden, dim, bias = bias)
            )

            layer = AdaNormConfig(
                layer,
                dim = dim,
                dim_cond = dim_cond
            )

            layers.append(layer)

        self.layers = ModuleList(layers)

        # maybe final norm

        self.norm = nn.RMSNorm(dim) if final_norm else nn.Identity()

        # proj in and out

        self.proj_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()
        self.proj_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        latent
    ):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)

        cond = self.to_latent_cond(latent)

        for layer in self.layers:
            x = layer(x, cond = cond) + x

        x = self.norm(x)

        return self.proj_out(x)
