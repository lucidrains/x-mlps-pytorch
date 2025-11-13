import torch
from torch import nn, cat
from torch.nn import Module, ModuleList

# functions

def exists(v):
    return v is not None

# main class

class Feedforwards(Module):

    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        activation = nn.GELU(),
        bias = True,
        expansion_factor = 4.,
        final_norm = False
    ):
        super().__init__()

        layers = []

        dim_hidden = int(dim * expansion_factor)

        # layers

        for _ in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim_hidden, bias = bias),
                activation,
                nn.Linear(dim_hidden, dim, bias = bias)
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
        x
    ):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        x = self.norm(x)

        return self.proj_out(x)
