import torch
from torch import nn
from torch.nn import Module, ModuleList

# main class

class Feedforwards(Module):

    def __init__(
        self,
        dim,
        depth,
        *,
        activation = nn.GELU(),
        bias = True,
        expansion_factor = 4.,
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

    def forward(
        self,
        x
    ):

        for layer in self.layers:
            x = layer(x) + x

        return x
