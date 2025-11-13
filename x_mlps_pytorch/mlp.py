import torch
from torch import nn, cat
from torch.nn import Module, ModuleList

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
        activate_last = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'

        layers = []

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = nn.Linear(dim_in, dim_out, bias = bias)

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

        return x

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
