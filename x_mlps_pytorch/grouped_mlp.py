import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import pack, unpack, rearrange, repeat

# functions

def exists(v):
    return v is not None

def first(arr):
    return arr[0]

def pack_with_inverse(t, pattern):
    packed, shape = pack([t], pattern)

    def inverse(out):
        return first(unpack(out, shape, pattern))

    return packed, inverse

# main class

class GroupedMLP(Module):
    def __init__(
        self,
        *dims,
        activation = nn.ReLU(),
        bias = True,
        activate_last = False,
        groups = 1,
        squeeze_if_one_group = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'

        layers = []

        # input output dimension pairs

        dims = tuple(dim * groups for dim in dims)
        first_dim = first(dims)
        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = nn.Conv1d(dim_in, dim_out, 1, groups = groups, bias = bias)

            # if not last, add an activation after each linear layer

            if not is_last or activate_last:
                layer = nn.Sequential(layer, activation)

            layers.append(layer)

        self.layers = ModuleList(layers)

        # groups

        self.groups = groups
        self.first_dim = first_dim
        self.squeeze_if_one_group = squeeze_if_one_group and groups == 1

    def forward(
        self,
        x
    ):
        dim = x.shape[-1]

        # channel first

        x = rearrange(x, 'b ... d -> b d ...')

        # repeat for groups if needed

        if dim != self.first_dim:
            x = repeat(x, 'b d ... -> b (g d) ...', g = self.groups)

        # pack

        x, inv_pack = pack_with_inverse(x, 'b d *')

        # layers

        for layer in self.layers:
            x = layer(x)

        # get back the spatial dimensions

        x = inv_pack(x)
        x = rearrange(x, 'b d ... -> b ... d')

        x = rearrange(x, 'b ... (g d) -> b ... g d', g = self.groups)

        if self.squeeze_if_one_group and self.groups == 1:
            x = rearrange(x, 'b ... 1 d -> b ... d')

        return x

# factory function

def create_grouped_mlp(
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

    return GroupedMLP(*dims, **mlp_kwargs)
