import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack

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

# modules

class GroupedRMSNorm(Module):
    def __init__(
        self,
        dim,
        groups = 1
    ):
        super().__init__()
        self.groups = groups
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # grouped l2norm

        x = rearrange(x, '... (g d) n -> ... g d n ', g = self.groups)
        x = F.normalize(x, dim = -2, p = 2)
        x = rearrange(x, '... g d n -> ... (g d) n')

        gamma = rearrange(self.gamma, 'd -> d 1') # channel first

        return gamma * x * self.scale

# main class

class GroupedFeedforwards(Module):

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
        final_norm = False,
        groups = 1,
        squeeze_if_one_group = False
    ):
        super().__init__()

        layers = []

        # take care of groups

        self.groups = groups
        self.squeeze_if_one_group = squeeze_if_one_group

        dim = dim * groups
        first_dim = dim * groups

        if exists(dim_in):
            dim_in *= groups
            first_dim = dim_in

        if exists(dim_out):
            dim_out *= groups

        dim_hidden = int(dim * expansion_factor)

        self.first_dim = first_dim

        # layers

        for _ in range(depth):

            layer = nn.Sequential(
                GroupedRMSNorm(dim, groups = groups),
                nn.Conv1d(dim, dim_hidden, 1, bias = bias, groups = groups),
                activation,
                nn.Conv1d(dim_hidden, dim, 1, bias = bias, groups = groups)
            )

            layers.append(layer)

        self.layers = ModuleList(layers)

        # maybe final norm

        self.norm = GroupedRMSNorm(dim, groups = groups) if final_norm else nn.Identity()

        # proj in and out

        self.proj_in = nn.Conv1d(dim_in, dim, 1, groups = groups) if exists(dim_in) else nn.Identity()
        self.proj_out = nn.Conv1d(dim, dim_out, 1, groups = groups) if exists(dim_out) else nn.Identity()

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

        # project in

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        x = self.norm(x)

        x = self.proj_out(x)

        # get back the spatial dimensions

        x = inv_pack(x)
        x = rearrange(x, 'b d ... -> b ... d')

        x = rearrange(x, 'b ... (g d) -> b ... g d', g = self.groups)

        if self.squeeze_if_one_group and self.groups == 1:
            x = rearrange(x, 'b ... 1 d -> b ... d')

        return x
