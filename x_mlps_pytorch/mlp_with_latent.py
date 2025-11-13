import torch
from torch import nn, cat
from torch.nn import Linear, Module, ModuleList

from einops import repeat

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
        dim_latent,
        latent_mlp = False,
        latent_mlp_dim_hidden = None,
        condition_hiddens = False,
        activation = nn.ReLU(),
        bias = True,
        activate_last = False,
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'

        layers = []

        # concatenative conditioning

        first_dim, *rest_dims = dims

        latent_mlp_dim_hidden = default(latent_mlp_dim_hidden, first_dim * 2)

        # maybe latent mlp

        dim_cond = dim_latent

        self.to_latent_cond = nn.Identity()

        if latent_mlp:
            self.to_latent_cond = nn.Sequential(
                Linear(dim_latent, latent_mlp_dim_hidden),
                nn.SiLU(),
            )

            dim_cond = latent_mlp_dim_hidden

        # dimensions again

        first_dim_with_latent_cond = first_dim + dim_cond

        dims = (first_dim_with_latent_cond, *rest_dims)

        # input output dimension pairs

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_first = i == 0
            is_last = i == len(dim_in_out)

            layer = nn.Linear(dim_in, dim_out, bias = bias)

            # if not last, add an activation after each linear layer

            if not is_last or activate_last:
                layer = nn.Sequential(layer, activation)

            # maybe additional layer cond

            latent_to_layer_cond = None

            if latent_to_layer_cond and not is_first:
                latent_to_layer_cond = Linear(dim_cond, dim_in)

            layers.append(ModuleList([
                layer,
                latent_to_layer_cond
            ]))

        self.layers = ModuleList(layers)

    def forward(
        self,
        x,
        latent
    ):
        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        cond = self.to_latent_cond(latent)

        if cond.ndim == 2 and x.ndim == 3:
            cond = repeat(cond, 'b d -> b n d', n = x.shape[1])
            x = cat((cond, x), dim = -1)

        for layer, maybe_cond in self.layers:

            # maybe hadamard conditioning with latent

            if exists(maybe_cond):
                x = x * maybe_cond(cond)

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
