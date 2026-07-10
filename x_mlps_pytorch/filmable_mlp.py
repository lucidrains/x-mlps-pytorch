from torch import nn, cat
from torch.nn import Linear, Module, ModuleList

from einops import rearrange

# functions

def exists(v):
    return v is not None

# main class

class FiLMableMLP(Module):
    def __init__(
        self,
        *dims,
        cond_dim,
        cond_hidden_dims = None,
        cond_prepared = False,
        activation = nn.ReLU(),
        bias = True,
        activate_last = False,
        squeeze_out = False
    ):
        super().__init__()
        assert len(dims) > 1, f'must have more than 1 layer'
        self.squeeze_out = squeeze_out

        if squeeze_out:
            assert dims[-1] == 1, 'last dimension must be 1 to squeeze out'

        # conditioning

        self.cond_prepared = cond_prepared

        self.cond_mlp = None
        final_cond_dim = cond_dim

        if not cond_prepared:
            if not exists(cond_hidden_dims):
                cond_hidden_dims = (cond_dim * 2, cond_dim * 2)

            if isinstance(cond_hidden_dims, int):
                cond_hidden_dims = (cond_hidden_dims, cond_hidden_dims)

            cond_layers = []
            cond_dim_in = cond_dim

            for cond_dim_out in cond_hidden_dims:
                cond_layers.extend([
                    Linear(cond_dim_in, cond_dim_out),
                    activation
                ])
                cond_dim_in = cond_dim_out

            self.cond_mlp = nn.Sequential(*cond_layers)
            final_cond_dim = cond_dim_in
        # main MLP layers

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))

        self.dim_outs = []
        layers = []

        for i, (dim_in, dim_out) in enumerate(dim_in_out, start = 1):
            is_last = i == len(dim_in_out)

            layer = Linear(dim_in, dim_out, bias = bias)

            layer_activation = nn.Identity()

            if not is_last or activate_last:
                layer_activation = activation

            layers.append(ModuleList([
                layer,
                layer_activation
            ]))

            self.dim_outs.append(dim_out)

        self.layers = ModuleList(layers)

        # project gamma beta

        self.to_gamma_beta = Linear(final_cond_dim, sum(self.dim_outs) * 2)

        # init identity

        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        x,
        cond
    ):
        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        # maybe prepare cond

        if exists(self.cond_mlp):
            cond = self.cond_mlp(cond)

        # project gamma beta

        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = gamma_beta.chunk(2, dim = -1)

        # split per layer

        gammas = gamma.split(self.dim_outs, dim = -1)
        betas = beta.split(self.dim_outs, dim = -1)

        # feed forward

        for (linear, act), layer_gamma, layer_beta in zip(self.layers, gammas, betas):
            x = linear(x)

            if x.ndim == 3 and layer_gamma.ndim == 2:
                layer_gamma = rearrange(layer_gamma, 'b d -> b 1 d')
                layer_beta = rearrange(layer_beta, 'b d -> b 1 d')

            x = x * (layer_gamma + 1.) + layer_beta

            x = act(x)

        if not self.squeeze_out:
            return x

        return rearrange(x, '... 1 -> ...')

# factory function

def create_filmable_mlp(
    dim,
    depth,
    *,
    cond_dim,
    dim_in = None,
    dim_out = None,
    bias = True,
    squeeze_out = False,
    **mlp_kwargs
):
    dims = (dim,) * (depth + 1)

    if exists(dim_in):
        dims = (dim_in, *dims)

    if exists(dim_out):
        dims = (*dims, dim_out)

    return FiLMableMLP(
        *dims,
        cond_dim = cond_dim,
        bias = bias,
        squeeze_out = squeeze_out,
        **mlp_kwargs
    )
