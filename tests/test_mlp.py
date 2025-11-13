import pytest
import torch
param = pytest.mark.parametrize

def test_mlp():
    from x_mlps_pytorch.mlp import MLP

    mlp = MLP(256, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)

# with depth

def test_create_mlp():
    from x_mlps_pytorch.mlp import create_mlp

    mlp = create_mlp(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        depth = 4
    )

    # same as MLP(256, 128, 128, 128, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)

@param('latent_mlp', (False, True))
@param('condition_hadamard_hiddens', (False, True))
def test_latent_conditioned_mlp(
    latent_mlp,
    condition_hadamard_hiddens
):
    from x_mlps_pytorch.mlp_with_latent import create_mlp

    mlp = create_mlp(256, 4, dim_in = 128, dim_out = 128, dim_latent = 33, latent_mlp = latent_mlp, condition_hiddens = condition_hadamard_hiddens)

    x = torch.randn(7, 3, 128)
    latent = torch.randn(7, 33)

    assert mlp(x, latent = latent).shape == x.shape

@param('rmsnorm', (False, True))
def test_mlp_with_norms(
    rmsnorm
):
    from x_mlps_pytorch.normed_mlp import MLP

    mlp = MLP(256, 128, 64, use_rmsnorm = rmsnorm)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)

def test_mlp_list_tensor_input():
    from x_mlps_pytorch.mlp import MLP

    mlp = MLP(256, 128, 64)

    x = [
        torch.randn(7, 3, 128),
        torch.randn(7, 3, 64),
        torch.randn(7, 3, 64),
    ]

    assert mlp(x).shape == (7, 3, 64)
