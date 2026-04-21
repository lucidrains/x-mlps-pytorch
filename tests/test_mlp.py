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

def test_create_mlp_depth_zero():
    from x_mlps_pytorch.mlp import create_mlp
    from x_mlps_pytorch.mlp_with_latent import create_mlp as create_mlp_latent
    from x_mlps_pytorch.normed_mlp import create_mlp as create_mlp_normed

    x = torch.randn(7, 3, 256)

    # 1. mlp
    mlp = create_mlp(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        depth = 0
    )
    assert mlp(x).shape == (7, 3, 64)
    assert isinstance(mlp, torch.nn.Linear)

    mlp_identity = create_mlp(dim = 128, depth = 0)
    assert isinstance(mlp_identity, torch.nn.Identity)

    # 2. mlp_with_latent
    mlp2 = create_mlp_latent(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        dim_latent = 33,
        depth = 0
    )
    assert mlp2(x).shape == (7, 3, 64)
    assert isinstance(mlp2, torch.nn.Linear)

    mlp2_identity = create_mlp_latent(dim = 128, dim_latent = 33, depth = 0)
    assert isinstance(mlp2_identity, torch.nn.Identity)

    # 3. normed_mlp
    mlp3 = create_mlp_normed(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        depth = 0
    )
    assert mlp3(x).shape == (7, 3, 64)
    assert isinstance(mlp3, torch.nn.Linear)

    mlp3_identity = create_mlp_normed(dim = 128, depth = 0)
    assert isinstance(mlp3_identity, torch.nn.Identity)

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

    mlp = MLP(256, 128, 128, 64, use_rmsnorm = rmsnorm)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)

@param('skip_to_output, keel_post_ln, use_orthogonal_residual', ((False, True, False), (True, False, False), (False, False, False), (False, False, True)))
def test_residual_normed_mlp(
    skip_to_output,
    keel_post_ln,
    use_orthogonal_residual
):
    from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

    mlp = ResidualNormedMLP(256, depth = 16, residual_every = 4, dim_out = 64, dim_in = 77, skip_to_output = skip_to_output, keel_post_ln = keel_post_ln, use_orthogonal_residual = use_orthogonal_residual)

    x = torch.randn(7, 3, 77)

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
