import pytest
import torch
param = pytest.mark.parametrize

def test_filmable_mlp():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(256, 128, 64, cond_dim = 32)

    x = torch.randn(7, 3, 256)
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 3, 64)

def test_filmable_mlp_2d():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(256, 128, 64, cond_dim = 32)

    x = torch.randn(7, 256)
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 64)

def test_create_filmable_mlp():
    from x_mlps_pytorch.filmable_mlp import create_filmable_mlp

    mlp = create_filmable_mlp(
        dim = 128,
        depth = 3,
        dim_in = 256,
        dim_out = 64,
        cond_dim = 32
    )

    x = torch.randn(7, 3, 256)
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 3, 64)

@param('cond_prepared', (False, True))
def test_filmable_mlp_cond_prepared(cond_prepared):
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    cond_dim = 32

    if cond_prepared:
        cond_dim = 64

    mlp = FiLMableMLP(
        256, 128, 64,
        cond_dim = cond_dim,
        cond_prepared = cond_prepared
    )

    x = torch.randn(7, 3, 256)
    cond = torch.randn(7, cond_dim)

    assert mlp(x, cond).shape == (7, 3, 64)

    from x_mlps_pytorch.filmable_mlp import exists

    # assert no cond mlp

    if cond_prepared:
        assert not exists(mlp.cond_mlp)
    else:
        assert exists(mlp.cond_mlp)

def test_filmable_mlp_squeeze_out():
    from x_mlps_pytorch.filmable_mlp import create_filmable_mlp

    mlp = create_filmable_mlp(
        dim = 128,
        depth = 2,
        dim_in = 256,
        dim_out = 1,
        cond_dim = 32,
        squeeze_out = True
    )

    x = torch.randn(7, 3, 256)
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 3)

def test_filmable_mlp_activate_last():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(
        256, 128, 64,
        cond_dim = 32,
        activate_last = True
    )

    x = torch.randn(7, 256)
    cond = torch.randn(7, 32)

    out = mlp(x, cond)
    assert out.shape == (7, 64)

    # assert non-negative

    assert (out >= 0).all()

def test_filmable_mlp_with_list_input():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(256, 128, 64, cond_dim = 32)

    x = [
        torch.randn(7, 3, 128),
        torch.randn(7, 3, 64),
        torch.randn(7, 3, 64),
    ]
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 3, 64)

@param('cond_hidden_dims', (128, (128, 64)))
def test_filmable_mlp_cond_hidden_dims(cond_hidden_dims):
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(
        256, 128, 64,
        cond_dim = 32,
        cond_hidden_dims = cond_hidden_dims,
        cond_prepared = False
    )

    x = torch.randn(7, 3, 256)
    cond = torch.randn(7, 32)

    assert mlp(x, cond).shape == (7, 3, 64)

def test_filmable_mlp_identity_init():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp_uncond = FiLMableMLP(256, 128, 64, cond_dim = 32, cond_prepared = True)

    x = torch.randn(7, 256)
    cond = torch.zeros(7, 32)

    # assert identity transform

    out = mlp_uncond(x, cond)
    assert out.shape == (7, 64)

def test_filmable_mlp_gradients():
    from x_mlps_pytorch.filmable_mlp import FiLMableMLP

    mlp = FiLMableMLP(256, 128, 64, cond_dim = 32)

    x = torch.randn(7, 3, 256, requires_grad = True)
    cond = torch.randn(7, 32, requires_grad = True)

    out = mlp(x, cond)
    loss = out.sum()
    loss.backward()

    from x_mlps_pytorch.filmable_mlp import exists

    assert exists(x.grad)
    assert exists(cond.grad)

    for p in mlp.parameters():
        assert exists(p.grad)
