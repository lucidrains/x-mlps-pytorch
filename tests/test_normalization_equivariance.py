import pytest
import torch

from x_mlps_pytorch import (
    LayerNorm,
    RMSNorm,
    layernorm,
    rmsnorm
)

param = pytest.mark.parametrize


def test_rmsnorm_is_scale_invariant():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    y = rmsnorm(x)

    assert torch.allclose(rmsnorm(x * 3.7), y, atol = 1e-5)
    assert y.shape == x.shape


def test_layernorm_is_scale_and_shift_invariant():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    y = layernorm(x)

    assert torch.allclose(layernorm(x * 3.7), y, atol = 1e-5)
    assert torch.allclose(layernorm(x + 2.3), y, atol = 1e-5)
    assert torch.allclose(layernorm(x * 3.7 + 2.3), y, atol = 1e-5)


def test_rmsnorm_inverse():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    rms = x.pow(2).mean(dim = -1, keepdim = True).add(1e-6).sqrt()

    assert torch.allclose(rmsnorm(x) * rms, x, atol = 1e-5)


def test_layernorm_inverse():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    mu = x.mean(dim = -1, keepdim = True)
    std = x.var(dim = -1, keepdim = True, unbiased = False).add(1e-6).sqrt()

    assert torch.allclose(layernorm(x) * std + mu, x, atol = 1e-5)


def test_rmsnorm_returns_inverse():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    x_norm, inverse = rmsnorm(x, return_inverse = True)

    assert torch.allclose(inverse(x_norm), x, atol = 1e-5)


def test_layernorm_returns_inverse():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    x_norm, inverse = layernorm(x, return_inverse = True)

    assert torch.allclose(inverse(x_norm), x, atol = 1e-5)


@param('norm', (rmsnorm, RMSNorm, layernorm, LayerNorm))
def test_return_stats(norm):
    x = torch.randn(2, 5, 8)

    norm = norm(8) if isinstance(norm, type) else norm
    out, inverse, stats = norm(x, return_inverse = True, return_stats = True)

    reconstructed = out * stats if torch.is_tensor(stats) else out * stats[1] + stats[0]

    assert torch.allclose(inverse(out), x, atol = 1e-5)
    assert torch.allclose(reconstructed, x, atol = 1e-5)


def test_modules_return_inverse():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)

    for norm in (RMSNorm(8), LayerNorm(8)):
        with torch.no_grad():
            norm.weight.mul_(2.5)

        x_norm, inverse = norm(x, return_inverse = True)

        assert torch.allclose(inverse(x_norm), x, atol = 1e-5)


def test_wne_identity():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)
    dims = tuple(range(1, x.ndim))

    mu = x.mean(dim = dims, keepdim = True)
    std = x.var(dim = dims, keepdim = True, unbiased = False).add(1e-6).sqrt()
    y = std * ((x - mu) / std) + mu

    assert torch.allclose(y, x, atol = 1e-5)


def test_modules_match_functions():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)

    rms = RMSNorm(8)
    ln = LayerNorm(8)

    assert torch.allclose(rms(x), rmsnorm(x), atol = 1e-6)
    assert torch.allclose(ln(x), layernorm(x), atol = 1e-6)


def test_modules_have_no_bias():
    rms = RMSNorm(8)
    ln = LayerNorm(8)

    assert set(dict(rms.named_parameters())) == {'weight'}
    assert set(dict(ln.named_parameters())) == {'weight'}
    assert not hasattr(rms, 'bias')
    assert not hasattr(ln, 'bias')


def test_modules_can_be_parameter_free():
    rms = RMSNorm(8, elementwise_affine = False)
    ln = LayerNorm(8, elementwise_affine = False)

    assert list(rms.parameters()) == []
    assert list(ln.parameters()) == []


def test_global_dims():
    torch.manual_seed(0)

    x = torch.randn(2, 5, 8)

    assert rmsnorm(x, dim = (1, 2)).shape == x.shape
    assert layernorm(x, dim = (1, 2)).shape == x.shape
