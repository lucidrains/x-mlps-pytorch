
import pytest

import torch
from x_mlps_pytorch.activations import ReluNelu, ReluSquared, StarRelu

def test_relu_nelu():
    inp = torch.randn(3)
    out = ReluNelu(0.01)(inp)

    assert inp.shape == out.shape

def test_signed_relu_squared():
    inp = torch.randn(3, 4).abs() * 2 - 1
    out = ReluSquared(signed = True)(inp)

    assert torch.allclose(out, inp.square() * inp.sign())
    assert torch.allclose(out, -ReluSquared(signed = True)(-inp))
    assert (out < 0).any() and (out > 0).any()

    inp.requires_grad = True
    out = ReluSquared(signed = True)(inp)
    out.sum().backward()

    assert torch.allclose(inp.grad, 2 * inp.abs())

def test_star_relu_defaults():
    assert StarRelu().beta.item() == pytest.approx(-0.5 / 1.25 ** 0.5)
    assert StarRelu().alpha.item() == pytest.approx(1 / 1.25 ** 0.5)

    assert StarRelu(signed = True).beta.item() == 0.
    assert StarRelu(signed = True).alpha.item() == pytest.approx(1 / 3 ** 0.5)

def test_star_relu_signed_unit_normalized():
    torch.manual_seed(0)
    inp = torch.randn(1_000_000)
    out = StarRelu(signed = True)(inp)

    assert out.mean().abs() < 1e-2
    assert out.std().sub(1).abs() < 1e-2
