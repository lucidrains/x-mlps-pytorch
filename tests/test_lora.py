import pytest
import torch
from torch import nn

from x_mlps_pytorch import LoRA


def test_lora():
    lora = LoRA(dim = 64, dim_out = 32, rank = 8, zero_init_up = False)

    x = torch.randn(4, 64)
    out = lora(x)

    assert out.shape == (4, 32)

    out.sum().backward()
    assert lora.down.weight.grad is not None
    assert lora.up.weight.grad is not None


def test_lora_leading_dims():
    lora = LoRA(dim = 128, rank = 16)

    x = torch.randn(2, 5, 7, 128)

    assert lora(x).shape == (2, 5, 7, 128)


def test_lora_fallback():
    lora = LoRA(dim = 8, dim_out = 8, rank = 16)

    assert lora.fallback
    assert isinstance(lora.linear, nn.Linear)

    x = torch.randn(3, 8)
    out = lora(x)

    assert out.shape == (3, 8)

    out.sum().backward()
    assert lora.linear.weight.grad is not None


def test_lora_assert_rank_less_than_dim():
    with pytest.raises(AssertionError):
        LoRA(dim = 8, dim_out = 8, rank = 16, assert_rank_less_than_dim = True)


def test_lora_defaults_to_identity():
    lora = LoRA(dim = 64, rank = 8)

    assert torch.all(lora.up.weight == 0)

    x = torch.randn(4, 64)

    assert torch.all(lora(x) == 0)


def test_lora_fallback_identity_init():
    lora = LoRA(dim = 8, dim_out = 8, rank = 16)

    assert lora.fallback

    x = torch.randn(3, 8)

    assert torch.all(lora(x) == 0)
