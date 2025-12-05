import pytest
import torch
param = pytest.mark.parametrize

def identity(t):
    return t

def to_half(t):
    return t.half()

@param('low_rank', (None, 2))
@param('test_tensor_noise_scale', (False, True))
@param('float16', (False, True))
def test_noisable(
    low_rank,
    test_tensor_noise_scale,
    float16
):
    from torch import nn
    from random import randrange

    from x_mlps_pytorch.noisable import Noisable

    # first define module, wrap with Noisable, make sure forward is the same

    linear = nn.Linear(32, 64)

    noisable_linear = Noisable(linear, low_rank = low_rank)

    x = torch.randn(3, 32)

    maybe_to_float16 = to_half if float16 else identity

    x = maybe_to_float16(x)
    linear = maybe_to_float16(linear)

    out = linear(x)
    not_noised_out = noisable_linear(x)

    assert out.shape == (3, 64)
    assert torch.allclose(out, not_noised_out)

    # make sure can noise by passing in {param_name: noise}

    noised_out = noisable_linear(x, noise_for_params = dict(weight = maybe_to_float16(torch.randn(64, 32))))

    assert not torch.allclose(out, noised_out)

    # make sure can noise by passing in {param_name: seed}

    seed = randrange(int(1e7))
    noised_out = noisable_linear(x, noise_for_params = dict(weight = seed), noise_scale_for_params = dict(weight = maybe_to_float16(torch.ones((64, 32)))))

    assert not torch.allclose(out, noised_out)

    noised_out_again = noisable_linear(x, noise_for_params = dict(weight = seed))

    # able to temporarily noise the weights of a network, for rolling out against environment in memory efficient manner

    with noisable_linear.temp_add_noise_(dict(weight = (seed, 1e-4))):
        assert not torch.allclose(out, noisable_linear(x))

    assert torch.allclose(out, noisable_linear(x), atol = 1e-2 if float16 else 1e-6)
