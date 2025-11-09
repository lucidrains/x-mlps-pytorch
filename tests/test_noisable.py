import pytest
import torch
param = pytest.mark.parametrize

def test_noisable():
    from torch import nn
    from random import randrange

    from x_mlps_pytorch.noisable import Noisable

    # first define module, wrap with Noisable, make sure forward is the same

    linear = nn.Linear(32, 64)

    noisable_linear = Noisable(linear, noise_scale = 1e-2)

    x = torch.randn(3, 32)

    out = linear(x)
    not_noised_out = noisable_linear(x)

    assert out.shape == (3, 64)
    assert torch.allclose(out, not_noised_out)

    # make sure can noise by passing in {param_name: noise}

    noised_out = noisable_linear(x, noise_for_params = dict(weight = torch.randn(64, 32)))

    assert not torch.allclose(out, noised_out)

    # make sure can noise by passing in {param_name: seed}

    seed = randrange(int(1e7))
    noised_out = noisable_linear(x, noise_for_params = dict(weight = seed))

    assert not torch.allclose(out, noised_out)

    noised_out_again = noisable_linear(x, noise_for_params = dict(weight = seed))

    # able to temporarily noise the weights of a network, for rolling out against environment in memory efficient manner

    with noisable_linear.temp_add_noise_(dict(weight = (seed, 1e-4))):
        assert not torch.allclose(out, noisable_linear(x))

    assert torch.allclose(out, noisable_linear(x))
