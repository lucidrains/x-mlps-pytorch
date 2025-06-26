import pytest
import torch

def test_ff():
    from x_mlps_pytorch.ff import Feedforwards

    ff = Feedforwards(256, 4, dim_in = 128, dim_out = 128)

    x = torch.randn(7, 3, 128)

    assert ff(x).shape == x.shape

@pytest.mark.parametrize('preserve_magnitude', (False, True))
def test_nff(
    preserve_magnitude
):
    from x_mlps_pytorch.nff import nFeedforwards, norm_weights_

    ff = nFeedforwards(256, 4, input_preserve_magnitude = preserve_magnitude)

    x = torch.randn(7, 3, 256)

    assert ff(x).shape == x.shape

    norm_weights_(ff)

@pytest.mark.parametrize('input_has_seq_dim', (False, True))
@pytest.mark.parametrize('latent_mlp', (False, True))
def test_ff_with_latent(
    input_has_seq_dim,
    latent_mlp
):
    from x_mlps_pytorch.ff_with_latent import LatentConditionedFeedforwards

    ff = LatentConditionedFeedforwards(256, 4, dim_in = 128, dim_out = 128, dim_latent = 33, latent_mlp = latent_mlp)

    x = torch.randn(7, 3, 128) if input_has_seq_dim else torch.randn(7, 128)
    latent = torch.randn(7, 33)

    assert ff(x, latent = latent).shape == x.shape
