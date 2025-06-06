import pytest
import torch

def test_ff():
    from x_mlps_pytorch.ff import Feedforwards

    mlp = Feedforwards(256, 4, dim_in = 128, dim_out = 128)

    x = torch.randn(7, 3, 128)

    assert mlp(x).shape == x.shape

@pytest.mark.parametrize('preserve_magnitude', (False, True))
def test_nff(
    preserve_magnitude
):
    from x_mlps_pytorch.nff import nFeedforwards

    mlp = nFeedforwards(256, 4, preserve_magnitude = preserve_magnitude)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == x.shape
