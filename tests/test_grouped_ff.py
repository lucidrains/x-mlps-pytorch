import pytest
import torch

def test_grouped_ff_one_group():
    from x_mlps_pytorch.grouped_ff import GroupedFeedforwards

    ff = GroupedFeedforwards(256, 4, dim_in = 128, dim_out = 128, squeeze_if_one_group = True)

    x = torch.randn(7, 3, 128)

    assert ff(x).shape == x.shape


def test_grouped_ff_one_group():
    from x_mlps_pytorch.grouped_ff import GroupedFeedforwards

    ff = GroupedFeedforwards(256, 4, dim_in = 128, dim_out = 128, squeeze_if_one_group = True, groups = 2)

    x = torch.randn(7, 3, 128)

    assert ff(x).shape == (7, 3, 2, 128)
