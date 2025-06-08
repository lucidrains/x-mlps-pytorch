import pytest
import torch

@pytest.mark.parametrize('groups', (1, 4))
def test_mlp(groups):
    from x_mlps_pytorch.grouped_mlp import GroupedMLP

    mlp = GroupedMLP(256, 128, 64, groups = groups)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, groups, 64)

# with depth

@pytest.mark.parametrize('groups', (1, 4))
def test_create_mlp(groups):
    from x_mlps_pytorch.grouped_mlp import create_grouped_mlp

    mlp = create_grouped_mlp(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        depth = 4,
        groups = groups
    )

    # same as GroupedMLP(256, 128, 128, 128, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, groups, 64)

# test auto squeeze 1 group, so it can act as regular MLP

def test_squeeze():
    from x_mlps_pytorch.grouped_mlp import GroupedMLP

    mlp = GroupedMLP(256, 128, 64, squeeze_if_one_group = True)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)
