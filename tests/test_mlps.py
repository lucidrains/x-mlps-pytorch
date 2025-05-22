import pytest
import torch

def test_mlp():
    from x_mlps_pytorch.mlp import MLP

    mlp = MLP(256, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)

# with depth

def test_create_mlp():
    from x_mlps_pytorch.mlp import create_mlp

    mlp = create_mlp(
        dim = 128,
        dim_in = 256,
        dim_out = 64,
        depth = 4
    )

    # same as MLP(256, 128, 128, 128, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)
