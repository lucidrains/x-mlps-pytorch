import pytest
import torch

def test_mlp():
    from x_mlps.x_mlps import MLP

    mlp = MLP(256, 128, 64)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == (7, 3, 64)
