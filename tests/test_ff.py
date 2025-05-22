import pytest
import torch

def test_ff():
    from x_mlps_pytorch.ff import Feedforwards

    mlp = Feedforwards(256, 4)

    x = torch.randn(7, 3, 256)

    assert mlp(x).shape == x.shape
