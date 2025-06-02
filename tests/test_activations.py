
import pytest

import torch
from x_mlps_pytorch.activations import ReluNelu

def test_relu_nelu():
    inp = torch.randn(3)
    out = ReluNelu(0.01)(inp)

    assert inp.shape == out.shape
