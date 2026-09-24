import pytest
import torch
from torch import nn, allclose
import torch.nn.functional as F
param = pytest.mark.parametrize

@param('use_rmsnorm', (False, True))
@param('final_norm', (False, True))
@param('use_bias', (False, True))
def test_weight_only_feedforwards(use_rmsnorm, final_norm, use_bias):
    from x_mlps_pytorch.weight_only_ff import WeightOnlyFeedforwards

    ff = WeightOnlyFeedforwards(16, 3, dim_in = 8, dim_out = 4, use_rmsnorm = use_rmsnorm, final_norm = final_norm, use_bias = use_bias)

    x1, x2 = torch.randn(7, 3, 4), torch.randn(7, 3, 4)

    out = ff([x1, x2])

    assert out.shape == (7, 3, 4)
    assert allclose(out, ff(torch.cat((x1, x2), dim = -1)))

    # only rank-2 weight matrices, no biases and no gammas

    named_params = list(ff.named_parameters())

    assert len(named_params) == 3 * 2 + 2
    assert all(name.endswith('.weight') for name, _ in named_params)
    assert all(param.ndim == 2 for _, param in named_params)

    # a folded bias reserves an extra input column, otherwise regular bias free linears

    pad = int(use_bias)
    dims = {name: tuple(param.shape) for name, param in named_params}
    assert dims['proj_in.linear.weight'] == (16, 8 + pad)
    assert dims['proj_out.linear.weight'] == (4, 16 + pad)
    assert dims['layers.0.to_hidden.linear.weight'] == (64, 16 + pad)
    assert dims['layers.0.to_out.linear.weight'] == (16, 64 + pad)

    out.sum().backward()
    assert all(param.grad is not None for _, param in named_params)

@param('use_bias', (False, True))
@param('bias_left', (False, True))
def test_weight_only_feedforwards_bias_equivalence(use_bias, bias_left):
    from x_mlps_pytorch.weight_only_ff import WeightOnlyFeedforwards

    torch.manual_seed(0)

    ff = WeightOnlyFeedforwards(8, 2, dim_in = 4, dim_out = 3, use_bias = use_bias, bias_left = bias_left)

    # a reference stack with explicit biases

    proj_in = nn.Linear(4, 8, bias = use_bias)
    blocks = [nn.Sequential(nn.Linear(8, 32, bias = use_bias), nn.GELU(), nn.Linear(32, 8, bias = use_bias)) for _ in range(2)]
    proj_out = nn.Linear(8, 3, bias = use_bias)

    def fold(weight_only_linear, ref_linear):
        with torch.no_grad():
            if not use_bias:
                weight_only_linear.weight.copy_(ref_linear.weight)
                return

            bias = ref_linear.bias[:, None]
            columns = (bias, ref_linear.weight) if bias_left else (ref_linear.weight, bias)
            weight_only_linear.weight.copy_(torch.cat(columns, dim = -1))

    fold(ff.proj_in.linear, proj_in)
    for layer, block in zip(ff.layers, blocks):
        fold(layer.to_hidden.linear, block[0])
        fold(layer.to_out.linear, block[2])
    fold(ff.proj_out.linear, proj_out)

    x = torch.randn(5, 4)

    h = proj_in(x)
    for block in blocks:
        h = block(h) + h
    expected = proj_out(h)

    assert allclose(ff(x), expected, atol = 1e-6)

@param('use_bias', (False, True))
def test_weight_only_feedforwards_rmsnorm_equivalence(use_bias):
    from x_mlps_pytorch.weight_only_ff import WeightOnlyFeedforwards

    torch.manual_seed(0)

    ff = WeightOnlyFeedforwards(8, 2, use_rmsnorm = True, norm_after_activation = True, final_norm = True, use_bias = use_bias)

    assert all(name.endswith('.weight') for name, _ in ff.named_parameters())

    x = torch.randn(5, 8)

    # matching parameter free rmsnorm pipeline over the same weights

    h = x

    for layer in ff.layers:
        normed = F.rms_norm(h, (8,), eps = 1e-5)
        hidden = layer.to_hidden.linear.weight
        hidden_bias = hidden[:, 8] if use_bias else None
        block = F.gelu(F.linear(normed, hidden[:, :8], hidden_bias))
        block = F.rms_norm(block, (32,), eps = 1e-5)
        out = layer.to_out.linear.weight
        out_bias = out[:, 32] if use_bias else None
        block = F.linear(block, out[:, :32], out_bias)
        h = block + h

    h = F.rms_norm(h, (8,), eps = 1e-5)

    assert allclose(ff(x), h, atol = 1e-6)
