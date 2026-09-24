import pytest
import torch
from torch import nn, allclose
import torch.nn.functional as F
param = pytest.mark.parametrize

@param('use_rmsnorm', (False, True))
@param('use_bias', (False, True))
def test_weight_only_mlp(use_rmsnorm, use_bias):
    from x_mlps_pytorch.weight_only_mlp import WeightOnlyMLP

    mlp = WeightOnlyMLP(2, 8, 1, use_rmsnorm = use_rmsnorm, use_bias = use_bias, squeeze_out = True)

    x1, x2 = torch.randn(3, 1), torch.randn(3, 1)

    out = mlp([x1, x2])

    assert out.shape == (3,)
    assert allclose(out, mlp(torch.cat((x1, x2), dim = -1)))

    # only rank-2 weight matrices, no biases and no gammas

    named_params = list(mlp.named_parameters())

    assert all(name.endswith('.weight') for name, _ in named_params)
    assert all(param.ndim == 2 for _, param in named_params)

    # a folded bias reserves an extra input column, otherwise regular bias free linears

    for (dim_in, dim_out), layer in zip(((2, 8), (8, 1)), mlp.layers):
        assert layer.linear.weight.shape == (dim_out, dim_in + int(use_bias))

    out.sum().backward()
    assert all(param.grad is not None for _, param in named_params)

@param('use_bias', (False, True))
@param('bias_left', (False, True))
def test_weight_only_mlp_bias_equivalence(use_bias, bias_left):
    from x_mlps_pytorch.weight_only_mlp import WeightOnlyMLP

    torch.manual_seed(0)

    ref = nn.Sequential(nn.Linear(4, 16, bias = use_bias), nn.ReLU(), nn.Linear(16, 3, bias = use_bias))
    mlp = WeightOnlyMLP(4, 16, 3, use_bias = use_bias, bias_left = bias_left)

    # fold each bias into the first or last column of its weight

    with torch.no_grad():
        for layer, linear in zip(mlp.layers, (ref[0], ref[2])):
            if not use_bias:
                layer.linear.weight.copy_(linear.weight)
                continue

            bias = linear.bias[:, None]
            columns = (bias, linear.weight) if bias_left else (linear.weight, bias)
            layer.linear.weight.copy_(torch.cat(columns, dim = -1))

    x = torch.randn(5, 4)
    assert allclose(mlp(x), ref(x), atol = 1e-6)

@param('use_bias', (False, True))
@param('bias_left', (False, True))
def test_weight_only_mlp_rmsnorm_equivalence(use_bias, bias_left):
    from x_mlps_pytorch.weight_only_mlp import WeightOnlyMLP

    torch.manual_seed(0)

    mlp = WeightOnlyMLP(4, 8, 3, use_rmsnorm = True, use_bias = use_bias, bias_left = bias_left)
    x = torch.randn(5, 4)

    # matching parameter free rmsnorm pipeline over the same weights

    def split(weight):
        if not use_bias:
            return None, weight
        return (weight[:, 0], weight[:, 1:]) if bias_left else (weight[:, -1], weight[:, :-1])

    hidden_bias, hidden_weight = split(mlp.layers[0].linear.weight)
    out_bias, out_weight = split(mlp.layers[1].linear.weight)

    h = F.rms_norm(x, (4,), eps = 1e-5)
    h = F.relu(F.linear(h, hidden_weight, hidden_bias))
    h = F.rms_norm(h, (8,), eps = 1e-5)
    h = F.linear(h, out_weight, out_bias)

    assert allclose(mlp(x), h, atol = 1e-6)
