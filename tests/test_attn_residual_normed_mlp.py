import pytest
import torch
from torch import nn

from x_mlps_pytorch import (
    AttentionResidual,
    AttnResidualNormedMLP,
    RelativePositionBias
)

param = pytest.mark.parametrize

@param('num_streams', (1, 2, 3))
def test_attn_residual_normed_mlp(num_streams):
    mlp = AttnResidualNormedMLP(
        dim = 256,
        depth = 4,
        dim_in = 77,
        dim_out = 64,
        lora_rank = 16,
        num_streams = num_streams,
    )

    x = torch.randn(7, 3, 77)

    assert mlp(x).shape == (7, 3, 64)

@param('num_streams', (1, 2, 3))
def test_attn_residual_normed_mlp_backward(num_streams):
    mlp = AttnResidualNormedMLP(dim = 128, dim_in = 128, depth = 2, lora_rank = 16, num_streams = num_streams)

    x = torch.randn(4, 128)
    out = mlp(x)
    out.sum().backward()

    query_lora = mlp.layers[0][1].to_query[1]

    assert mlp.layers[0][0][0].weight.grad is not None
    assert query_lora.down.weight.grad is not None
    assert query_lora.up.weight.grad is not None

@param('num_streams', (1, 2, 3))
def test_attn_residual_normed_mlp_streams(num_streams):
    mlp = AttnResidualNormedMLP(
        dim = 32,
        depth = 2,
        dim_in = 8,
        dim_out = 16,
        lora_rank = 4,
        num_streams = num_streams,
    )

    x = torch.randn(5, 8)
    out, hiddens = mlp(x, return_hiddens = True)

    assert mlp.proj_in.out_features == 32 * num_streams
    assert out.shape == (5, 16)
    assert all(hidden.shape == (5, num_streams, 32) for hidden in hiddens)

@param('num_streams', (1, 2, 3))
def test_attn_residual_normed_mlp_loops(num_streams):
    mlp = AttnResidualNormedMLP(
        dim = 32,
        dim_in = 32,
        dim_out = 32,
        depth = 2,
        lora_rank = 8,
        loops = 3,
        num_streams = num_streams,
    )

    x = torch.randn(4, 32)
    out, hiddens = mlp(x, return_hiddens = True)

    assert out.shape == (4, 32)
    assert len(hiddens) == 1 + 2 * 3

@param('num_streams', (1, 2, 3))
def test_attn_residual_normed_mlp_external_hiddens(num_streams):
    mlp = AttnResidualNormedMLP(dim = 16, dim_in = 16, dim_out = 16, depth = 2, lora_rank = 4, num_streams = num_streams)

    x = torch.randn(2, 16)
    hiddens = [torch.randn(2, num_streams, 16) for _ in range(2)]
    out, accumulated = mlp(x, hiddens = hiddens, return_hiddens = True)

    assert out.shape == (2, 16)
    assert len(accumulated) == 2 + 2

def test_attn_residual_query_is_identity_at_init():
    attn_residual = AttentionResidual(dim = 32, lora_rank = 8, use_rmsnorm = True)

    query = torch.randn(3, 32)
    context = torch.randn(3, 5, 32)

    assert attn_residual(context, query = query).shape == (3, 32)

    query_lora = attn_residual.to_query[1]
    assert torch.all(query_lora.up.weight == 0)

def test_attn_residual_custom_query_activation():
    attn_residual = AttentionResidual(
        dim = 32,
        lora_rank = 8,
        use_rmsnorm = True,
        activation = nn.Identity()
    )

    assert isinstance(attn_residual.to_query[1].activation, nn.Identity)

def test_attention_residual_multiple_queries():
    attn_residual = AttentionResidual(dim = 32, lora_rank = 8, use_rmsnorm = True)

    context = torch.randn(3, 5, 32)
    query = torch.randn(3, 2, 32)

    assert attn_residual(context, query = query).shape == (3, 2, 32)

def test_attn_residual_normed_mlp_lora_rank_fallback():
    mlp = AttnResidualNormedMLP(dim = 8, depth = 2, lora_rank = 16)

    x = torch.randn(3, 8)

    assert mlp(x).shape == (3, 8)

@param('num_streams', (1, 2, 3))
def test_relative_position_bias(num_streams):
    depth = 4
    rel_pos_bias = RelativePositionBias(depth = depth, num_streams = num_streams)

    with torch.no_grad():
        rel_pos_bias.bias.copy_(torch.arange(depth + 1).float())

    out = rel_pos_bias(depth * num_streams)

    expected = torch.arange(depth).flip(0).repeat_interleave(num_streams).float()

    assert out.shape == (depth * num_streams,)
    assert torch.allclose(out, expected)

@param('num_streams', (2, 3))
def test_relative_position_bias_per_stream(num_streams):
    depth = 2
    rel_pos_bias = RelativePositionBias(depth = depth, num_streams = num_streams)

    with torch.no_grad():
        for stream in range(num_streams):
            rel_pos_bias.bias[stream] = (stream + 1) * torch.arange(depth + 1).float()

    out = rel_pos_bias(2 * num_streams)

    streams = torch.arange(num_streams) + 1
    expected = torch.cat((streams.float(), streams.float() * 0.))

    assert torch.allclose(out, expected)

@param('num_streams', (1, 2, 3))
@param('loops', (1, 2))
def test_attn_residual_normed_mlp_relative_bias(num_streams, loops):
    depth = 4
    mlp = AttnResidualNormedMLP(
        dim = 32,
        depth = depth,
        dim_in = 8,
        dim_out = 16,
        lora_rank = 4,
        loops = loops,
        num_streams = num_streams,
    )

    assert mlp.rel_pos_bias.bias.shape == (num_streams, depth * loops + 1)

    x = torch.randn(5, 8)
    out = mlp(x)
    out.sum().backward()

    assert mlp.rel_pos_bias.bias.grad is not None

def test_attn_residual_normed_mlp_relative_bias_off():
    mlp = AttnResidualNormedMLP(dim = 32, depth = 2, dim_in = 8, dim_out = 16, relative_bias = False)

    assert mlp.rel_pos_bias is None
    assert mlp(torch.randn(3, 8)).shape == (3, 16)
