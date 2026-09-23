from __future__ import annotations

import torch
from torch import nn, cat, stack, Tensor
from torch.nn import Module, ModuleList, Identity

import einx
from einops import einsum, rearrange, repeat

from x_mlps_pytorch.norms import RMSNorm, LayerNorm
from x_mlps_pytorch.lora import LoRA

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# learned relative position bias, shared across all attention residuals
# indexed from the most recent hidden (0) into the past, one learned row per stream

class RelativePositionBias(Module):
    def __init__(
        self,
        depth,
        num_streams = 1,
        init = 0.
    ):
        super().__init__()
        self.depth = depth
        self.num_streams = num_streams

        bias = float(init) * torch.arange(depth + 1, dtype = torch.float32)
        self.bias = nn.Parameter(repeat(bias, 'l -> s l', s = num_streams).contiguous())

    def forward(self, num_context_hiddens):
        num_layers = num_context_hiddens // self.num_streams

        layer_indices = repeat(torch.arange(num_layers, device = self.bias.device), 'l -> (l s)', s = self.num_streams)
        ages = ((num_layers - 1) - layer_indices).clamp(max = self.depth)

        stream_indices = repeat(torch.arange(self.num_streams, device = self.bias.device), 's -> (l s)', l = num_layers)

        return self.bias[stream_indices, ages]

# attention residual

class AttentionResidual(Module):
    """
    attention over all preceding hiddens, with the query conditioned on the latest hidden
    through a low-rank residual, which starts as identity

    https://arxiv.org/abs/2603.15031
    """

    def __init__(
        self,
        dim,
        lora_rank = 16,
        norm_fn: Module | None = None,
        use_rmsnorm = False,
        activation = nn.SiLU(),
        rel_pos_bias: Module | None = None,
    ):
        super().__init__()
        self.scale = dim ** -0.5

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

        self.norm = norm_fn(dim)
        self.to_keys = norm_fn(dim)
        self.rel_pos_bias = rel_pos_bias

        self.to_query = nn.Sequential(
            norm_fn(dim),
            LoRA(dim, rank = lora_rank, activation = activation)
        )

    def forward(
        self,
        context: list[Tensor] | Tensor,
        query: Tensor
    ):
        if isinstance(context, (list, tuple)):
            context = stack(context, dim = -2)

        query = self.norm(query + self.to_query(query))
        keys = self.to_keys(context)

        is_multi_query = query.ndim == context.ndim

        if not is_multi_query:
            query = rearrange(query, '... d -> ... 1 d')

        sim = einsum(query, keys, '... s d, ... j d -> ... s j') * self.scale

        if exists(self.rel_pos_bias):
            num_context_hiddens = context.shape[-2]
            sim = einx.add('... s j, j -> ... s j', sim, self.rel_pos_bias(num_context_hiddens))

        attn = sim.softmax(dim = -1)
        out = einsum(attn, context, '... s j, ... j d -> ... s d')

        if not is_multi_query:
            out = rearrange(out, '... 1 d -> ... d')

        return out

# main class

class AttnResidualNormedMLP(Module):
    """
    residual normed mlp, with residual connections replaced by attention-aggregated
    residuals over all layer hiddens, each layer output added onto its input first

    the attention is shared across layers and carries a learned relative position
    bias, indexed from the most recent hidden into the past, with one learned row
    for each stream

    with `num_streams` greater than 1, the input is projected to that many streams
    before being flattened and combined at the output projection

    https://arxiv.org/abs/2601.21582
    https://arxiv.org/abs/2603.15031
    """

    def __init__(
        self,
        dim,
        depth = 32,
        dim_in = None,
        dim_out = None,
        activation = nn.SiLU(),
        bias = True,
        norm_fn: Module | None = None,
        use_rmsnorm = False,
        final_norm = True,
        lora_rank = 16,
        loops = 1,
        num_streams = 1,
        relative_bias = True,
        relative_bias_init = 0.,
    ):
        super().__init__()

        self.loops = loops
        self.num_streams = num_streams

        self.rel_pos_bias = RelativePositionBias(
            depth = depth * loops,
            num_streams = num_streams,
            init = relative_bias_init
        ) if relative_bias else None

        self.proj_in = nn.Linear(dim_in, dim * num_streams) if exists(dim_in) else Identity()
        self.proj_out = nn.Linear(dim * num_streams, dim_out) if exists(dim_out) else Identity()

        if not exists(norm_fn):
            norm_fn = RMSNorm if use_rmsnorm else LayerNorm

        self.layers = ModuleList([])

        for _ in range(depth):
            layer = nn.Sequential(
                nn.Linear(dim, dim, bias = bias),
                norm_fn(dim),
                activation,
            )

            attn_residual = AttentionResidual(
                dim,
                lora_rank = lora_rank,
                norm_fn = norm_fn,
                activation = activation,
                rel_pos_bias = self.rel_pos_bias
            )

            self.layers.append(ModuleList([layer, attn_residual]))

        self.final_norm = norm_fn(dim) if final_norm else Identity()

    def forward(
        self,
        x,
        loops: int | None = None,
        hiddens: list[Tensor] | None = None,
        return_hiddens: bool = False,
    ):

        if isinstance(x, (list, tuple)):
            x = cat(x, dim = -1)

        x = self.proj_in(x)
        x = rearrange(x, '... (s d) -> ... s d', s = self.num_streams)

        if not exists(hiddens):
            hiddens = [x]
        else:
            hiddens = list(hiddens)

        num_loops = default(loops, self.loops)
        assert num_loops >= 1

        for _ in range(num_loops):
            for layer, attn_residual in self.layers:
                out = layer(x) + x
                hiddens.append(out)

                context = rearrange(stack(hiddens, dim = -3), '... l s d -> ... (l s) d')
                x = attn_residual(context, query = out)

        out = self.final_norm(x)
        out = rearrange(out, '... s d -> ... (s d)')
        out = self.proj_out(out)

        if not return_hiddens:
            return out

        return out, hiddens

# quick test

if __name__ == '__main__':

    mlp = AttnResidualNormedMLP(
        dim = 256,
        depth = 64,
        dim_in = 77,
        dim_out = 64,
    )

    x = torch.randn(7, 3, 77)

    out = mlp(x)

    assert out.shape == (7, 3, 64)
