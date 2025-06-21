# https://arxiv.org/abs/2410.01131

from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nn.utils.parametrize import register_parametrization

from einops import pack, unpack, rearrange, repeat

# functions

def exists(v):
    return v is not None

def first(arr):
    return arr[0]

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t, length = 1):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert len(out) == length
    return out

def l2norm(t, dim = 1):
    return F.normalize(t, dim = dim)

def pack_with_inverse(t, pattern):
    packed, shape = pack([t], pattern)

    def inverse(out):
        return first(unpack(out, shape, pattern))

    return packed, inverse

# norming of the weights

def norm_weights_(parent_module: Module):
    for module in parent_module.modules():
        if not isinstance(module, NormLinear):
            continue

        module.norm_weights_()

# scale

class Scale(Module):
    def __init__(
        self,
        dim,
        init = 1.,
        scale = 1.,
        groups = 1
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim * groups, 1) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale

# residual slerp update with learned scale

class Residual(Module):
    def __init__(
        self,
        fn: Module,
        dim: int,
        init: float,
        scale: float | None = None,
        groups = 1
    ):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim ** -0.5), groups = groups)

    def forward(self, x, **kwargs):
        residual = x

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        out = l2norm(out)
        out = l2norm(residual.lerp(out, self.branch_scale()))

        if tuple_output:
            out = (out, *rest)

        return out

# for use with parametrize

class L2Norm(Module):
    def __init__(
        self,
        norm_dim_in = True,
        groups = 1
    ):
        super().__init__()
        self.groups = groups
        self.norm_dim_in = norm_dim_in

    def forward(self, weight):
        g = self.groups
        assert weight.ndim == 3

        if self.norm_dim_in:
            weight = rearrange(weight, 'o (g i) k -> o g i k', g = g)
            weight = l2norm(weight, dim = -1)
            weight = rearrange(weight, 'o g i k -> o (g i) k')
        else:
            weight = rearrange(weight, '(g o) i k -> g o i k', g = g)
            weight = l2norm(weight, dim = -1)
            weight = rearrange(weight, 'g o i k -> (g o) i k')

        return weight

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 1,
        norm_dim_in = True,
        parametrize = True
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out

        self.linear = nn.Conv1d(dim * groups, dim_out * groups, 1, groups = groups, bias = False)

        self.parametrize = parametrize
        self.l2norm = L2Norm(norm_dim_in)

        if parametrize:
            register_parametrization(
                self.linear,
                'weight',
                self.l2norm
            )

        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original

            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# feedforward

class nFeedforward(Module):
    def __init__(
        self,
        dim,
        *,
        groups = 1,
        expand_factor = 4,
        manual_norm_weights = False,
        s_hidden_init = 1.,
        s_hidden_scale = 1.,
        s_gate_init = 1.,
        s_gate_scale = 1.,
    ):
        super().__init__()
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights)

        self.dim = dim
        self.expand_factor = expand_factor

        dim_inner = int(dim * expand_factor * 2 / 3)

        self.dim_inner = dim_inner

        self.to_hidden = NormLinear_(dim, dim_inner, groups = groups)
        self.to_gate = NormLinear_(dim, dim_inner, groups = groups)

        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale, groups = groups)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale, groups = groups)

        self.to_out = NormLinear_(dim_inner, dim, groups = groups, norm_dim_in = False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale()
        gate = gate * self.gate_scale() * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)

# classes

class nGroupedFeedforwards(Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        groups = 1,
        squeeze_if_one_group = False,
        dim_in = None,
        dim_out = None,
        ff_expand_factor = 4.,
        input_preserve_magnitude = False,
        constant_shift = 3., # simbav2 concatted a constant of 3. before l2norm of the input to preserve magnitude information
        manual_norm_weights = False,
        # below are all the scale related hyperparameters, for controlling effective relative learning rates throughout the network
        alpha_init: float | None = None,  # this would set the alpha init for all residuals, but would be overridden by alpha_ff_init if they are specified
        alpha_attn_init: float | tuple[float, ...] | None = None,
        alpha_attn_scale: float | tuple[float, ...] | None = None,
        alpha_ff_init: float | tuple[float, ...] | None = None,
        alpha_ff_scale: float | tuple[float, ...] | None = None,
        s_ff_hidden_init: float | tuple[float, ...] = 1.,
        s_ff_hidden_scale: float | tuple[float, ...] = 1.,
        s_ff_gate_init: float | tuple[float, ...] = 1.,
        s_ff_gate_scale: float | tuple[float, ...] = 1.,

    ):
        super().__init__()
        NormLinear_ = partial(NormLinear, parametrize = not manual_norm_weights)

        self.groups = groups
        self.squeeze_if_one_group = groups == 1 and squeeze_if_one_group

        self.dim = dim
        self.depth = depth
        self.ff_expand_factor = ff_expand_factor

        alpha_init = default(alpha_init, 1. / depth)

        self.layers = ModuleList([])

        scale_hparams = (
            alpha_attn_init,
            alpha_attn_scale,
            alpha_ff_init,
            alpha_ff_scale,
            s_ff_hidden_init,
            s_ff_hidden_scale,
            s_ff_gate_init,
            s_ff_gate_scale
        )

        scale_hparams = tuple(cast_tuple(hparam, depth) for hparam in scale_hparams)

        for (
            alpha_attn_init_,
            alpha_attn_scale_,
            alpha_ff_init_,
            alpha_ff_scale_,
            s_ff_hidden_init_,
            s_ff_hidden_scale_,
            s_ff_gate_init_,
            s_ff_gate_scale_
        ) in zip(*scale_hparams):

            ff = nFeedforward(
                dim,
                groups = groups,
                expand_factor = ff_expand_factor,
                manual_norm_weights = manual_norm_weights,
                s_hidden_init = s_ff_hidden_init_,
                s_hidden_scale = s_ff_hidden_scale_,
                s_gate_init = s_ff_gate_init_,
                s_gate_scale = s_ff_gate_scale_,
            )

            ff_with_residual = Residual(
                ff,
                dim,
                default(alpha_ff_init_, alpha_init),
                default(alpha_ff_scale_, dim ** -0.5),
                groups = groups
            )

            self.layers.append(ff_with_residual)

        # appending the magnitude

        self.input_preserve_magnitude = input_preserve_magnitude
        self.constant_shift = constant_shift

        # projecting in

        self.need_proj_in = exists(dim_in) or input_preserve_magnitude

        if self.need_proj_in:
            dim_in = default(dim_in, dim)
            dim_constant_shift = int(input_preserve_magnitude)

            self.proj_in = NormLinear(dim_in + dim_constant_shift, dim, groups = groups, norm_dim_in = False)
            self.proj_in_scale = Scale(dim, groups = groups)

        # projecting out

        self.need_proj_out = exists(dim_out)

        if self.need_proj_out:
            self.proj_out = NormLinear_(dim, dim_out, groups = groups)
            self.proj_out_scale = Scale(dim_out, 1., dim ** -0.5, groups = groups)

    @torch.no_grad()
    def norm_weights_(self):
        norm_weights_(self)

    def forward(
        self,
        x        
    ):

        x, inverse_pack = pack_with_inverse(x, 'b * d')

        if self.input_preserve_magnitude:
            x = F.pad(x, (0, 1), value = self.constant_shift)
            x = l2norm(x)

        x = rearrange(x, 'b n d -> b d n')

        x = repeat(x, 'b d n -> b (g d) n', g = self.groups)

        if self.need_proj_in:
            x = self.proj_in(x) * self.proj_in_scale()
            x = l2norm(x)

        for ff in self.layers:
            x = ff(x)

        if self.need_proj_out:
            x = self.proj_out(x) * self.proj_out_scale()

        x = rearrange(x, 'b d n -> b n d')

        x = inverse_pack(x)

        x = rearrange(x, 'b ... (g d) -> b ... g d', g = self.groups)

        if not self.squeeze_if_one_group:
            return x

        return rearrange(x, 'b ... 1 d -> b ... d')

# copy-pasteable file

if __name__ == '__main__':

    nff = nGroupedFeedforwards(512, 4, dim_in = 128, dim_out = 128, input_preserve_magnitude = True, squeeze_if_one_group = True)
    x = torch.randn((2, 128))

    assert nff(x).shape == x.shape
