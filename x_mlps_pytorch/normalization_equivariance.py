# normalization equivariant wrapper
# Saied & Fleuret - https://arxiv.org/abs/2605.08193
# Kim et al - https://openreview.net/forum?id=cGDAkQo1C0p

import torch
from torch import nn
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

# functional

def rmsnorm(
    x,
    dim = -1,
    eps = 1e-6,
    return_inverse = False,
    return_stats = False
):
    rms = x.pow(2).mean(dim = dim, keepdim = True).add(eps).sqrt()
    out = x / rms

    def inverse(out):
        return out * rms

    if not return_inverse and not return_stats:
        return out

    ret = (out,)

    if return_inverse:
        ret = (*ret, inverse)

    if return_stats:
        ret = (*ret, rms)

    return ret


def layernorm(
    x,
    dim = -1,
    eps = 1e-6,
    return_inverse = False,
    return_stats = False
):
    mu = x.mean(dim = dim, keepdim = True)
    std = x.var(dim = dim, keepdim = True, unbiased = False).add(eps).sqrt()
    out = (x - mu) / std

    def inverse(out):
        return out * std + mu

    if not return_inverse and not return_stats:
        return out

    ret = (out,)

    if return_inverse:
        ret = (*ret, inverse)

    if return_stats:
        ret = (*ret, (mu, std))

    return ret

# modules

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-6,
        elementwise_affine = True
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(
        self,
        x,
        return_inverse = False,
        return_stats = False
    ):
        weight = self.weight

        out, inverse, rms = rmsnorm(x, eps = self.eps, return_inverse = True, return_stats = True)

        if exists(weight):
            out = out * weight
            base_inverse = inverse
            inverse = lambda out: base_inverse(out / weight)

        if not return_inverse and not return_stats:
            return out

        ret = (out,)

        if return_inverse:
            ret = (*ret, inverse)

        if return_stats:
            ret = (*ret, rms)

        return ret


class LayerNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-6,
        elementwise_affine = True
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(
        self,
        x,
        return_inverse = False,
        return_stats = False
    ):
        weight = self.weight

        out, inverse, (mu, std) = layernorm(x, eps = self.eps, return_inverse = True, return_stats = True)

        if exists(weight):
            out = out * weight
            base_inverse = inverse
            inverse = lambda out: base_inverse(out / weight)

        if not return_inverse and not return_stats:
            return out

        ret = (out,)

        if return_inverse:
            ret = (*ret, inverse)

        if return_stats:
            ret = (*ret, (mu, std))

        return ret


if __name__ == '__main__':
    x = torch.randn(2, 5, 8)
    backbone = nn.Identity()

    x_norm, inverse = rmsnorm(x, return_inverse = True)
    out = inverse(backbone(x_norm))
    assert torch.allclose(out, x, atol = 1e-5)
