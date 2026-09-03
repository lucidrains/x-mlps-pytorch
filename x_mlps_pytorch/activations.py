import torch
from math import sqrt
from torch import tensor
from torch.nn import Module, Parameter, ReLU

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# relu squared with optional signing
# signed variant is the odd extension, sign(x)·x² = x·|x|, so negative inputs get a response

def relu_squared(x, signed = False):
    if signed:
        return x.square() * x.sign()

    return x.relu().square()

class ReluSquared(Module):
    def __init__(self, signed = False):
        super().__init__()
        self.signed = signed

    def forward(self, x):
        return relu_squared(x, signed = self.signed)

# star relu - relu squared with learned scale and bias (CaFormer paper)
# defaults give zero mean, unit variance outputs for a unit gaussian input

class StarRelu(Module):
    def __init__(
        self,
        signed = False,
        alpha = None,
        beta = None
    ):
        super().__init__()
        self.relu_squared = ReluSquared(signed = signed)

        # signed variant is x·|x|, odd and zero mean over a unit gaussian, with variance 3

        if signed:
            alpha = default(alpha, 1 / sqrt(3))
            beta = default(beta, 0.)
        else:
            alpha = default(alpha, 1 / sqrt(5 / 4))
            beta = default(beta, -alpha / 2)

        self.alpha = Parameter(tensor(alpha))
        self.beta = Parameter(tensor(beta))

    def forward(self, x):
        return self.alpha * self.relu_squared(x) + self.beta

# sugar-(bsilu | nelu)

class BSiLU(Module):
    # eq (7) in paper

    def __init__(self, alpha = 1.67):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        α = self.alpha
        return (x + α) * x.sigmoid() - α / 2

class NeLU(Module):
    def __init__(self, alpha = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        α = self.alpha
        return -α / (1. + x.square())

class StraightThrough(Module):
    def __init__(
        self,
        forward_fn: Module,
        backward_fn: Module
    ):
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x):
        hard = self.forward_fn(x)

        if not x.requires_grad:
            return hard

        soft = self.backward_fn(x)

        # straight-through during training

        return soft + (hard - soft).detach()

class Sugar(Module):
    def __init__(
        self,
        forward_fn: Module,
        backward_fn: Module,
        neg_region_only = False
    ):
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn
        self.neg_region_only = neg_region_only

    def forward(self, x):
        forward_out = self.forward_fn(x)

        if not x.requires_grad:
            return forward_out

        backward_out = self.backward_fn(x)

        # maybe only neg region for backward function gradients

        soft = torch.where(x > 0, forward_out, backward_out) if self.neg_region_only else backward_out

        # straight-through during training

        return soft + (forward_out - soft).detach()

# the one that beat gelu in transformer setting for me

def ReluNelu(alpha = 0.05):
    return Sugar(ReLU(), NeLU(alpha))

# sugar bsilu - witnessed effects in a relu attention

def SugarBSiLU(alpha = 1.67, neg_region_only = False):
    return Sugar(ReLU(), BSiLU(alpha), neg_region_only = neg_region_only)
