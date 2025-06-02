from functools import partial

import torch
from torch.nn import Module, ReLU

# relu squared with optional signing

class ReluSquared(Module):
    def __init__(self, signed = False):
        super().__init__()
        self.signed = signed

    def forward(self, x):
        out = x.relu().square()

        if not self.signed:
            return out

        return out * x.sign()

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
        backward_fn: Module
    ):
        super().__init__()
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, x):
        forward_out = self.forward_fn(x)

        if not x.requires_grad:
            return forward_out

        backward_out = self.backward_fn(x)

        # only neg region for backward function gradients

        soft = torch.where(x > 0, forward_out, backward_out)
        
        # straight-through during training

        return soft + (forward_out - soft).detach()

# the one that beat gelu in transformer setting for me

def ReluNelu(alpha = 0.05):
    return Sugar(ReLU(), NeLU(alpha))
