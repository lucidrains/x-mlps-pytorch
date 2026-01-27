import torch
from torch.nn import Module

# gradient dropout

class GradientDropout(Module):
    def __init__(
        self,
        prob
    ):
        super().__init__()
        self.prob = prob
        self.scale = (1. - prob) ** -1

    def forward(self, t):

        if not t.requires_grad or not self.training:
            return t

        mask = torch.full_like(t, self.prob).bernoulli()

        # this trick was initially employed in a sparse memory update paper from Lin et al.
        # refashioned as a way to do easy gradient dropout

        out = t * (1. - mask) + t.detach() * mask

        return out * self.scale

# quick test

if __name__ == '__main__':
    grad_dropout = GradientDropout(0.5)

    t = torch.randn(50, requires_grad = True)

    grad_dropout(t).sum().backward()

    print(t.grad) # about half of gradients dropped
