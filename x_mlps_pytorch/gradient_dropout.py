import torch
from torch.nn import Module

# gradient dropout

class GradientDropout(Module):
    def __init__(
        self,
        prob
    ):
        super().__init__()
        assert 0. <= prob < 1.

        self.prob = prob
        self.keep_prob = 1. - prob
        self.scale = self.keep_prob ** -1

    def forward(self, t):

        if not t.requires_grad or not self.training:
            return t

        keep = torch.rand_like(t) < self.keep_prob
        grad_mask = keep.float() * self.scale

        # this trick was initially employed in a sparse memory update paper from Lin et al.
        # refashioned as a way to do easy gradient dropout

        return t * grad_mask + t.detach() * (1. - grad_mask)

# quick test

if __name__ == '__main__':
    grad_dropout = GradientDropout(0.5)

    t = torch.randn(50, requires_grad = True)

    out = grad_dropout(t)
    out.sum().backward()

    assert torch.allclose(t, out)

    print(t.grad) # about half of gradients dropped
