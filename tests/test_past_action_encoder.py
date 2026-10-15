import pytest
import torch
import torch.nn.functional as F

from x_mlps_pytorch import (
    ContinuousPastActions,
    DiscretePastActions
)

param = pytest.mark.parametrize

@param('discrete', (False, True))
def test_past_actions(discrete):
    num_actions, dim_out = 2, 8
    actions = torch.tensor([1, 0])

    if discrete:
        encoder = DiscretePastActions(num_actions, dim_out)
    else:
        encoder = ContinuousPastActions(num_actions, dim_out)
        actions = F.one_hot(actions, num_actions).float()

    # nothing passed — the reserved embedding row, or zeros with the presence flag off

    assert encoder(None).shape == (dim_out,)

    # explicitly passed — a plain bool is the same as all present

    all_present = encoder(actions)

    assert all_present.shape == (2, dim_out)
    assert torch.equal(all_present, encoder(actions, has_past_actions = True))

    # a mixed batch — the mask is applied to the passed action itself,
    # so the sample without a past action falls back to the absent encoding

    mixed = encoder(actions, has_past_actions = torch.tensor([True, False]))

    assert torch.equal(mixed[0], all_present[0])
    assert torch.equal(mixed[1], encoder(None, has_past_actions = torch.tensor([False]))[0])

    # gradients reach every parameter

    mixed.sum().backward()
    assert all(p.grad is not None for p in encoder.parameters())
