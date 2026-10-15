import pytest
import torch
import torch.nn.functional as F

param = pytest.mark.parametrize

def one_hot(actions, num_actions):
    return F.one_hot(actions.long(), num_actions).float()

def test_past_action_encoder_none():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2)

    out = encoder(None)

    assert out.shape == (3,)
    assert torch.equal(out, torch.tensor([0., 0., 0.]))

def test_past_action_encoder_present():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2)

    actions = one_hot(torch.tensor([1, 0]), 2)
    out = encoder(actions)

    assert out.shape == (2, 3)
    assert torch.equal(out[:, :2], actions)
    assert torch.equal(out[:, -1], torch.ones(2))

def test_past_action_encoder_no_presence_flag():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2, presence_flag = False)

    assert encoder(None).shape == (2,)
    assert encoder(one_hot(torch.tensor([0, 1]), 2)).shape == (2, 2)

def test_past_action_encoder_mixed_batch():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2)

    actions = one_hot(torch.tensor([1, 0]), 2)
    has_past_actions = torch.tensor([True, False])

    out = encoder(actions, has_past_actions = has_past_actions)

    assert torch.equal(out[0], torch.tensor([0., 1., 1.]))
    assert torch.equal(out[1], torch.tensor([0., 0., 0.]))

def test_past_action_encoder_none_with_mask_batch():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2)

    has_past_actions = torch.tensor([False, False])
    out = encoder(None, has_past_actions = has_past_actions)

    assert out.shape == (2, 3)
    assert (out == 0.).all()

def test_past_action_encoder_present_bool():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(2)

    actions = one_hot(torch.tensor([0, 1]), 2)

    # a plain bool is fine — none passed is the same as nothing being present

    assert torch.equal(encoder(None), encoder(None, has_past_actions = False))
    assert torch.equal(encoder(actions), encoder(actions, has_past_actions = True))
    assert torch.equal(encoder(None, has_past_actions = False), encoder(None, has_past_actions = torch.tensor(False)))

@param('depth', (0, 2))
def test_past_action_encoder_proj(depth):
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder

    encoder = PastActionEncoder(4, dim_out = 8, depth = depth)

    assert encoder(None).shape == (8,)
    assert encoder(torch.randn(5, 4)).shape == (5, 8)

def test_past_action_encoder_gradients():
    from x_mlps_pytorch.past_action_encoder import PastActionEncoder
    from x_mlps_pytorch import PastActionEncoder as exported

    assert exported is PastActionEncoder

    encoder = PastActionEncoder(2, dim_out = 4)

    actions = one_hot(torch.tensor([1, 0]), 2)
    has_past_actions = torch.tensor([True, False])

    encoder(actions, has_past_actions = has_past_actions).sum().backward()

    for param in encoder.parameters():
        assert param.grad is not None
