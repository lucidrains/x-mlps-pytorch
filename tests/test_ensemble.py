import pytest

import torch
from torch import cat
from torch.nn import Module

def test_critic():
    from x_mlps_pytorch.ensemble import Ensemble
    from x_mlps_pytorch.normed_mlp import MLP

    critic = MLP(10, 5, 1)
    state = torch.randn(2, 10)

    assert critic(state).shape == (2, 1)
    critics = Ensemble(critic, 10)

    assert critics(state).shape == (10, 2, 1)

    subset_ids = torch.tensor([0, 3, 5])
    assert critics(state, ids = subset_ids).shape == (3, 2, 1)

    subset_ids = torch.tensor([0, 0])
    assert critics(state, ids = subset_ids, each_batch_sample = True).shape == (2, 1)

    critic = critics.get_one(2)
    assert critic(state).shape == (2, 1)

    assert critics.forward_one(state, id = 0).shape == (2, 1)

    critic = critics.get_one([2, 3], weights = [0.1, 0.2])
    assert critic(state).shape == (2, 1)

def test_critic_multiple_args():
    from x_mlps_pytorch.ensemble import Ensemble
    from x_mlps_pytorch.normed_mlp import MLP

    class Critic(Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, states, actions):
            return self.net(torch.cat((states, actions), dim = -1))

    critic = Critic(MLP(12, 5, 1))

    state = torch.randn(2, 10)
    action = torch.randn(2, 2)

    assert critic(state, action).shape == (2, 1)
    critics = Ensemble(critic, 10)

    assert critics(state, action).shape == (10, 2, 1)

    subset_ids = torch.tensor([0, 3, 5])
    assert critics(state, action, ids = subset_ids).shape == (3, 2, 1)

    critic = critics.get_one(2)
    assert critic(state, action).shape == (2, 1)

    critic = critics.get_one([2, 3], weights = [0.1, 0.2])
    assert critic(state, action).shape == (2, 1)
