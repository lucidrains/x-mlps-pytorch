import pytest

def test_critic():
    import torch
    from x_mlps_pytorch.ensemble import Ensemble
    from x_mlps_pytorch.normed_mlp import MLP

    critic = MLP(10, 5, 1)
    state = torch.randn(2, 10)

    assert critic(state).shape == (2, 1)
    critics = Ensemble(critic, 10)

    assert critics(state).shape == (10, 2, 1)

    subset_ids = torch.tensor([0, 3, 5])
    assert critics(state, ids = subset_ids).shape == (3, 2, 1)

    critic = critics.get_one(2)
    assert critic(state).shape == (2, 1)

    critic = critics.get_one([2, 3], weights = [0.1, 0.2])
    assert critic(state).shape == (2, 1)
