import torch
from torch import nn
from torch.nn import Module

from x_mlps_pytorch.mlp import create_mlp

# functions

def exists(v):
    return v is not None

# main class

class PastActionEncoder(Module):
    """
    encodes the action that led into the current observation — `None` on the first step of an episode.

    takes `None` or a float action `... num_actions`, along with an optional `has_past_actions` bool mask
    for the samples that actually have one — the rest get zeros, and a trailing flag
    marks whether an action was present at all — then a linear or mlp projection (`depth = 0` is linear)
    """

    def __init__(
        self,
        num_actions,
        dim_out = None,
        *,
        depth = 0,
        presence_flag = True,
        activation = nn.ReLU(),
        bias = True
    ):
        super().__init__()
        self.num_actions = num_actions
        self.presence_flag = presence_flag

        dim_in = num_actions + int(presence_flag)

        # without a dim out, the encoded action is returned as is

        self.proj = nn.Identity()

        if exists(dim_out):
            self.proj = create_mlp(dim_out, depth, dim_in = dim_in, dim_out = dim_out, bias = bias, activation = activation)

    def forward(
        self,
        past_action = None,
        has_past_actions = None
    ):
        # has_past_actions — bool mask of the samples that have one, broadcastable to the leading dims of past_action

        has_past_actions = torch.as_tensor(has_past_actions) if exists(has_past_actions) else None

        if exists(past_action):
            batch_shape, device = past_action.shape[:-1], past_action.device
        elif exists(has_past_actions):
            batch_shape, device = has_past_actions.shape, has_past_actions.device
        else:
            batch_shape, device = (), None

        # nothing passed — nothing has a past action

        present = torch.full(batch_shape, exists(past_action), dtype = torch.bool, device = device)

        if not exists(past_action):
            past_action = torch.zeros(*batch_shape, self.num_actions, device = device)

        if exists(has_past_actions):
            present = has_past_actions.to(device).expand(batch_shape)
            past_action = past_action.masked_fill(~present.unsqueeze(-1), 0.)

        if self.presence_flag:
            past_action = torch.cat((past_action, present[..., None].float()), dim = -1)

        return self.proj(past_action)
