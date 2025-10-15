from copy import deepcopy

import torch
from torch import Tensor, tensor, is_tensor

from torch.nn import Module, ParameterList
from torch.func import vmap, functional_call

from einops import einsum

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class Ensemble(Module):
    def __init__(
        self,
        net: Module,
        ensemble_size,
        init_std_dev = 2e-2
    ):
        super().__init__()
        self.net = net
        self.ensemble_size = ensemble_size

        params = dict(net.named_parameters())
        device = next(iter(params.values())).device

        ensemble_params = {name: (torch.randn((ensemble_size, *param.shape), device = device) * init_std_dev).requires_grad_() for name, param in params.items()}

        self.param_names = list(ensemble_params.keys())
        self.param_values = ParameterList(list(ensemble_params.values()))

        def _forward(params, args, kwargs):
            return functional_call(net, params, args, kwargs)

        self._forward = _forward
        self.ensemble_forward = vmap(_forward, in_dims = (0, None, None))
        self.each_batch_sample_forward = vmap(_forward, in_dims = (0, 0, 0))

    @property
    def ensemble_params(self):
        return dict(zip(self.param_names, self.param_values))

    def parameters(self):
        return iter(self.ensemble_params.values())

    @property
    def device(self):
        return next(self.parameters()).device

    def pick_params(
        self,
        indices: int | Tensor | list[int] | None = None
    ):
        params = self.ensemble_params

        if not exists(indices):
            return params

        # converts list[int] to Int

        if isinstance(indices, list):
            indices = tensor(indices, device = self.device)

        # some validation

        if is_tensor(indices):
            assert (indices < self.ensemble_size).all()
        elif isinstance(indices, int):
            assert 0 <= indices < self.ensemble_size

        params = {key: param[indices] for key, param in params.items()}
        return params

    def get_one(
        self,
        indices: int | Tensor | list[int],
        weights: list[float] = None
    ):

        params = self.pick_params(indices)

        needs_combine = not isinstance(indices, int)

        if needs_combine:
            num_nets = len(indices)

            weights = default(weights, [1. / num_nets] * num_nets)
            weights = tensor(weights, device = self.device)

            params = {key: einsum(param, weights, 'n ..., n -> ...') for key, param in params.items()}

        out = deepcopy(self.net)
        out.load_state_dict(params)

        return out

    def forward_one(
        self,
        *args,
        id: int,
        **kwargs
    ):
        params = {key: param[id] for key, param in self.ensemble_params.items()}
        return self._forward(dict(params), args, kwargs)

    def forward(
        self,
        *args,
        ids = None,
        each_batch_sample = False,
        **kwargs,
    ):

        params = self.pick_params(ids)

        fn = self.ensemble_forward if not each_batch_sample else self.each_batch_sample_forward

        return fn(dict(params), args, kwargs)
