from __future__ import annotations
from copy import deepcopy
from functools import wraps
from contextlib import contextmanager

import torch
from torch import is_tensor
from torch.nn import Module
from torch.func import functional_call

has_cuda = torch.cuda.is_available()

# helper functions

def exists(v):
    return v is not None

def is_empty(arr):
    return len(arr) == 0

def default(v, d):
    return v if exists(v) else d

# temporary seed

def with_seed(seed):

    def decorator(fn):

        @wraps(fn)
        def inner(*args, **kwargs):
            orig_torch_state = torch.get_rng_state()

            orig_cuda_states = None
            if has_cuda:
                orig_cuda_states = torch.cuda.get_rng_state_all()

            torch.manual_seed(seed)

            if has_cuda:
                torch.cuda.manual_seed_all(seed)

            try:
                out = fn(*args, **kwargs)

            finally:
                torch.set_rng_state(orig_torch_state)

                if has_cuda and orig_cuda_states:
                    torch.cuda.set_rng_state_all(orig_cuda_states)

            return out
        return inner

    return decorator

# wrapper

class Noisable(Module):
    def __init__(
        self,
        model: Module,
        noise_scale = 1.
    ):
        super().__init__()
        assert not is_empty(list(model.parameters()))

        self.model = model
        self.noise_scale = noise_scale

    @property
    def device(self):
        return next(self.model.parameters()).device

    @contextmanager
    def temp_add_noise_(
        self,
        noise_for_params = dict(),
        noise_scale = None,
    ):
        self.get_noised_params(noise_for_params, noise_scale = noise_scale, inplace = True)

        yield

        self.get_noised_params(noise_for_params, noise_scale = noise_scale, inplace = True, negate = True)

    def add_noise_(
        self,
        noise_for_params = dict(),
        noise_scale = None,
        negate = False
    ):
        self.get_noised_params(noise_for_params, noise_scale = noise_scale, inplace = True, negate = negate)

    def get_noised_params(
        self,
        noise_for_params = dict(),
        inplace = False,
        noise_scale = None,
        negate = False
    ):
        # get named params

        named_params = dict(self.model.named_parameters())

        # noise the params

        if not inplace:
            noised_params = deepcopy(named_params)
            return_params = noised_params
        else:
            return_params = named_params

        for name, param in named_params.items():

            param_shape = param.shape

            noise_or_seed = noise_for_params.get(name, None)
            noise_scale = default(noise_scale, self.noise_scale)

            if not exists(noise_or_seed):
                continue

            # determine the noise

            if isinstance(noise_or_seed, int):
                noise = with_seed(noise_or_seed)(torch.randn)(param_shape)

            elif isinstance(noise_or_seed, tuple) and len(noise_or_seed) == 2:

                # overriding noise scale per param

                seed, noise_scale = noise_or_seed
                noise = with_seed(seed)(torch.randn)(param_shape)

            elif is_tensor(noise_or_seed):
                noise = noise_or_seed
            else:
                raise ValueError('invalid type, noise must be float tensor or int')

            noise = noise.to(self.device)

            # scale the noise

            if noise_scale != 1.:
                noise = noise * noise_scale

            if negate:
                noise = noise * -1

            # if inplace, add directly to param, else set the new dictionary and return that

            if inplace:
                param.data.add_(noise)
            else:
                noised_params[name] = param + noise

        return return_params

    def forward(
        self,
        *args,
        noise_for_params = dict(),
        noise_scale = None,
        **kwargs
    ):
        if is_empty(noise_for_params):
            return self.model(*args, **kwargs)

        noised_params = self.get_noised_params(noise_for_params, noise_scale = noise_scale)

        # use functional call with noised params

        return functional_call(self.model, noised_params, args, kwargs)
