
from __future__ import annotations
from contextlib import contextmanager

import torch
from torch import is_tensor
from torch.nn import Module
from torch.func import functional_call

# helper functions

def exists(v):
    return v is not None

def is_empty(arr):
    return len(arr) == 0

def default(v, d):
    return v if exists(v) else d

# temporary seed

@contextmanager
def temp_seed(seed):
    orig_torch_state = torch.get_rng_state()

    orig_cuda_states = None
    if torch.cuda.is_available():
        orig_cuda_states = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    try:
        yield
        
    finally:
        torch.set_rng_state(orig_torch_state)

        if torch.cuda.is_available() and orig_cuda_states:
            torch.cuda.set_rng_state_all(orig_cuda_states)

# torch.randn with seed

def randn_from_seed(seed, shape, device = None):

    with temp_seed(seed):
        return torch.randn(shape, device = device)

# wrapper

class Noisable(Module):
    def __init__(
        self,
        model: Module
    ):
        super().__init__()
        assert not is_empty(list(model.parameters()))

        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(
        self,
        *args,
        noise_for_params = dict(),
        **kwargs
    ):
        if is_empty(noise_for_params):
            return self.model(*args, **kwargs)

        # get named params

        named_params = dict(self.model.named_parameters())

        # noise the params

        noised_params = dict()

        for name, param in named_params.items():

            noise_or_seed = noise_for_params.get(name, None)

            if not exists(noise_or_seed):
                continue

            if isinstance(noise_or_seed, int):
                noise = randn_from_seed(noise_or_seed, param.shape)
            elif is_tensor(noise_or_seed):
                noise = noise_or_seed
            else:
                raise ValueError('invalid type, noise must be float tensor or int')

            noised_params[name] = param + noise.to(self.device)

        # use functional call with noised params

        return functional_call(self.model, noised_params, args, kwargs)
