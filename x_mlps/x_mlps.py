import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import einsum, rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class MLP(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
