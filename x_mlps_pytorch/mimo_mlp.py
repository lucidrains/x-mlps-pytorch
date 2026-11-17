from __future__ import annotations
from typing import Callable
from collections import namedtuple

from torch import Tensor, cat
from torch.nn import Identity, Module, ModuleList

from einops import rearrange

from x_mlps_pytorch.mlp import MLP

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def to_module_list(modules):
    if not exists(modules):
        return None

    # one module -> one entry, ModuleList is a container

    if isinstance(modules, ModuleList):
        modules = tuple(modules)
    elif isinstance(modules, Module):
        modules = (modules,)
    else:
        modules = tuple(modules)

    assert len(modules) > 0, 'at least one module must be given'

    return ModuleList(modules)

def to_indices(indices, count):
    if not exists(indices):
        return tuple(range(count))

    if isinstance(indices, int):
        indices = (indices,)

    return tuple(indices)

def default_combine_inputs(features):
    return cat(features, dim = -1)

def squeeze_last_dim(t):
    if t.shape[-1] != 1:
        return t

    return rearrange(t, '... 1 -> ...')

# intermediates

Intermediates = namedtuple('Intermediates', ('inputs', 'trunk_input', 'trunk_output', 'outputs'))

# main class

class MimoMLP(Module):
    """
    mlp with multiple inputs and outputs

    inputs -> input modules -> `combine_inputs` (concat by default) -> trunk -> heads

    `select` picks heads by index, `select_inputs` picks inputs by index.
    selected inputs must combine to the trunk input dim.
    `return_intermediates = True` returns the selected inputs, trunk input/output, and all heads
    """

    def __init__(
        self,
        *dims,
        inputs: Module | tuple[Module, ...] | list[Module] | None = None,
        outputs: Module | tuple[Module, ...] | list[Module] | None = None,
        combine_inputs: Callable[[tuple[Tensor, ...]], Tensor] | None = None,
        squeeze_out = False,
        **mlp_kwargs
    ):
        super().__init__()

        self.squeeze_out = squeeze_out

        self.inputs = to_module_list(inputs)
        self.outputs = default(to_module_list(outputs), ModuleList([Identity()]))

        self.combine_inputs = default(combine_inputs, default_combine_inputs)

        # no hidden layers -> just combine inputs and apply heads

        self.trunk = Identity()
        self.dim_trunk_in = None

        if len(dims) > 1:
            self.trunk = MLP(*dims, squeeze_out = False, **mlp_kwargs)
            self.dim_trunk_in = dims[0]

    def forward(
        self,
        x,
        select = None,
        select_inputs = None,
        return_intermediates = False
    ):
        # inputs -> encoded features

        xs = x if isinstance(x, (list, tuple)) else (x,)

        num_inputs = len(self.inputs) if exists(self.inputs) else len(xs)
        assert len(xs) == num_inputs, f'expected {num_inputs} inputs, got {len(xs)}'

        input_indices = to_indices(select_inputs, num_inputs)

        if exists(self.inputs):
            features = tuple(self.inputs[i](xs[i]) for i in input_indices)
        else:
            features = tuple(xs[i] for i in input_indices)

        # features -> trunk input -> trunk output

        trunk_input = self.combine_inputs(features)

        # selected inputs must feed the fixed trunk width

        if exists(self.dim_trunk_in):
            assert trunk_input.shape[-1] == self.dim_trunk_in, f'combined inputs have dim {trunk_input.shape[-1]}, expected trunk input dim {self.dim_trunk_in}'

        trunk_output = self.trunk(trunk_input)

        # all intermediates

        if return_intermediates:
            outputs = tuple(module(trunk_output) for module in self.outputs)

            if self.squeeze_out:
                outputs = tuple(squeeze_last_dim(t) for t in outputs)

            return Intermediates(features, trunk_input, trunk_output, outputs)

        # selected heads

        select = to_indices(select, len(self.outputs))

        outputs = tuple(self.outputs[index](trunk_output) for index in select)

        # squeeze

        if self.squeeze_out:
            outputs = tuple(squeeze_last_dim(t) for t in outputs)

        # one output -> tensor

        return outputs[0] if len(outputs) == 1 else outputs
