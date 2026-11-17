import pytest
import torch
from torch import nn

param = pytest.mark.parametrize

def test_mimo_mlp_defaults_to_regular_mlp():
    from x_mlps_pytorch.mlp import MLP
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(256, 128, 64)
    ref = MLP(256, 128, 64)

    ref.load_state_dict(mlp.trunk.state_dict())

    x = torch.randn(7, 3, 256)

    out = mlp(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (7, 3, 64)
    assert torch.allclose(out, ref(x))

def test_mimo_mlp_single_input_and_output_module():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        32,
        8,
        inputs = nn.Linear(4, 16),
        outputs = nn.Linear(8, 2)
    )

    x = torch.randn(5, 4)

    out = mlp(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 2)

def test_mimo_mlp_single_module_in_sequence_stays_single():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        inputs = (nn.Linear(4, 16),),
        outputs = (nn.Linear(8, 2),)
    )

    out = mlp(torch.randn(3, 4))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)

def test_mimo_mlp_no_trunk():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        8,
        inputs = (
            nn.Linear(3, 4),
            nn.Linear(5, 4)
        ),
        outputs = (
            nn.Linear(8, 1),
            nn.Linear(8, 2)
        )
    )

    out1, out2 = mlp((torch.randn(4, 3), torch.randn(4, 5)))

    assert out1.shape == (4, 1)
    assert out2.shape == (4, 2)

def test_mimo_mlp_no_dims():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        inputs = (nn.Identity(), nn.Identity()),
        outputs = nn.Linear(5, 2)
    )

    out = mlp((torch.randn(4, 3), torch.randn(4, 2)))

    assert out.shape == (4, 2)

def test_mimo_mlp_multiple_inputs_and_outputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        64,
        64,
        inputs = (
            nn.Linear(3, 32),
            nn.Linear(5, 32)
        ),
        outputs = (
            nn.Linear(64, 1),
            nn.Linear(64, 4)
        )
    )

    x1 = torch.randn(10, 3)
    x2 = torch.randn(10, 5)

    value, action = mlp((x1, x2))

    assert value.shape == (10, 1)
    assert action.shape == (10, 4)

@param('container', (list, tuple))
def test_mimo_mlp_multiple_inputs_no_outputs(container):
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        64,
        32,
        inputs = (
            nn.Linear(3, 32),
            nn.Linear(5, 32)
        )
    )

    x1 = torch.randn(10, 3)
    x2 = torch.randn(10, 5)

    out = mlp(container([x1, x2]))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (10, 32)

def test_mimo_mlp_single_input_multiple_outputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        16,
        outputs = (
            nn.Linear(16, 1),
            nn.Linear(16, 4)
        )
    )

    value, action = mlp(torch.randn(3, 16))

    assert value.shape == (3, 1)
    assert action.shape == (3, 4)

@param('container', (list, tuple, nn.ModuleList))
def test_mimo_mlp_accepts_module_containers(container):
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        32,
        16,
        inputs = container([
            nn.Linear(3, 16),
            nn.Linear(5, 16)
        ]),
        outputs = container([
            nn.Linear(16, 1),
            nn.Linear(16, 2)
        ])
    )

    out1, out2 = mlp((torch.randn(4, 3), torch.randn(4, 5)))

    assert out1.shape == (4, 1)
    assert out2.shape == (4, 2)

def test_mimo_mlp_squeeze_out():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        16,
        outputs = (
            nn.Linear(16, 1),
            nn.Linear(16, 4)
        ),
        squeeze_out = True
    )

    value, action = mlp(torch.randn(3, 16))

    assert value.shape == (3,)
    assert action.shape == (3, 4)

def test_mimo_mlp_squeeze_out_single():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        outputs = nn.Linear(8, 1),
        squeeze_out = True
    )

    out = mlp(torch.randn(3, 16))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3,)

def test_mimo_mlp_trunk_only_accepts_list_input():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(8, 4)

    out = mlp([torch.randn(2, 3), torch.randn(2, 5)])

    assert out.shape == (2, 4)

def test_mimo_mlp_custom_combine_inputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    def combine(features):
        pooled = features[0].mean(dim = -1, keepdim = True)
        return torch.cat((pooled, features[1]), dim = -1)

    mlp = MimoMLP(
        9,
        16,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(5, 8)
        ),
        outputs = nn.Linear(16, 2),
        combine_inputs = combine
    )

    out = mlp((torch.randn(6, 3), torch.randn(6, 5)))

    assert out.shape == (6, 2)

def test_mimo_mlp_select_outputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        32,
        16,
        outputs = (
            nn.Linear(16, 1),
            nn.Linear(16, 2),
            nn.Linear(16, 3)
        )
    )

    out0, out2 = mlp(torch.randn(5, 32), select = (0, 2))

    assert out0.shape == (5, 1)
    assert out2.shape == (5, 3)

def test_mimo_mlp_select_single_output():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        outputs = (
            nn.Linear(8, 1),
            nn.Linear(8, 2)
        )
    )

    out = mlp(torch.randn(3, 16), select = 1)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 2)

def test_mimo_mlp_select_skips_unused_heads():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    class Fail(nn.Module):
        def forward(self, _):
            raise AssertionError('unused head should not be evaluated')

    mlp = MimoMLP(
        16,
        8,
        outputs = (
            nn.Linear(8, 1),
            Fail(),
            nn.Linear(8, 2)
        )
    )

    out0, out2 = mlp(torch.randn(3, 16), select = (0, 2))

    assert out0.shape == (3, 1)
    assert out2.shape == (3, 2)

def test_mimo_mlp_select_negative_index():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        outputs = (
            nn.Linear(8, 1),
            nn.Linear(8, 2)
        )
    )

    out = mlp(torch.randn(3, 16), select = -1)

    assert out.shape == (3, 2)

def test_mimo_mlp_select_invalid_index():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        outputs = (
            nn.Linear(8, 1),
            nn.Linear(8, 2)
        )
    )

    with pytest.raises(IndexError):
        mlp(torch.randn(3, 16), select = 2)

def test_mimo_mlp_return_intermediates():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        64,
        32,
        inputs = (
            nn.Linear(3, 32),
            nn.Linear(5, 32)
        ),
        outputs = (
            nn.Linear(32, 1),
            nn.Linear(32, 4)
        )
    )

    intermediates = mlp(
        (torch.randn(7, 3), torch.randn(7, 5)),
        return_intermediates = True
    )

    enc1, enc2 = intermediates.inputs

    assert enc1.shape == (7, 32)
    assert enc2.shape == (7, 32)
    assert intermediates.trunk_input.shape == (7, 64)
    assert intermediates.trunk_output.shape == (7, 32)
    assert len(intermediates.outputs) == 2
    assert intermediates.outputs[0].shape == (7, 1)
    assert intermediates.outputs[1].shape == (7, 4)

def test_mimo_mlp_return_intermediates_ignores_select():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        8,
        outputs = (
            nn.Linear(8, 1),
            nn.Linear(8, 2)
        )
    )

    intermediates = mlp(
        torch.randn(3, 16),
        select = (0,),
        return_intermediates = True
    )

    assert len(intermediates.outputs) == 2

def test_mimo_mlp_select_inputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        13,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8),
            nn.Identity()
        )
    )

    x0 = torch.randn(5, 3)
    x1 = torch.randn(5, 4)
    x2 = torch.randn(5, 5)

    out = mlp((x0, x1, x2), select_inputs = (0, 2))
    assert out.shape == (5, 2)

    out = mlp((x0, x1, x2), select_inputs = (1, 2))
    assert out.shape == (5, 2)

def test_mimo_mlp_select_single_input():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        8,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8)
        )
    )

    out = mlp((torch.randn(5, 3), torch.randn(5, 4)), select_inputs = 1)

    assert out.shape == (5, 2)

def test_mimo_mlp_select_inputs_skips_unused():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    class Fail(nn.Module):
        def forward(self, _):
            raise AssertionError('unused input should not be evaluated')

    mlp = MimoMLP(
        13,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            Fail(),
            nn.Identity()
        )
    )

    out = mlp(
        (torch.randn(5, 3), torch.randn(5, 4), torch.randn(5, 5)),
        select_inputs = (0, 2)
    )

    assert out.shape == (5, 2)

def test_mimo_mlp_select_inputs_raw():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(8, 16, 2)

    out = mlp(
        (torch.randn(5, 3), torch.randn(5, 5)),
        select_inputs = (0, 1)
    )

    assert out.shape == (5, 2)

def test_mimo_mlp_select_inputs_and_outputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        13,
        16,
        outputs = (
            nn.Linear(16, 1),
            nn.Linear(16, 2)
        ),
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8),
            nn.Identity()
        )
    )

    out = mlp(
        (torch.randn(5, 3), torch.randn(5, 4), torch.randn(5, 5)),
        select_inputs = (0, 2),
        select = 1
    )

    assert out.shape == (5, 2)

def test_mimo_mlp_return_intermediates_with_select_inputs():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        13,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8),
            nn.Identity()
        )
    )

    intermediates = mlp(
        (torch.randn(5, 3), torch.randn(5, 4), torch.randn(5, 5)),
        select_inputs = (0, 2),
        return_intermediates = True
    )

    enc0, enc2 = intermediates.inputs

    assert enc0.shape == (5, 8)
    assert enc2.shape == (5, 5)
    assert intermediates.trunk_input.shape == (5, 13)
    assert intermediates.trunk_output.shape == (5, 2)

def test_mimo_mlp_select_inputs_negative():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        8,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8)
        )
    )

    out = mlp((torch.randn(5, 3), torch.randn(5, 4)), select_inputs = -1)

    assert out.shape == (5, 2)

def test_mimo_mlp_select_inputs_invalid():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        8,
        16,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8)
        )
    )

    with pytest.raises(IndexError):
        mlp((torch.randn(5, 3), torch.randn(5, 4)), select_inputs = 2)

def test_mimo_mlp_trunk_input_stays_fixed():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    mlp = MimoMLP(
        16,
        32,
        2,
        inputs = (
            nn.Linear(3, 8),
            nn.Linear(4, 8)
        )
    )

    x = (torch.randn(5, 3), torch.randn(5, 4))

    out = mlp(x, select_inputs = (0, 1))
    assert out.shape == (5, 2)

    with pytest.raises(AssertionError):
        mlp(x, select_inputs = 0)

def test_mimo_mlp_gradients_reach_all_modules():
    from x_mlps_pytorch.mimo_mlp import MimoMLP

    in1 = nn.Linear(3, 16)
    in2 = nn.Linear(5, 16)
    out1 = nn.Linear(16, 1)
    out2 = nn.Linear(16, 2)

    mlp = MimoMLP(
        32,
        16,
        inputs = (in1, in2),
        outputs = (out1, out2)
    )

    o1, o2 = mlp((torch.randn(4, 3), torch.randn(4, 5)))

    (o1.sum() + o2.sum()).backward()

    for module in (in1, in2, out1, out2):
        assert all(p.grad is not None for p in module.parameters())
