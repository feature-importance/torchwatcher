import torch
from torch import nn

from torchwatcher.interjection import (
    WrappedForwardInterjection,
    interject_by_match,
    interject_by_module_class,
)
from torchwatcher.interjection.node_selector import NodeSelector


class _ModuleRecorder(WrappedForwardInterjection):
    def __init__(self):
        super().__init__()
        self.modules = []

    def process(self, name, module, inputs, outputs):
        self.modules.append(module)


def test_wrapped_interjection_receives_underlying_module_for_module_call():
    interjection = _ModuleRecorder()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )

    watched_model(torch.ones(1, 2))

    assert len(interjection.modules) == 1
    assert isinstance(interjection.modules[0], nn.Linear)
    assert not isinstance(interjection.modules[0], torch.fx.GraphModule)
    assert any(
        interjection.modules[0].weight is parameter
        for parameter in watched_model.parameters()
    )


class _FunctionalReLU(nn.Module):
    def forward(self, inputs):
        return torch.relu(inputs)


def test_wrapped_interjection_receives_none_for_function_call():
    interjection = _ModuleRecorder()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is torch.relu
        )
    )
    watched_model = interject_by_match(
        _FunctionalReLU(),
        selector,
        interjection,
    )

    watched_model(torch.tensor([[-1.0, 2.0]]))

    assert interjection.modules == [None]


class _AddInputs(nn.Module):
    def forward(self, left, right):
        return torch.add(input=left, other=right, alpha=2)


def test_wrapped_interjection_supports_multi_input_node():
    interjection = _ModuleRecorder()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is torch.add
        )
    )
    watched_model = interject_by_match(
        _AddInputs(),
        selector,
        interjection,
    )
    left = torch.tensor([[1.0, 2.0]], requires_grad=True)
    right = torch.tensor([[3.0, 4.0]], requires_grad=True)

    result = watched_model(left, right)
    result.sum().backward()

    torch.testing.assert_close(result, left + 2 * right)
    assert interjection.modules == [None]
    torch.testing.assert_close(left.grad, torch.ones_like(left))
    torch.testing.assert_close(right.grad, torch.full_like(right, 2))


class _AddModule(nn.Module):
    def forward(self, left, right):
        return left + right


class _UsesAddModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = _AddModule()

    def forward(self, left, right):
        return self.add(left, right)


def test_wrapped_interjection_supports_multi_input_module():
    interjection = _ModuleRecorder()
    watched_model = interject_by_module_class(
        _UsesAddModule(),
        _AddModule,
        interjection,
    )
    left = torch.tensor([[1.0, 2.0]])
    right = torch.tensor([[3.0, 4.0]])

    result = watched_model(left, right)

    torch.testing.assert_close(result, left + right)
    assert len(interjection.modules) == 1
    assert isinstance(interjection.modules[0], _AddModule)


class _AddRepeatedInput(nn.Module):
    def forward(self, inputs):
        return torch.add(inputs, inputs)


def test_wrapped_interjection_supports_repeated_input():
    interjection = _ModuleRecorder()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is torch.add
        )
    )
    watched_model = interject_by_match(
        _AddRepeatedInput(),
        selector,
        interjection,
    )
    inputs = torch.tensor([[1.0, 2.0]])

    result = watched_model(inputs)

    torch.testing.assert_close(result, inputs * 2)
    assert interjection.modules == [None]
