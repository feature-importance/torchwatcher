import torch
import torch.nn.functional as F
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
        self.names = []

    def process(self, name, module, inputs, outputs):
        self.names.append(name)
        self.modules.append(module)


def test_wrapped_interjection_receives_underlying_module_for_module_call():
    interjection = _ModuleRecorder()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )

    watched_model(torch.ones(1, 2))
    watched_model.eval()
    watched_model(torch.ones(1, 2))

    assert len(interjection.modules) == 2
    assert isinstance(interjection.modules[0], nn.Linear)
    assert interjection.modules[1] is interjection.modules[0]
    assert not isinstance(interjection.modules[0], torch.fx.GraphModule)
    assert len(list(watched_model.parameters())) == 2
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


class _FunctionalDropout(nn.Module):
    def forward(self, inputs):
        return F.dropout(inputs, p=1.0, training=self.training)


def test_wrapped_interjection_preserves_train_mode_operation():
    interjection = _ModuleRecorder()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is F.dropout
        )
    )
    watched_model = interject_by_match(
        _FunctionalDropout(),
        selector,
        interjection,
    )
    inputs = torch.ones(4)

    watched_model.train()
    train_result = watched_model(inputs)
    watched_model.eval()
    eval_result = watched_model(inputs)

    torch.testing.assert_close(train_result, torch.zeros_like(inputs))
    torch.testing.assert_close(eval_result, inputs)
    assert interjection.names == ["dropout", "dropout"]
