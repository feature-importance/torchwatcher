import pickle

import torch
from torch import nn

from torchwatcher.analysis import NameAnalyser, analyser_observations
from torchwatcher.interjection import (
    ForwardInterjection,
    WrappedForwardBackwardInterjection,
    WrappedForwardInterjection,
    interject_by_module_class_native,
)
from torchwatcher.parameters.selectors import (
    matches_full_name,
    matches_module_class,
    matches_module_name,
    select_parameters,
)


class _RecordingReplacement(ForwardInterjection):
    def __init__(self):
        super().__init__()
        self.calls = []

    def process(self, name, module, inputs):
        self.calls.append((name, module, inputs.detach().clone()))
        return inputs + 5


class _RecordingWrappedInterjection(WrappedForwardInterjection):
    def process(self, name, module, inputs, outputs):
        pass


class _RecordingBackwardInterjection(WrappedForwardBackwardInterjection):
    def __init__(self):
        super().__init__()
        self.names = []

    def process_backward(self, name, module, grad_input, grad_output):
        self.names.append(name)


class _SharedReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.ReLU()
        self.left = shared
        self.right = shared

    def forward(self, inputs):
        return self.right(self.left(inputs))


def test_native_forward_interjection_observes_and_replaces_module_output():
    model = nn.Sequential(nn.ReLU())
    interjection = _RecordingReplacement()
    watched_model = interject_by_module_class_native(
        model,
        nn.ReLU,
        interjection,
    )
    inputs = torch.tensor([[-2.0, 3.0]])

    outputs = watched_model(inputs)

    assert len(interjection.calls) == 1
    name, module, observed = interjection.calls[0]
    assert name == "0"
    assert isinstance(module, nn.ReLU)
    torch.testing.assert_close(observed, torch.tensor([[0.0, 3.0]]))
    torch.testing.assert_close(outputs, torch.tensor([[5.0, 8.0]]))
    torch.testing.assert_close(model(inputs), torch.tensor([[0.0, 3.0]]))


def test_native_interjection_can_wrap_the_root_module():
    interjection = _RecordingReplacement()
    watched_model = interject_by_module_class_native(
        nn.ReLU(),
        nn.ReLU,
        interjection,
    )

    outputs = watched_model(torch.tensor([-2.0, 3.0]))

    assert len(interjection.calls) == 1
    assert interjection.calls[0][0] == "root"
    torch.testing.assert_close(outputs, torch.tensor([5.0, 8.0]))


def test_native_root_analyser_produces_valid_observations():
    analyser = NameAnalyser()
    watched_model = interject_by_module_class_native(
        nn.ReLU(),
        nn.ReLU,
        analyser,
    )

    watched_model(torch.tensor([-2.0, 3.0]))
    observations = list(analyser_observations(analyser))

    assert analyser.to_dict() == {"root": "root"}
    assert [observation.path for observation in observations] == [
        ("analysers", "NameAnalyser", "root"),
    ]


class _NestedTarget(nn.Module):
    def __init__(self, child=None):
        super().__init__()
        if child is not None:
            self.root = child

    def forward(self, inputs):
        if hasattr(self, "root"):
            return self.root(inputs)
        return inputs


def test_native_root_name_does_not_collide_with_child_name():
    interjection = _RecordingReplacement()
    watched_model = interject_by_module_class_native(
        _NestedTarget(_NestedTarget()),
        _NestedTarget,
        interjection,
    )

    outputs = watched_model(torch.tensor([1.0]))

    assert [call[0] for call in interjection.calls] == ["root", "root@1"]
    torch.testing.assert_close(outputs, torch.tensor([11.0]))


def test_native_interjection_observes_every_call_to_a_shared_module():
    interjection = _RecordingReplacement()
    watched_model = interject_by_module_class_native(
        _SharedReLUModel(),
        nn.ReLU,
        interjection,
    )

    outputs = watched_model(torch.tensor([1.0]))

    assert [call[0] for call in interjection.calls] == ["left", "right"]
    assert interjection.calls[0][1] is interjection.calls[1][1]
    torch.testing.assert_close(outputs, torch.tensor([11.0]))


def test_native_shared_module_backward_hooks_remain_alias_specific():
    interjection = _RecordingBackwardInterjection()
    watched_model = interject_by_module_class_native(
        _SharedReLUModel(),
        nn.ReLU,
        interjection,
    )
    inputs = torch.tensor([1.0], requires_grad=True)

    watched_model(inputs).sum().backward()

    assert interjection.names == ["right", "left"]


def test_native_backward_interjection_handles_keyword_input_gradients():
    class MixedAdd(nn.Module):
        def forward(self, left, *, right):
            return left + right

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.add = MixedAdd()

        def forward(self, left, right):
            return self.add(left, right=right)

    class ScaleInputGradients(WrappedForwardBackwardInterjection):
        def __init__(self):
            super().__init__()
            self.grad_input = None

        def process_backward(
                self, name, module, grad_input, grad_output):
            self.grad_input = grad_input
            if not isinstance(grad_input, tuple) or len(grad_input) != 2:
                return None
            return grad_input[0] * 2, grad_input[1] * 3

    interjection = ScaleInputGradients()
    watched_model = interject_by_module_class_native(
        Model(),
        MixedAdd,
        interjection,
    )
    left = torch.tensor([1.0], requires_grad=True)
    right = torch.tensor([2.0], requires_grad=True)

    watched_model(left, right).sum().backward()

    assert isinstance(interjection.grad_input, tuple)
    assert len(interjection.grad_input) == 2
    torch.testing.assert_close(
        interjection.grad_input[0],
        torch.ones_like(left),
    )
    torch.testing.assert_close(
        interjection.grad_input[1],
        torch.ones_like(right),
    )
    torch.testing.assert_close(left.grad, torch.full_like(left, 2.0))
    torch.testing.assert_close(right.grad, torch.full_like(right, 3.0))


def test_native_interjection_mode_does_not_change_eval_model_operation():
    interjection = _RecordingWrappedInterjection()
    watched_model = interject_by_module_class_native(
        nn.Sequential(nn.Dropout(p=1.0)),
        nn.Dropout,
        interjection,
    )
    watched_model.eval()

    interjection.train()

    torch.testing.assert_close(
        watched_model(torch.ones(4)),
        torch.ones(4),
    )


def test_native_interjection_mode_does_not_change_train_model_operation():
    interjection = _RecordingWrappedInterjection()
    watched_model = interject_by_module_class_native(
        nn.Sequential(nn.Dropout(p=1.0)),
        nn.Dropout,
        interjection,
    )
    watched_model.train()

    interjection.eval()

    torch.testing.assert_close(
        watched_model(torch.ones(4)),
        torch.zeros(4),
    )


def test_native_interjection_preserves_original_parameter_names_and_owners():
    model = nn.Sequential(nn.Linear(2, 1))
    original_parameter_names = [name for name, _ in model.named_parameters()]
    original_state_names = list(model.state_dict())
    interjection = _RecordingWrappedInterjection()

    watched_model = interject_by_module_class_native(
        model,
        nn.Linear,
        interjection,
    )

    assert [name for name, _ in watched_model.named_parameters()] == (
        original_parameter_names
    )
    assert list(watched_model.state_dict()) == original_state_names
    assert isinstance(dict(watched_model.named_modules())["0"], nn.Linear)
    assert [
        info.full_name
        for info in select_parameters(
            watched_model,
            matches_full_name("0.weight"),
        )
    ] == ["0.weight"]
    assert [
        info.full_name
        for info in select_parameters(
            watched_model,
            matches_module_name("0"),
        )
    ] == ["0.weight", "0.bias"]
    selected_by_class = select_parameters(
        watched_model,
        matches_module_class(nn.Linear),
    )
    assert [info.full_name for info in selected_by_class] == [
        "0.weight",
        "0.bias",
    ]
    assert all(isinstance(info.module, nn.Linear) for info in selected_by_class)


def test_native_interjection_preserves_checkpoint_and_dtype_behaviour():
    model = nn.Sequential(nn.Linear(2, 1))
    watched_model = interject_by_module_class_native(
        model,
        nn.Linear,
        _RecordingWrappedInterjection(),
    )
    replacement_state = {
        "0.weight": torch.full((1, 2), 3.0),
        "0.bias": torch.full((1,), 4.0),
    }

    watched_model.load_state_dict(replacement_state, strict=True)
    watched_model.double()
    output = watched_model(torch.ones(1, 2, dtype=torch.float64))

    torch.testing.assert_close(
        output,
        torch.full((1, 1), 10.0, dtype=torch.float64),
    )
    assert all(
        parameter.dtype == torch.float64
        for parameter in watched_model.parameters()
    )
    restored = nn.Sequential(nn.Linear(2, 1)).double()
    restored.load_state_dict(watched_model.state_dict(), strict=True)
    torch.testing.assert_close(
        restored(torch.ones(1, 2, dtype=torch.float64)),
        output,
    )


def test_native_interjection_delegates_apply_to_original_module():
    watched_model = interject_by_module_class_native(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        _RecordingWrappedInterjection(),
    )
    visited = []

    def initialise(module):
        if isinstance(module, nn.Linear):
            visited.append(module)
            nn.init.constant_(module.weight, 3.0)

    watched_model.apply(initialise)

    assert len(visited) == 1
    torch.testing.assert_close(
        watched_model(torch.ones(1, 2)),
        visited[0].bias.unsqueeze(0) + 6.0,
    )


def test_native_interjected_model_can_be_pickled():
    watched_model = interject_by_module_class_native(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        _RecordingWrappedInterjection(),
    )
    inputs = torch.ones(1, 2)
    expected = watched_model(inputs)

    restored = pickle.loads(pickle.dumps(watched_model))

    torch.testing.assert_close(restored(inputs), expected)


def test_native_forward_interjection_adapter_can_be_pickled():
    watched_model = interject_by_module_class_native(
        nn.Sequential(nn.ReLU()),
        nn.ReLU,
        _RecordingReplacement(),
    )

    restored = pickle.loads(pickle.dumps(watched_model))

    torch.testing.assert_close(
        restored(torch.tensor([-2.0, 3.0])),
        torch.tensor([5.0, 8.0]),
    )
