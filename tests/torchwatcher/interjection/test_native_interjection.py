import torch
from torch import nn

from torchwatcher.interjection import (
    ForwardInterjection,
    WrappedForwardBackwardInterjection,
    WrappedForwardInterjection,
    interject_by_module_class_native,
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
    assert interjection.calls[0][0] == ""
    torch.testing.assert_close(outputs, torch.tensor([5.0, 8.0]))


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
