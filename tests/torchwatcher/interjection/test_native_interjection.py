import torch
from torch import nn

from torchwatcher.interjection import (
    ForwardInterjection,
    interject_by_module_class_native,
)


class _RecordingReplacement(ForwardInterjection):
    def __init__(self):
        super().__init__()
        self.calls = []

    def process(self, name, module, inputs):
        self.calls.append((name, module, inputs.detach().clone()))
        return inputs + 5


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
