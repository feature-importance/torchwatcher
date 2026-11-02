import torch
import torch.nn.functional as F
import pytest
from torch import nn

from torchwatcher.analysis import NameAnalyser
from torchwatcher.drawing import draw_graph, draw_graph_pretty
from torchwatcher.interjection import (
    WrappedForwardInterjection,
    interject_by_match,
    interject_by_module_class,
    interject_by_name,
    trace,
    trace_shapes,
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


def test_interjection_does_not_mutate_source_model():
    model = nn.Sequential(nn.Linear(2, 1))
    modules_before = dict(model.named_modules())
    state_keys_before = list(model.state_dict())
    analyser = NameAnalyser()

    watched_model = interject_by_name(model, "0", analyser)

    assert dict(model.named_modules()) == modules_before
    assert list(model.state_dict()) == state_keys_before
    watched_model(torch.ones(1, 2))
    assert analyser.to_dict() == {"0": "0"}


class _InplaceScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.inplace = True

    def forward(self, inputs):
        if self.inplace:
            return inputs.mul_(self.weight)
        return inputs * self.weight


def test_interjection_does_not_mutate_source_inplace_module():
    source_module = _InplaceScale()
    model = nn.Sequential(source_module)
    interjection = _ModuleRecorder()

    watched_model = interject_by_module_class(
        model,
        _InplaceScale,
        interjection,
    )

    assert source_module.inplace
    for training in (True, False):
        wrapped_module = interjection.wrapped_module(
            "0",
            training=training,
        ).get_submodule("0")
        assert wrapped_module is not source_module
        assert wrapped_module.weight is source_module.weight
        assert not wrapped_module.inplace

    result = watched_model(torch.ones(2))

    torch.testing.assert_close(result, torch.full((2,), 2.0))
    assert interjection.modules == [source_module]
    assert source_module.inplace


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


def test_pretty_drawing_supports_multi_input_wrapped_node():
    analyser = NameAnalyser()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is torch.add
        )
    )
    watched_model = interject_by_match(
        _AddInputs(),
        selector,
        analyser,
    )

    graph = draw_graph_pretty(
        watched_model,
        torch.ones(1, 2),
        torch.ones(1, 2),
    )

    graph_source = graph.to_string().lower()
    assert "add" in graph_source
    assert "2 x (1, 2)" in graph_source


def test_pretty_drawing_supports_shape_changing_wrapped_node():
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        NameAnalyser(),
    )

    graph = draw_graph_pretty(watched_model, torch.ones(1, 2))

    graph_source = graph.to_string().lower()
    assert "linear" in graph_source
    assert "input:</td>" in graph_source
    assert "(1, 2)" in graph_source
    assert "output: </td>" in graph_source
    assert "(1, 1)" in graph_source


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


class _SingletonTuple(nn.Module):
    def forward(self, inputs):
        return (inputs,)


class _UsesSingletonTuple(nn.Module):
    def __init__(self):
        super().__init__()
        self.tuple = _SingletonTuple()

    def forward(self, inputs):
        return self.tuple(inputs)[0] + 1


def test_wrapped_interjection_preserves_singleton_tuple_output():
    interjection = _ModuleRecorder()
    watched_model = interject_by_module_class(
        _UsesSingletonTuple(),
        _SingletonTuple,
        interjection,
    )
    inputs = torch.arange(4.0).reshape(2, 2)

    result = watched_model(inputs)

    torch.testing.assert_close(result, inputs + 1)
    assert len(interjection.modules) == 1
    assert isinstance(interjection.modules[0], _SingletonTuple)


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


@pytest.mark.parametrize("model_training", [True, False])
@pytest.mark.parametrize("analyser_training", [True, False])
def test_analyser_mode_does_not_select_wrapped_operation(
        model_training,
        analyser_training):
    analyser = NameAnalyser()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is F.dropout
        )
    )
    watched_model = interject_by_match(
        _FunctionalDropout(),
        selector,
        analyser,
    )
    inputs = torch.ones(4)

    watched_model.train(model_training)
    analyser.train(analyser_training)
    result = watched_model(inputs)

    expected = torch.zeros_like(inputs) if model_training else inputs
    torch.testing.assert_close(result, expected)
    assert analyser.training is analyser_training


@pytest.mark.parametrize("model_training", [True, False])
@pytest.mark.parametrize("analyser_training", [True, False])
def test_analyser_mode_does_not_change_wrapped_module_operation(
        model_training,
        analyser_training):
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Dropout(p=1.0)),
        nn.Dropout,
        analyser,
    )
    inputs = torch.ones(4)

    watched_model.train(model_training)
    analyser.train(analyser_training)
    result = watched_model(inputs)

    expected = torch.zeros_like(inputs) if model_training else inputs
    torch.testing.assert_close(result, expected)
    assert analyser.training is analyser_training


def test_drawing_resolves_mode_aware_analyser_interjection():
    watched_model = interject_by_name(
        nn.Sequential(nn.ReLU()),
        "0",
        NameAnalyser(),
    )

    for training in (True, False):
        watched_model.train(training)
        graph = draw_graph(watched_model)
        pretty_graph = draw_graph_pretty(watched_model, torch.ones(1, 2))

        assert "relu" in graph.to_string().lower()
        assert "relu" in pretty_graph.to_string().lower()


def test_drawing_resolves_mode_aware_direct_wrapped_interjection():
    watched_model = interject_by_module_class(
        nn.Sequential(nn.ReLU()),
        nn.ReLU,
        _ModuleRecorder(),
    )

    for training in (True, False):
        watched_model.train(training)
        assert "relu" in draw_graph(watched_model).to_string().lower()


class _ModeRecorder(nn.Module):
    observations = []

    def forward(self, inputs):
        self.observations.append(self.training)
        return inputs


class _RaisesInEval(nn.Module):
    def forward(self, inputs):
        if not self.training:
            raise RuntimeError("eval tracing failed")
        return inputs


def test_trace_restores_source_mode_after_failure():
    model = _RaisesInEval()
    model.train()

    with pytest.raises(RuntimeError, match="eval tracing failed"):
        trace(model)

    assert model.training


def test_interjection_does_not_mutate_tracer_kwargs():
    leaf_modules = [_ModeRecorder]
    tracer_kwargs = {"leaf_modules": leaf_modules}

    interject_by_module_class(
        nn.Sequential(nn.ReLU()),
        nn.ReLU,
        NameAnalyser(),
        tracer_kwargs=tracer_kwargs,
    )

    assert tracer_kwargs == {"leaf_modules": [_ModeRecorder]}
    assert tracer_kwargs["leaf_modules"] is leaf_modules


def test_trace_shapes_uses_each_graphs_corresponding_mode():
    _ModeRecorder.observations = []
    watched_model = trace(
        nn.Sequential(_ModeRecorder()),
        tracer_kwargs={"leaf_modules": [_ModeRecorder]},
    )
    watched_model.train()

    trace_shapes(watched_model, torch.ones(1, 2))

    assert _ModeRecorder.observations == [True, False]
    assert watched_model.training
    assert watched_model.train_module.training
    assert watched_model.eval_module.training


def test_trace_shapes_restores_functorch_config_after_failure():
    import torch._functorch.config as functorch_config

    previous_device_type = functorch_config.fake_tensor_prefer_device_type
    functorch_config.fake_tensor_prefer_device_type = "cpu"
    watched_model = trace(nn.Sequential(nn.Linear(2, 1)))

    try:
        with pytest.raises(RuntimeError):
            trace_shapes(watched_model, torch.ones(1, 3))

        assert functorch_config.fake_tensor_prefer_device_type == "cpu"
    finally:
        functorch_config.fake_tensor_prefer_device_type = previous_device_type
