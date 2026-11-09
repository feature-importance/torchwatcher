import pickle

import torch
import torch.nn.functional as F
import pytest
from torch import nn

from torchwatcher.analysis import NameAnalyser
from torchwatcher.drawing import draw_graph, draw_graph_pretty
from torchwatcher.interjection import (
    ForwardInterjection,
    WrappedForwardBackwardInterjection,
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
        self.outputs = []

    def process(self, name, module, inputs, outputs):
        self.names.append(name)
        self.modules.append(module)
        self.outputs.append(outputs)


class _StatefulInterjection(WrappedForwardInterjection):
    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.tensor(3.0))

    def process(self, name, module, inputs, outputs):
        pass


class _ExtraStateLinear(nn.Linear):
    def __init__(self, in_features, out_features, value=0):
        super().__init__(in_features, out_features)
        self.value = value

    def get_extra_state(self):
        return self.value

    def set_extra_state(self, state):
        self.value = state


class _VersionedLinear(nn.Linear):
    _version = 7

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.loaded_versions = []

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        self.loaded_versions.append(local_metadata.get("version"))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _FailingVersionedLinear(_VersionedLinear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.fail_loading = True

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        if self.fail_loading:
            raise RuntimeError("deliberate load failure")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _ExtraStateBlock(nn.Module):
    def __init__(self, value=0):
        super().__init__()
        self.linear = _ExtraStateLinear(2, 1, value=value)

    def forward(self, inputs):
        return self.linear(inputs)


class _PartiallyUnusedStateModel(nn.Module):
    def __init__(self, extra_state=0):
        super().__init__()
        self.root_parameter = nn.Parameter(torch.ones(1))
        self.register_buffer("root_buffer", torch.ones(1))
        self.used = nn.Linear(2, 1)
        self.unused = nn.Linear(2, 1)
        self.extra_state = extra_state

    def get_extra_state(self):
        return self.extra_state

    def set_extra_state(self, state):
        self.extra_state = state

    def forward(self, inputs):
        return self.used(inputs)


class _ExtraStateInterjection(_StatefulInterjection):
    def __init__(self):
        super().__init__()
        self.value = 5

    def get_extra_state(self):
        return self.value

    def set_extra_state(self, state):
        self.value = state


class _ScaleBackward(WrappedForwardBackwardInterjection):
    def process_backward(self, name, module, grad_input, grad_output):
        return grad_input * 2


class _RecordBackward(WrappedForwardBackwardInterjection):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def process_backward(self, name, module, grad_input, grad_output):
        self.calls += 1


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


def test_qualified_module_names_treat_periods_literally():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.ReLU()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a_b = nn.ReLU()
            self.a = Nested()

        def forward(self, inputs):
            return self.a.b(self.a.b(self.a_b(inputs)))

    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        Model(),
        nn.ReLU,
        analyser,
    )
    inputs = torch.tensor([-1.0, 2.0])

    outputs = watched_model(inputs)

    torch.testing.assert_close(outputs, torch.tensor([0.0, 2.0]))
    assert analyser.to_dict() == {
        "a_b": "a_b",
        "a.b": "a.b",
        "a.b_1": "a.b_1",
    }


def test_colliding_qualified_names_keep_distinct_registered_operations():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.Linear(2, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a_b = nn.Linear(2, 2)
            self.a = Nested()

        def forward(self, inputs):
            return self.a.b(self.a_b(inputs))

    analyser = NameAnalyser()
    interject_by_module_class(Model(), nn.Linear, analyser)
    registered = analyser.interjection._modules

    operations = [
        analyser.interjection.wrapped_module(name, training=training)
        for name in ("a_b", "a.b")
        for training in (True, False)
    ]
    assert len(registered) == 4
    assert all(operation in registered.values() for operation in operations)

    operation_parameters = {
        id(parameter)
        for operation in operations
        for parameter in operation.parameters()
    }
    assert {id(parameter) for parameter in analyser.parameters()} == (
        operation_parameters
    )


def test_numeric_suffix_in_module_name_is_not_treated_as_fx_suffix():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Linear(1, 1, bias=False)
            self.block_1 = nn.Linear(1, 1, bias=False)
            self.block.weight.data.fill_(2)
            self.block_1.weight.data.fill_(3)

        def forward(self, inputs):
            # Calling block_1 first exposed the old name normalisation: it was
            # labelled block, and the real block was then labelled block_1.
            return self.block(self.block_1(inputs))

    interjection = _ModuleRecorder()
    watched_model = interject_by_name(Model(), "block_1", interjection)

    watched_model(torch.ones(1, 1))

    assert interjection.names == ["block_1"]
    torch.testing.assert_close(
        interjection.modules[0].weight,
        torch.full((1, 1), 3.0),
    )


def test_wrapped_backward_interjection_accepts_single_tensor_gradient():
    interjection = _ScaleBackward()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Identity()),
        nn.Identity,
        interjection,
    )
    inputs = torch.ones(2, requires_grad=True)

    watched_model(inputs).sum().backward()

    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.0))


class _FunctionalInplaceReLU(nn.Module):
    def forward(self, inputs):
        return F.relu(inputs, inplace=True)


class _FunctionInplaceReLU(nn.Module):
    def forward(self, inputs):
        return torch.relu_(inputs)


class _MethodInplaceReLU(nn.Module):
    def forward(self, inputs):
        return inputs.relu_()


@pytest.mark.parametrize(
    ("model", "node_match"),
    [
        (
            _FunctionalInplaceReLU(),
            lambda node: (
                node.op == "call_function" and node.target is F.relu
            ),
        ),
        (
            _FunctionInplaceReLU(),
            lambda node: (
                node.op == "call_function" and node.target is torch.relu_
            ),
        ),
        (
            _MethodInplaceReLU(),
            lambda node: (
                node.op == "call_method" and node.target == "relu_"
            ),
        ),
    ],
)
def test_backward_interjection_makes_functional_inplace_node_out_of_place(
        model,
        node_match):
    interjection = _RecordBackward()
    watched_model = interject_by_match(
        model,
        NodeSelector(lambda state: node_match(state[1])),
        interjection,
    )
    source_inputs = torch.tensor([-1.0, 2.0])
    source_outputs = model(source_inputs)
    inputs = torch.tensor([-1.0, 2.0], requires_grad=True)

    outputs = watched_model(inputs)
    outputs.sum().backward()

    assert source_outputs.data_ptr() == source_inputs.data_ptr()
    torch.testing.assert_close(outputs, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(inputs.grad, torch.tensor([0.0, 1.0]))
    assert interjection.calls == 1


class _UnsupportedInplaceMethod(nn.Module):
    def forward(self, inputs):
        return inputs.zero_()


def test_backward_interjection_rejects_unsupported_inplace_method():
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_method" and state[1].target == "zero_"
        )
    )

    with pytest.raises(ValueError, match="no out-of-place counterpart"):
        interject_by_match(
            _UnsupportedInplaceMethod(),
            selector,
            _RecordBackward(),
        )


def test_forward_interjection_preserves_inplace_alias_semantics():
    class _AliasedInplaceReLU(nn.Module):
        def forward(self, inputs):
            aliased = inputs.clone()
            activated = torch.relu_(aliased)
            return aliased + activated

    model = _AliasedInplaceReLU()
    selector = NodeSelector(
        lambda state: (
            state[1].op == "call_function"
            and state[1].target is torch.relu_
        )
    )
    watched_model = interject_by_match(model, selector, NameAnalyser())
    inputs = torch.tensor([-1.0, 2.0])

    torch.testing.assert_close(watched_model(inputs), model(inputs))


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
        assert wrapped_module.inplace

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


class _SingletonTupleReplacement(ForwardInterjection):
    def process(self, name, module, inputs):
        return (inputs[0] + 1,)


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
    assert isinstance(interjection.outputs[0], tuple)
    assert len(interjection.outputs[0]) == 1
    torch.testing.assert_close(interjection.outputs[0][0], inputs)


def test_forward_interjection_preserves_singleton_tuple_replacement():
    interjection = _SingletonTupleReplacement()
    interjection.register("tuple", None)
    inputs = torch.arange(4.0).reshape(2, 2)

    result = interjection("tuple", (inputs,))

    assert isinstance(result, tuple)
    assert len(result) == 1
    torch.testing.assert_close(result[0], inputs + 1)


def test_inserted_forward_interjection_preserves_singleton_tuple_replacement():
    interjection = _SingletonTupleReplacement()
    watched_model = interject_by_module_class(
        _UsesSingletonTuple(),
        _SingletonTuple,
        interjection,
    )
    inputs = torch.arange(4.0).reshape(2, 2)

    result = watched_model(inputs)

    torch.testing.assert_close(result, inputs + 2)


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


class _CompositeDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=1.0)

    def forward(self, inputs):
        return self.dropout(inputs)


def test_wrapped_composite_leaf_retains_descendant_modes():
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        nn.Sequential(_CompositeDropout()),
        _CompositeDropout,
        analyser,
    )
    inputs = torch.ones(4)

    watched_model.train()
    analyser.eval()
    torch.testing.assert_close(
        watched_model(inputs),
        torch.zeros_like(inputs),
    )

    watched_model.eval()
    analyser.train()
    torch.testing.assert_close(watched_model(inputs), inputs)


class _CompositeBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, inputs):
        return self.batch_norm(inputs)


def test_wrapped_composite_leaf_shares_parameters_and_buffers():
    source = nn.Sequential(_CompositeBatchNorm())
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        source,
        _CompositeBatchNorm,
        analyser,
    )
    train_operation = analyser.interjection.wrapped_module(
        "0",
        training=True,
    ).get_submodule("0")
    eval_operation = analyser.interjection.wrapped_module(
        "0",
        training=False,
    ).get_submodule("0")

    assert train_operation is not eval_operation
    assert train_operation.batch_norm is not eval_operation.batch_norm
    assert (
        train_operation.batch_norm.weight
        is eval_operation.batch_norm.weight
        is source[0].batch_norm.weight
    )
    assert (
        train_operation.batch_norm.running_mean
        is eval_operation.batch_norm.running_mean
        is source[0].batch_norm.running_mean
    )

    watched_model.train()
    watched_model(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert source[0].batch_norm.num_batches_tracked.item() == 1
    torch.testing.assert_close(
        eval_operation.batch_norm.running_mean,
        source[0].batch_norm.running_mean,
    )


class _CompositeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, inputs):
        return self.linear(inputs)


def test_wrapped_composite_leaf_optimiser_update_is_shared_between_modes():
    source = nn.Sequential(_CompositeLinear())
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        source,
        _CompositeLinear,
        analyser,
    )
    optimiser = torch.optim.SGD(watched_model.parameters(), lr=0.1)
    inputs = torch.ones(1, 1)

    watched_model.train()
    watched_model(inputs).sum().backward()
    optimiser.step()
    watched_model.eval()

    torch.testing.assert_close(
        watched_model(inputs),
        torch.tensor([[0.9]]),
    )
    torch.testing.assert_close(source(inputs), torch.tensor([[0.9]]))


class _CompositeAlias(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.Linear(1, 1)
        self.left = shared
        self.right = shared

    def forward(self, inputs):
        return self.right(self.left(inputs))


def test_wrapped_composite_leaf_preserves_descendant_aliases():
    analyser = NameAnalyser()
    interject_by_module_class(
        nn.Sequential(_CompositeAlias()),
        _CompositeAlias,
        analyser,
    )
    train_operation = analyser.interjection.wrapped_module(
        "0",
        training=True,
    ).get_submodule("0")
    eval_operation = analyser.interjection.wrapped_module(
        "0",
        training=False,
    ).get_submodule("0")

    assert train_operation.left is train_operation.right
    assert eval_operation.left is eval_operation.right
    assert train_operation.left is not eval_operation.left
    assert train_operation.left.weight is eval_operation.left.weight


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


class _ModeStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_layer = nn.Linear(2, 2)
        self.eval_layer = nn.Linear(2, 2)
        self.register_buffer("train_scale", torch.ones(2))
        self.register_buffer("eval_scale", torch.ones(2))

    def forward(self, inputs):
        if self.training:
            return self.train_layer(inputs) * self.train_scale
        return self.eval_layer(inputs) * self.eval_scale


def test_trace_restores_mixed_source_modes():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.ReLU())
    model.train()
    model[0].eval()

    trace(model)

    assert model.training
    assert not model[0].training


def test_dual_graph_names_are_isolated_per_instance():
    class FirstModel(nn.Module):
        def forward(self, inputs):
            return inputs + 1

    class SecondModel(nn.Module):
        def forward(self, inputs):
            return inputs * 2

    first = trace(FirstModel())
    second = trace(SecondModel())

    assert first.graph_name == "DualGraphModule_FirstModel"
    assert second.graph_name == "DualGraphModule_SecondModel"
    assert type(first) is type(second)
    assert type(first).__name__ == "DualGraphModule"


def test_dual_graph_apply_visits_each_logical_module_once():
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        analyser,
    )
    visited = []

    def initialise(module):
        if isinstance(module, nn.Linear):
            visited.append(module)
            nn.init.constant_(module.weight, 3.0)

    watched_model.apply(initialise)

    assert len(visited) == 1
    for training in (True, False):
        operation = analyser.interjection.wrapped_module(
            "0",
            training=training,
        ).get_submodule("0")
        torch.testing.assert_close(
            operation.weight,
            torch.full_like(operation.weight, 3.0),
        )


def test_dual_graph_buffers_use_logical_names():
    watched_model = trace(_ModeStateModel())

    assert [name for name, _ in watched_model.named_buffers()] == [
        "train_scale",
        "eval_scale",
    ]


def test_dual_graph_buffers_remain_deduplicated_after_dtype_conversion():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))

        def forward(self, inputs):
            return inputs * self.scale

    watched_model = trace(Model()).double()

    assert [name for name, _ in watched_model.named_buffers()] == ["scale"]


def test_source_only_state_participates_in_dual_graph_lifecycle():
    source = _PartiallyUnusedStateModel()
    watched_model = trace(source)

    assert [name for name, _ in watched_model.named_parameters()] == [
        "root_parameter",
        "used.weight",
        "used.bias",
        "unused.weight",
        "unused.bias",
    ]
    assert [name for name, _ in watched_model.named_buffers()] == [
        "root_buffer",
    ]
    assert all(
        actual is expected
        for actual, expected in zip(
            watched_model.parameters(),
            source.parameters(),
            strict=True,
        )
    )

    watched_model.double()

    assert all(
        parameter.dtype == torch.float64
        for parameter in watched_model.parameters()
    )
    assert all(
        buffer.dtype == torch.float64
        for buffer in watched_model.buffers()
    )
    assert all(
        value.dtype == torch.float64
        for value in source.state_dict().values()
        if torch.is_tensor(value)
    )


def test_source_only_parameter_aliases_survive_dtype_conversion():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Parameter(torch.ones(1))
            self.second = self.first

        def forward(self, inputs):
            return inputs

    watched_model = trace(Model()).double()
    parameters = dict(watched_model.named_parameters(
        remove_duplicate=False,
    ))

    assert list(parameters) == ["first", "second"]
    assert parameters["first"] is parameters["second"]
    assert parameters["first"].dtype == torch.float64


def test_dual_graph_apply_visits_source_only_modules_once():
    source = _PartiallyUnusedStateModel()
    watched_model = trace(source)
    visited = []

    watched_model.apply(
        lambda module: visited.append(module)
        if isinstance(module, nn.Linear)
        else None
    )

    assert visited.count(source.used) == 1
    assert visited.count(source.unused) == 1
    assert len(visited) == 2


def test_dual_graph_checkpoint_uses_logical_names_and_loads_strictly():
    source = _ModeStateModel()
    source.train_layer.weight.data.fill_(2.0)
    source.eval_layer.weight.data.fill_(3.0)
    source.train_scale.fill_(4.0)
    source.eval_scale.fill_(5.0)
    watched_model = trace(_ModeStateModel())

    watched_model.load_state_dict(source.state_dict(), strict=True)
    checkpoint = watched_model.state_dict()

    assert set(checkpoint) == set(source.state_dict())
    assert all(
        not name.startswith(("train_module.", "eval_module."))
        for name in checkpoint
    )
    restored = _ModeStateModel()
    restored.load_state_dict(checkpoint, strict=True)
    for training in (True, False):
        inputs = torch.ones(1, 2)
        watched_model.train(training)
        restored.train(training)
        torch.testing.assert_close(
            watched_model(inputs),
            restored(inputs),
        )


def test_dual_graph_checkpoint_preserves_state_unused_by_forward_graph():
    source = _PartiallyUnusedStateModel(extra_state=23)
    source.root_parameter.data.fill_(2.0)
    source.root_buffer.fill_(3.0)
    source.used.weight.data.fill_(4.0)
    source.unused.weight.data.fill_(5.0)
    watched_model = trace(_PartiallyUnusedStateModel())

    watched_model.load_state_dict(source.state_dict(), strict=True)
    checkpoint = watched_model.state_dict()

    assert list(checkpoint) == list(source.state_dict())
    for name, value in source.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(checkpoint[name], value)
        else:
            assert checkpoint[name] == value

    restored = _PartiallyUnusedStateModel()
    restored.load_state_dict(checkpoint, strict=True)
    assert restored.extra_state == 23
    torch.testing.assert_close(restored.root_parameter, torch.full((1,), 2.0))
    torch.testing.assert_close(restored.root_buffer, torch.full((1,), 3.0))
    torch.testing.assert_close(
        restored.unused.weight,
        torch.full_like(restored.unused.weight, 5.0),
    )


def test_missing_unused_source_state_is_reported_with_outer_prefix():
    model = nn.Module()
    model.watched = trace(_PartiallyUnusedStateModel())
    checkpoint = model.state_dict()
    del checkpoint["watched.unused.weight"]

    incompatible = model.load_state_dict(checkpoint, strict=False)

    assert incompatible.missing_keys == ["watched.unused.weight"]


def test_interjected_checkpoint_uses_original_model_state_names():
    source = nn.Sequential(nn.Linear(2, 1))
    watched_model = interject_by_module_class(
        source,
        nn.Linear,
        _ModuleRecorder(),
    )
    replacement_state = {
        "0.weight": torch.full((1, 2), 3.0),
        "0.bias": torch.full((1,), 4.0),
    }

    assert list(watched_model.state_dict()) == list(source.state_dict())
    watched_model.load_state_dict(replacement_state, strict=True)

    for training in (True, False):
        watched_model.train(training)
        torch.testing.assert_close(
            watched_model(torch.ones(1, 2)),
            torch.full((1, 1), 10.0),
        )

    restored = nn.Sequential(nn.Linear(2, 1))
    restored.load_state_dict(watched_model.state_dict(), strict=True)
    torch.testing.assert_close(
        restored(torch.ones(1, 2)),
        torch.full((1, 1), 10.0),
    )


def test_interjected_checkpoint_uses_original_buffer_names():
    source = nn.Sequential(nn.BatchNorm1d(2))
    watched_model = interject_by_module_class(
        source,
        nn.BatchNorm1d,
        _ModuleRecorder(),
    )
    checkpoint = {
        name: value.clone()
        for name, value in source.state_dict().items()
    }
    checkpoint["0.running_mean"].fill_(5.0)

    assert list(watched_model.state_dict()) == list(source.state_dict())
    watched_model.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(
        watched_model.state_dict()["0.running_mean"],
        torch.full((2,), 5.0),
    )


def test_interjected_checkpoint_translates_original_extra_state():
    source = nn.Sequential(_ExtraStateLinear(2, 1, value=11))
    watched_model = interject_by_module_class(
        source,
        _ExtraStateLinear,
        _ModuleRecorder(),
    )
    checkpoint = source.state_dict()
    checkpoint["0._extra_state"] = 23

    assert list(watched_model.state_dict()) == list(source.state_dict())
    watched_model.load_state_dict(checkpoint, strict=True)

    interjection = next(
        module
        for module in watched_model.modules()
        if isinstance(module, _ModuleRecorder)
    )
    for training in (True, False):
        operation = interjection.wrapped_module(
            "0",
            training=training,
        ).get_submodule("0")
        assert operation.value == 23

    restored = nn.Sequential(_ExtraStateLinear(2, 1))
    restored.load_state_dict(watched_model.state_dict(), strict=True)
    assert restored[0].value == 23


def test_tensor_extra_state_is_not_shape_validated_as_a_parameter():
    source = nn.Sequential(
        _ExtraStateLinear(2, 1, value=torch.ones(2)),
    )
    watched_model = interject_by_module_class(
        source,
        _ExtraStateLinear,
        _ModuleRecorder(),
    )
    checkpoint = source.state_dict()
    checkpoint["0._extra_state"] = torch.ones(3)

    watched_model.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(
        watched_model.state_dict()["0._extra_state"],
        torch.ones(3),
    )


def test_interjected_checkpoint_translates_module_version_metadata():
    source = nn.Sequential(_VersionedLinear(2, 1))
    watched_model = interject_by_module_class(
        source,
        _VersionedLinear,
        _ModuleRecorder(),
    )

    watched_model.load_state_dict(source.state_dict(), strict=True)

    assert source[0].loaded_versions
    assert set(source[0].loaded_versions) == {_VersionedLinear._version}

    restored = nn.Sequential(_VersionedLinear(2, 1))
    restored.load_state_dict(watched_model.state_dict(), strict=True)
    assert restored[0].loaded_versions == [_VersionedLinear._version]


def test_dual_graph_load_hooks_run_once_in_the_original_namespace():
    source = nn.Sequential(nn.Linear(2, 1))
    pre_hook_prefixes = []
    post_hook_calls = []
    source[0].register_load_state_dict_pre_hook(
        lambda _module, _state, prefix, *_args: (
            pre_hook_prefixes.append(prefix)
        )
    )
    source[0].register_load_state_dict_post_hook(
        lambda module, _incompatible: post_hook_calls.append(module)
    )
    watched_model = trace(source)

    watched_model.load_state_dict(watched_model.state_dict(), strict=True)

    assert pre_hook_prefixes == ["0."]
    assert post_hook_calls == [source[0]]


def test_nested_source_load_hooks_see_public_diagnostics():
    source = nn.Sequential(nn.Linear(2, 1))
    pre_hook_prefixes = []
    post_hook_diagnostics = []
    source[0].register_load_state_dict_pre_hook(
        lambda _module, _state, prefix, *_args: (
            pre_hook_prefixes.append(prefix)
        )
    )
    source[0].register_load_state_dict_post_hook(
        lambda _module, incompatible: post_hook_diagnostics.append((
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        ))
    )
    container = nn.Module()
    container.watched = trace(source)

    incompatible = container.load_state_dict({}, strict=False)

    expected_missing = ["watched.0.weight", "watched.0.bias"]
    assert pre_hook_prefixes == ["watched.0."]
    assert post_hook_diagnostics == [(expected_missing, [])]
    assert incompatible.missing_keys == expected_missing


def test_tracing_does_not_run_state_dict_hooks():
    source = nn.Sequential(nn.Linear(2, 1))
    calls = []
    source[0].register_state_dict_pre_hook(
        lambda _module, prefix, _keep_vars: calls.append(("pre", prefix))
    )
    source[0].register_state_dict_post_hook(
        lambda _module, _state, prefix, _metadata: (
            calls.append(("post", prefix))
        )
    )

    trace(source)

    assert calls == []


def test_dual_graph_state_dict_hooks_run_once_in_original_namespace():
    source = nn.Sequential(nn.Linear(2, 1))
    calls = []

    def post_hook(_module, state, prefix, _metadata):
        calls.append(("post", prefix))
        state[f"{prefix}weight"] = state[f"{prefix}weight"] + 1

    source[0].register_state_dict_pre_hook(
        lambda _module, prefix, _keep_vars: calls.append(("pre", prefix))
    )
    source[0].register_state_dict_post_hook(post_hook)
    container = nn.Module()
    container.watched = trace(source)

    checkpoint = container.state_dict()

    assert calls == [
        ("pre", "watched.0."),
        ("post", "watched.0."),
    ]
    torch.testing.assert_close(
        checkpoint["watched.0.weight"],
        source[0].weight.detach() + 1,
    )


def test_state_dict_hook_added_keys_round_trip_through_load_hook():
    source = nn.Sequential(nn.Linear(2, 1))
    loaded = []

    def save_hook(_module, state, prefix, _metadata):
        state[f"{prefix}custom"] = torch.tensor(7)

    def load_hook(_module, state, prefix, *_args):
        loaded.append(state.pop(f"{prefix}custom").item())

    source[0].register_state_dict_post_hook(save_hook)
    source[0].register_load_state_dict_pre_hook(load_hook)
    watched_model = trace(source)
    checkpoint = watched_model.state_dict()

    watched_model.load_state_dict(checkpoint, strict=True)

    assert loaded == [7]


def test_state_dict_hook_renamed_key_loads_mapped_graph_state():
    source = nn.Sequential(nn.Linear(2, 1))

    def save_hook(_module, state, prefix, _metadata):
        state[f"{prefix}renamed_weight"] = state.pop(f"{prefix}weight")

    def load_hook(_module, state, prefix, *_args):
        renamed_key = f"{prefix}renamed_weight"
        if renamed_key in state:
            state[f"{prefix}weight"] = state.pop(renamed_key)

    source[0].register_state_dict_post_hook(save_hook)
    source[0].register_load_state_dict_pre_hook(load_hook)
    container = nn.Module()
    container.watched = trace(source)
    checkpoint = container.state_dict()
    checkpoint["watched.0.renamed_weight"] = torch.full((1, 2), 4.0)

    container.load_state_dict(checkpoint, strict=True, assign=True)

    train_weight = container.watched.train_module.get_parameter("0.weight")
    eval_weight = container.watched.eval_module.get_parameter("0.weight")
    assert train_weight is source[0].weight
    assert eval_weight is source[0].weight
    torch.testing.assert_close(
        source[0].weight,
        torch.full((1, 2), 4.0),
    )


def test_auxiliary_state_dict_hooks_run_once_in_public_namespace():
    interjection = _StatefulInterjection()
    calls = []
    interjection.register_state_dict_pre_hook(
        lambda _module, prefix, _keep_vars: calls.append(("pre", prefix))
    )
    interjection.register_state_dict_post_hook(
        lambda _module, _state, prefix, _metadata: (
            calls.append(("post", prefix))
        )
    )
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )

    checkpoint = watched_model.state_dict()

    prefix = "_torchwatcher._StatefulInterjection0."
    assert calls == [("pre", prefix), ("post", prefix)]
    assert f"{prefix}marker" in checkpoint


def test_auxiliary_hook_renamed_key_loads_once():
    interjection = _StatefulInterjection()
    load_prefixes = []

    def save_hook(_module, state, prefix, _metadata):
        state[f"{prefix}renamed_marker"] = state.pop(f"{prefix}marker")

    def load_hook(_module, state, prefix, *_args):
        load_prefixes.append(prefix)
        renamed_key = f"{prefix}renamed_marker"
        if renamed_key in state:
            state[f"{prefix}marker"] = state.pop(renamed_key)

    interjection.register_state_dict_post_hook(save_hook)
    interjection.register_load_state_dict_pre_hook(load_hook)
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )
    checkpoint = watched_model.state_dict()
    prefix = "_torchwatcher._StatefulInterjection0."
    checkpoint[f"{prefix}renamed_marker"] = torch.tensor(9.0)

    watched_model.load_state_dict(checkpoint, strict=True)

    assert load_prefixes == [prefix]
    torch.testing.assert_close(interjection.marker, torch.tensor(9.0))


def test_failed_dual_graph_load_restores_temporary_adapters():
    source = nn.Sequential(_FailingVersionedLinear(2, 1))
    watched_model = trace(source)
    original_pre_hooks = source[0]._load_state_dict_pre_hooks
    original_post_hooks = source[0]._load_state_dict_post_hooks

    with pytest.raises(RuntimeError, match="deliberate load failure"):
        watched_model.load_state_dict(
            watched_model.state_dict(),
            strict=True,
        )

    assert source[0]._load_state_dict_pre_hooks is original_pre_hooks
    assert source[0]._load_state_dict_post_hooks is original_post_hooks
    assert "_torchwatcher_version_load_patch" not in source[0].__dict__
    assert (
        "_torchwatcher_load_state_dict_contexts"
        not in watched_model.__dict__
    )

    source[0].fail_loading = False
    watched_model.load_state_dict(watched_model.state_dict(), strict=True)


def test_nested_dual_graph_load_supplies_current_descendant_versions():
    source = nn.Sequential(_VersionedLinear(2, 1))
    watched_model = interject_by_module_class(
        source,
        _VersionedLinear,
        _ModuleRecorder(),
    )
    source[0].loaded_versions.clear()
    container = nn.Module()
    container.watched = watched_model
    checkpoint_source = nn.Module()
    checkpoint_source.watched = nn.Sequential(_VersionedLinear(2, 1))

    container.load_state_dict(checkpoint_source.state_dict(), strict=True)

    assert source[0].loaded_versions
    assert set(source[0].loaded_versions) == {_VersionedLinear._version}
    assert all(
        "_torchwatcher_version_load_patch" not in module.__dict__
        for module in watched_model.modules()
    )


def test_interjected_checkpoint_translates_descendant_extra_state():
    source = nn.Sequential(_ExtraStateBlock(value=11))
    watched_model = interject_by_module_class(
        source,
        _ExtraStateBlock,
        _ModuleRecorder(),
    )
    checkpoint = source.state_dict()
    checkpoint["0.linear._extra_state"] = 23

    watched_model.load_state_dict(checkpoint, strict=True)

    assert watched_model.state_dict()["0.linear._extra_state"] == 23


def test_interjection_extra_state_remains_in_auxiliary_namespace():
    interjection = _ExtraStateInterjection()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )

    checkpoint = watched_model.state_dict()

    assert checkpoint[
        "_torchwatcher._ExtraStateInterjection0._extra_state"
    ] == 5
    interjection.value = 9
    watched_model.load_state_dict(checkpoint, strict=True)
    assert interjection.value == 5


def test_interjected_checkpoint_preserves_original_shared_alias_names():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

        def forward(self, inputs):
            return self.right(self.left(inputs))

    source = Model()
    watched_model = interject_by_module_class(
        source,
        nn.Linear,
        _ModuleRecorder(),
    )

    assert set(watched_model.state_dict()) == set(source.state_dict())
    watched_model.load_state_dict(source.state_dict(), strict=True)


def test_original_checkpoint_keeps_auxiliary_interjection_state_initialised():
    source = nn.Sequential(nn.Linear(2, 1))
    interjection = _StatefulInterjection()
    watched_model = interject_by_module_class(
        source,
        nn.Linear,
        interjection,
    )

    watched_model.load_state_dict(source.state_dict(), strict=True)

    torch.testing.assert_close(interjection.marker, torch.tensor(3.0))


def test_full_checkpoint_restores_namespaced_interjection_state():
    interjection = _StatefulInterjection()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        interjection,
    )
    checkpoint = {
        name: value.clone()
        for name, value in watched_model.state_dict().items()
    }
    auxiliary_names = [
        name
        for name in checkpoint
        if name.startswith("_torchwatcher.")
    ]

    assert auxiliary_names == [
        "_torchwatcher._StatefulInterjection0.marker"
    ]
    interjection.marker.fill_(9.0)
    watched_model.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(interjection.marker, torch.tensor(3.0))


def test_nested_dual_graph_checkpoint_uses_logical_names():
    source = nn.Module()
    source.watched = trace(_ModeStateModel())
    source.watched.train_module.train_scale.fill_(4.0)
    source.watched.eval_module.eval_scale.fill_(5.0)
    checkpoint = source.state_dict()

    assert "watched.train_scale" in checkpoint
    assert "watched.eval_scale" in checkpoint
    assert not any(".train_module." in name for name in checkpoint)
    assert not any(".eval_module." in name for name in checkpoint)

    restored = nn.Module()
    restored.watched = trace(_ModeStateModel())
    restored.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(
        restored.watched.train_module.train_scale,
        torch.full((2,), 4.0),
    )
    torch.testing.assert_close(
        restored.watched.eval_module.eval_scale,
        torch.full((2,), 5.0),
    )


def test_dual_graph_load_diagnostics_use_logical_names():
    watched_model = trace(nn.Sequential(nn.Linear(2, 1)))

    incompatible = watched_model.load_state_dict({}, strict=False)

    assert incompatible.missing_keys == ["0.weight", "0.bias"]
    assert incompatible.unexpected_keys == []


def test_interjected_load_diagnostics_use_original_names():
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        _ModuleRecorder(),
    )

    incompatible = watched_model.load_state_dict({}, strict=False)

    assert incompatible.missing_keys == ["0.weight", "0.bias"]
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize("interjected", [False, True])
def test_size_mismatch_uses_one_logical_name(interjected):
    watched_model = (
        interject_by_module_class(
            nn.Sequential(nn.Linear(2, 1)),
            nn.Linear,
            _ModuleRecorder(),
        )
        if interjected
        else trace(nn.Sequential(nn.Linear(2, 1)))
    )
    checkpoint = watched_model.state_dict()
    checkpoint["0.weight"] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError) as error:
        watched_model.load_state_dict(checkpoint, strict=True)

    message = str(error.value)
    assert message.count("size mismatch for 0.weight:") == 1
    assert "train_module" not in message
    assert "eval_module" not in message
    assert ".interjection." not in message


def test_nested_size_mismatch_preserves_outer_prefix():
    model = nn.Module()
    model.watched = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        _ModuleRecorder(),
    )
    checkpoint = model.state_dict()
    checkpoint["watched.0.weight"] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError) as error:
        model.load_state_dict(checkpoint, strict=True)

    message = str(error.value)
    assert message.count("size mismatch for watched.0.weight:") == 1
    assert "train_module" not in message
    assert "eval_module" not in message


def test_size_mismatch_with_assign_keeps_the_existing_parameter():
    watched_model = trace(nn.Sequential(nn.Linear(2, 1)))
    parameter = watched_model.get_parameter("train_module.0.weight")
    checkpoint = watched_model.state_dict()
    checkpoint["0.weight"] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError, match="size mismatch for 0.weight"):
        watched_model.load_state_dict(checkpoint, strict=True, assign=True)

    assert watched_model.get_parameter("train_module.0.weight") is parameter


def test_size_mismatch_uses_final_shared_alias_once():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

        def forward(self, inputs):
            return self.right(self.left(inputs))

    watched_model = interject_by_module_class(
        Model(),
        nn.Linear,
        _ModuleRecorder(),
    )
    checkpoint = watched_model.state_dict()
    checkpoint["right.weight"] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError) as error:
        watched_model.load_state_dict(checkpoint, strict=True)

    message = str(error.value)
    assert message.count("size mismatch for right.weight:") == 1
    assert "left.weight" not in message


def test_shape_validation_preserves_scalar_checkpoint_compatibility():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.0))

        def forward(self, inputs):
            return inputs * self.scale

    watched_model = trace(Model())
    checkpoint = watched_model.state_dict()
    checkpoint["scale"] = torch.tensor([3.0])

    watched_model.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(
        watched_model.state_dict()["scale"],
        torch.tensor(3.0),
    )


def test_shape_validation_preserves_lazy_parameter_loading():
    watched_model = trace(nn.Sequential(nn.LazyLinear(2)))
    checkpoint = nn.Sequential(nn.Linear(3, 2)).state_dict()

    watched_model.load_state_dict(checkpoint, strict=True)

    assert watched_model.state_dict()["0.weight"].shape == (2, 3)


def test_nested_dual_graph_load_diagnostics_preserve_outer_prefix():
    model = nn.Module()
    model.watched = trace(nn.Sequential(nn.Linear(2, 1)))

    incompatible = model.load_state_dict({}, strict=False)

    assert incompatible.missing_keys == [
        "watched.0.weight",
        "watched.0.bias",
    ]


def test_dual_graph_load_diagnostics_do_not_rewrite_sibling_paths():
    model = nn.Module()
    model.sibling = nn.Module()
    model.sibling.train_module = nn.Sequential(nn.Linear(2, 1))
    model.watched = trace(nn.Sequential(nn.Linear(2, 1)))

    incompatible = model.load_state_dict({}, strict=False)

    assert incompatible.missing_keys == [
        "sibling.train_module.0.weight",
        "sibling.train_module.0.bias",
        "watched.0.weight",
        "watched.0.bias",
    ]


def test_partial_shared_alias_checkpoint_reports_original_missing_alias():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(2, 2)
            self.left = shared
            self.right = shared

        def forward(self, inputs):
            return self.right(self.left(inputs))

    watched_model = interject_by_module_class(
        Model(),
        nn.Linear,
        _ModuleRecorder(),
    )
    checkpoint = watched_model.state_dict()
    del checkpoint["right.weight"]

    incompatible = watched_model.load_state_dict(checkpoint, strict=False)

    assert incompatible.missing_keys == ["right.weight"]


def test_wrapped_backward_interjection_can_be_pickled():
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        _RecordBackward(),
    )

    restored = pickle.loads(pickle.dumps(watched_model))
    inputs = torch.ones(1, 2, requires_grad=True)
    restored(inputs).sum().backward()

    restored_interjection = next(
        module
        for module in restored.modules()
        if isinstance(module, _RecordBackward)
    )
    assert restored_interjection.calls == 1


def test_trace_rejects_mixed_mode_module_traced_through_by_fx():
    class ModeDependentBlock(nn.Module):
        def forward(self, inputs):
            return inputs + 1 if self.training else inputs - 1

    model = nn.Sequential(ModeDependentBlock())
    model.train()
    model[0].eval()

    with pytest.raises(
            ValueError,
            match="mixed training modes.*'0'.*leaf_modules"):
        trace(model)

    assert model.training
    assert not model[0].training


def test_trace_allows_mixed_mode_custom_leaf_module():
    class ModeDependentBlock(nn.Module):
        def forward(self, inputs):
            return inputs + 1 if self.training else inputs - 1

    model = nn.Sequential(ModeDependentBlock())
    model.train()
    model[0].eval()

    trace(model, tracer_kwargs={"leaf_modules": [ModeDependentBlock]})

    assert model.training
    assert not model[0].training


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


@pytest.mark.parametrize(
    ("model_training", "analyser_training"),
    [(True, False), (False, True)],
)
def test_trace_shapes_restores_independent_analyser_mode(
        model_training,
        analyser_training):
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.ReLU()),
        nn.ReLU,
        analyser,
    )
    watched_model.train(model_training)
    analyser.train(analyser_training)

    trace_shapes(watched_model, torch.ones(1, 2))

    assert watched_model.training is model_training
    assert analyser.training is analyser_training


class _TrainSideEffect(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = "train"

    def train(self, mode: bool = True):
        super().train(mode)
        self.phase = "train" if mode else "eval"
        return self

    def forward(self, inputs):
        return inputs


def test_trace_shapes_restores_custom_train_side_effects():
    watched_model = trace(
        nn.Sequential(_TrainSideEffect()),
        tracer_kwargs={"leaf_modules": [_TrainSideEffect]},
    )
    module = watched_model.train_module.get_submodule("0")
    watched_model.train()

    trace_shapes(watched_model, torch.ones(1, 2))

    assert module.training
    assert module.phase == "train"


def test_trace_shapes_restores_functorch_config_after_failure():
    import torch._functorch.config as functorch_config

    previous_device_type = functorch_config.fake_tensor_prefer_device_type
    functorch_config.fake_tensor_prefer_device_type = "cpu"
    analyser = NameAnalyser()
    watched_model = interject_by_module_class(
        nn.Sequential(nn.Linear(2, 1)),
        nn.Linear,
        analyser,
    )
    watched_model.train()
    analyser.eval()

    try:
        with pytest.raises(RuntimeError):
            trace_shapes(watched_model, torch.ones(1, 3))

        assert functorch_config.fake_tensor_prefer_device_type == "cpu"
        assert watched_model.training
        assert not analyser.training
    finally:
        functorch_config.fake_tensor_prefer_device_type = previous_device_type
