import torch
from torch import nn

from torchwatcher.drawing import draw_graph, draw_graph_pretty
from torchwatcher.interjection import trace


def _dot_id(value):
    return value.strip('"')


def test_parameter_edges_use_declared_qualified_node_ids():
    graph = draw_graph(
        trace(nn.Sequential(nn.Linear(2, 1))),
        ignore_parameters_and_buffers=False,
    )
    node_ids = {_dot_id(node.get_name()) for node in graph.get_nodes()}
    edges = {
        (_dot_id(edge.get_source()), _dot_id(edge.get_destination()))
        for edge in graph.get_edges()
    }
    graph_name = "DualGraphModule_Sequential"
    operation_id = f"{graph_name}/_0"
    weight_id = f"{operation_id}.weight"
    bias_id = f"{operation_id}.bias"

    assert (weight_id, operation_id) in edges
    assert (bias_id, operation_id) in edges
    assert all(
        endpoint in node_ids
        for edge in edges
        for endpoint in edge
    )


def test_drawing_uses_the_dual_graph_instance_name():
    first = trace(nn.Sequential(nn.Linear(2, 1)))

    class OtherModel(nn.Module):
        def forward(self, inputs):
            return inputs + 1

    trace(OtherModel())
    graph_source = draw_graph(first).to_string()

    assert "DualGraphModule_Sequential" in graph_source
    assert "DualGraphModule_OtherModel" not in graph_source


class _NestedOutput(nn.Module):
    def forward(self, inputs):
        return {"full": (inputs, {"slice": inputs[:, :1]})}


def test_pretty_drawing_supports_nested_structured_outputs():
    graph = draw_graph_pretty(
        trace(_NestedOutput()),
        torch.ones(1, 2),
    )

    graph_source = graph.to_string()
    assert "(1, 2)" in graph_source
    assert "(1, 1)" in graph_source


class _KeywordOnlyInput(nn.Module):
    def forward(self, *, inputs):
        return inputs[:, :1]


def test_pretty_drawing_supports_keyword_only_inputs():
    graph = draw_graph_pretty(
        trace(_KeywordOnlyInput()),
        input_kwargs={"inputs": torch.ones(1, 2)},
    )

    graph_source = graph.to_string()
    assert "(1, 2)" in graph_source
    assert "(1, 1)" in graph_source


class _MixedInputs(nn.Module):
    def forward(self, inputs, *, other):
        return inputs + other


def test_pretty_drawing_supports_mixed_positional_and_keyword_inputs():
    graph = draw_graph_pretty(
        trace(_MixedInputs()),
        torch.ones(1, 2),
        input_kwargs={"other": torch.ones(1, 2)},
    )

    assert "(1, 2)" in graph.to_string()
