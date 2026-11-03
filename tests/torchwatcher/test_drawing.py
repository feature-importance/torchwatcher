from torch import nn

from torchwatcher.drawing import draw_graph
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
