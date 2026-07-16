from torch import nn

from torchwatcher.interjection.node_selector import (
    first,
    is_module,
    last,
    match_slice,
    nth,
)
from torchwatcher.interjection.tracing import symbolic_trace


def _selected_names(model, selector):
    graph_module = symbolic_trace(model)
    return [
        node.name
        for node in graph_module.graph.nodes
        if selector.fn((graph_module, node))
    ]


def _model():
    return nn.Sequential(
        nn.Linear(1, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    )


def test_nth_selects_matching_node_by_positive_index():
    assert _selected_names(_model(), nth(is_module, 1)) == ["_1"]


def test_nth_selects_matching_node_by_negative_index():
    assert _selected_names(_model(), nth(is_module, -1)) == ["_4"]


def test_first_and_last_select_matching_boundaries():
    assert _selected_names(_model(), first(is_module)) == ["_0"]
    assert _selected_names(_model(), last(is_module)) == ["_4"]


def test_nth_out_of_range_selects_no_nodes():
    assert _selected_names(_model(), nth(is_module, 99)) == []
    assert _selected_names(_model(), nth(is_module, -99)) == []


def test_match_slice_selects_matching_node_range():
    assert _selected_names(_model(), match_slice(is_module, 1, 4, 2)) == [
        "_1",
        "_3",
    ]
