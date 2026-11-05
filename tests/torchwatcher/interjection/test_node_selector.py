import torch
from torch import nn

from torchwatcher.interjection.node_selector import (
    first,
    has_following,
    has_preceding,
    is_module,
    last,
    match_slice,
    matches_name,
    nth,
)
from torchwatcher.interjection.node_selector.node_types import Shuffle
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


def test_has_following_uses_inclusive_start_and_exclusive_stop():
    selector = (
        matches_name("_0")
        & has_following(matches_name("_2"), start=2, stop=3)
    )

    assert _selected_names(_model(), selector) == ["_0"]
    assert _selected_names(
        _model(),
        matches_name("_0")
        & has_following(matches_name("_2"), start=1, stop=2),
    ) == []


def test_has_preceding_uses_inclusive_start_and_exclusive_stop():
    selector = (
        matches_name("_2")
        & has_preceding(matches_name("_0"), start=2, stop=3)
    )

    assert _selected_names(_model(), selector) == ["_2"]
    assert _selected_names(
        _model(),
        matches_name("_2")
        & has_preceding(matches_name("_0"), start=1, stop=2),
    ) == []


class _BranchedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_1 = nn.Linear(2, 2)
        self.left_2 = nn.Linear(2, 2)
        self.right_1 = nn.Linear(2, 2)
        self.right_2 = nn.Linear(2, 2)

    def forward(self, inputs):
        left = self.left_2(self.left_1(inputs))
        right = self.right_2(self.right_1(inputs))
        return left + right


def test_has_following_ignores_later_nodes_on_unrelated_branch():
    selector = (
        matches_name("left_1")
        & has_following(matches_name("right_2"))
    )

    assert _selected_names(_BranchedModel(), selector) == []


def test_has_preceding_ignores_earlier_nodes_on_unrelated_branch():
    selector = (
        matches_name("right_2")
        & has_preceding(matches_name("left_1"))
    )

    assert _selected_names(_BranchedModel(), selector) == []


def test_shuffle_selectors_match_channel_and_pixel_shuffle_modules():
    assert _selected_names(
        nn.Sequential(nn.ChannelShuffle(2)),
        Shuffle.is_channel_shuffle,
    ) == ["_0"]
    assert _selected_names(
        nn.Sequential(nn.PixelShuffle(2)),
        Shuffle.is_pixel_shuffle,
    ) == ["_0"]
    assert _selected_names(
        nn.Sequential(nn.PixelUnshuffle(2)),
        Shuffle.is_pixel_unshuffle,
    ) == ["_0"]


def test_shuffle_selectors_match_functional_calls():
    class FunctionalShuffles(nn.Module):
        def forward(self, inputs):
            shuffled = torch.channel_shuffle(inputs, 2)
            shuffled = torch.pixel_shuffle(shuffled, 2)
            return torch.pixel_unshuffle(shuffled, 2)

    assert _selected_names(
        FunctionalShuffles(),
        Shuffle.is_shuffle,
    ) == ["channel_shuffle", "pixel_shuffle", "pixel_unshuffle"]
