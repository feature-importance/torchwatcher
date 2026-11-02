import torch
from torch import nn

from torchwatcher.analysis import (
    AnalyserList,
    LayerWeightRankAnalyser,
    PerChannelWeightRankAnalyser,
)
from torchwatcher.interjection import interject_by_module_class


def test_layer_weight_rank_analyser_reports_rank_for_interjected_module():
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))

    analyser = LayerWeightRankAnalyser()
    watched_model = interject_by_module_class(model, nn.Linear, analyser)

    watched_model(torch.ones(1, 3))

    assert analyser.to_dict() == {
        "0": {"layer_weight_rank": 2},
    }


def test_per_channel_weight_rank_analyser_reports_convolution_filter_ranks():
    model = nn.Sequential(nn.Conv2d(2, 3, 2, bias=False))
    weights = torch.zeros_like(model[0].weight)
    weights[0, 0, 0, 0] = 1.0
    weights[0, 1, 0, 0] = 2.0
    weights[1, 0, 0, 0] = 1.0
    weights[1, 1, 0, 1] = 1.0
    model[0].weight = nn.Parameter(
        weights.to(memory_format=torch.channels_last)
    )

    analyser = PerChannelWeightRankAnalyser()
    watched_model = interject_by_module_class(model, nn.Conv2d, analyser)

    watched_model(torch.ones(1, 2, 3, 3))

    result = analyser.to_dict()["0"]["per_channel_weight_rank"]
    torch.testing.assert_close(result, torch.tensor([1, 2, 0]))


def test_analyser_list_propagates_original_module_to_weight_rank_analyser():
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    analyser = LayerWeightRankAnalyser()
    watched_model = interject_by_module_class(
        model,
        nn.Linear,
        AnalyserList(analyser),
    )

    watched_model(torch.ones(1, 2))

    assert analyser.to_dict()["0"]["layer_weight_rank"] == 2
