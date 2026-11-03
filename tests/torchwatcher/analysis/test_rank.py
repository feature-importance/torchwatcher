import pytest
import torch
from torch import nn

from torchwatcher.analysis import (
    AnalyserList,
    CovarianceSpectrumStatisticsAnalyser,
    LayerWeightRankAnalyser,
    PerChannelWeightRankAnalyser,
    RankAnalyser,
)
from torchwatcher.interjection import interject_by_module_class, interject_by_name


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_weight_rank_analyser_supports_low_precision_weights(dtype):
    model = nn.Sequential(nn.Linear(3, 2, bias=False)).to(dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))
    analyser = LayerWeightRankAnalyser()
    watched_model = interject_by_module_class(model, nn.Linear, analyser)

    watched_model(torch.ones(1, 3, dtype=dtype))

    assert analyser.to_dict() == {
        "0": {"layer_weight_rank": 2},
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_per_channel_weight_rank_analyser_supports_low_precision_weights(dtype):
    model = nn.Sequential(nn.Linear(2, 2, bias=False)).to(dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.0, 0.0],
        ]))
    analyser = PerChannelWeightRankAnalyser()
    watched_model = interject_by_module_class(model, nn.Linear, analyser)

    watched_model(torch.ones(1, 2, dtype=dtype))

    result = analyser.to_dict()["0"]["per_channel_weight_rank"]
    torch.testing.assert_close(result, torch.tensor([1, 0]))


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


def test_rank_analyser_accepts_non_contiguous_activations():
    features = torch.arange(24.0).reshape(3, 2, 2, 2).to(
        memory_format=torch.channels_last
    )
    assert not features.is_contiguous()
    analyser = RankAnalyser(n=-1)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    result = analyser.to_dict()["0"]
    assert result["features_dim"] == 8


def test_rank_analyser_non_positive_sample_size_uses_every_feature():
    features = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 4.0, 1.0],
        [3.0, 6.0, 0.0],
    ])
    analyser = RankAnalyser(n=-1)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    assert analyser.to_dict()["0"]["features_rank"] == 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["aTa", "aaT", "a"])
def test_rank_analyser_supports_low_precision_activations(mode, dtype):
    features = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=dtype)
    analyser = RankAnalyser(mode=mode, n=-1)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    assert analyser.to_dict()["0"]["features_rank"] == 2


@pytest.mark.parametrize("mode", ["a", "aaT"])
def test_rank_analyser_retains_features_before_downstream_inplace_update(mode):
    features = torch.tensor([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
    ])
    analyser = RankAnalyser(mode=mode, n=-1)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity(), nn.ReLU(inplace=True)),
        "0",
        analyser,
    )

    watched_model(features.clone())

    assert analyser.to_dict()["0"]["features_rank"] == 2


@pytest.mark.parametrize("n", [-1, 0])
def test_covariance_spectrum_non_positive_sample_size_uses_full_dimension(n):
    features = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    analyser = CovarianceSpectrumStatisticsAnalyser(n=n)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    result = analyser.to_dict()["0"]
    assert result["features_dim"] == 3
    assert result["features_rank"] == 3
    assert result["normalized_features_rank"] == 1.0


@pytest.mark.parametrize(
    ("mode", "expected_rank"),
    [("aTa", 2), ("aaT", 1), ("a", 1)],
)
def test_rank_analyser_samples_the_axis_selected_by_mode(mode, expected_rank):
    values = torch.arange(16)
    features = torch.stack([
        2 * ((values >> bit) & 1) - 1
        for bit in range(4)
    ], dim=1)
    features = features.float()
    analyser = RankAnalyser(mode=mode, n=2)
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features[:8])
    watched_model(features[8:])

    assert analyser.to_dict()["0"]["features_rank"] == expected_rank


def test_rank_analyser_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        RankAnalyser(mode="unknown")
