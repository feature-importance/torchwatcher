import pytest
import torch
from torch import nn

from torchwatcher.analysis import FeatureStatistics
from torchwatcher.interjection import interject_by_name


def test_feature_statistics_preserves_channels_in_channel_summaries():
    features = torch.tensor([
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[2.0, 2.0], [2.0, 2.0]],
            [[0.0, 0.0], [0.0, 4.0]],
        ],
    ]).to(memory_format=torch.channels_last)
    assert not features.is_contiguous()
    analyser = FeatureStatistics()
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    result = analyser.to_dict()["0"]
    torch.testing.assert_close(
        result["channel_sparsity_mean"],
        torch.tensor([1.0, 0.125]),
    )
    torch.testing.assert_close(
        result["channel_sparsity_var"],
        torch.tensor([0.0, 0.03125]),
    )
    torch.testing.assert_close(
        result["channel_activations_mean"],
        torch.tensor([1.5, 0.5]),
    )
    torch.testing.assert_close(
        result["channel_activations_var"],
        torch.tensor([0.5, 0.5]),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_feature_statistics_accumulates_low_precision_features_in_float32(dtype):
    features = torch.tensor([
        [[1000.0]],
        [[-1000.0]],
    ], dtype=dtype)
    analyser = FeatureStatistics()
    watched_model = interject_by_name(
        nn.Sequential(nn.Identity()),
        "0",
        analyser,
    )

    watched_model(features)

    result = analyser.to_dict()["0"]
    for value in result.values():
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    torch.testing.assert_close(
        result["channel_activations_var"],
        torch.tensor([2_000_000.0]),
    )
    torch.testing.assert_close(
        result["feature_activations_var"],
        torch.tensor([[2_000_000.0]]),
    )
