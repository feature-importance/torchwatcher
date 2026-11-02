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
    ])
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
