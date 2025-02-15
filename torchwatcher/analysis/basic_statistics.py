import torch
from torch import nn

from .analysis import Analyser
from .running_stats import Variance


class FeatureStats(Analyser):
    def __init__(self):
        self.channel_sparsity = Variance()
        self.channel_activations = Variance()
        self.feature_activations = Variance()

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        channel_features = features.view(features.shape[0], 1, -1)

        self.channel_sparsity.add(channel_features.count_nonzero(dim=-1) / channel_features.shape[-1])
        self.channel_activations.add(channel_features.mean(dim=-1))
        self.feature_activations.add(features)

    def get_result(self) -> dict:
        rec = dict()
        rec['channel_sparsity_mean'] = self.channel_sparsity.mean()
        rec['channel_sparsity_var'] = self.channel_sparsity.variance()
        rec['channel_activations_mean'] = self.channel_activations.mean()
        rec['channel_activations_var'] = self.channel_activations.variance()
        rec['feature_activations_mean'] = self.feature_activations.mean()
        rec['feature_activations_var'] = self.feature_activations.variance()
        return rec


