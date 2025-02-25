import torch

from .analysis import Analyzer, AnalyzerState
from .running_stats import Variance


class _FeatureStatistics:
    def __init__(self):
        self.channel_sparsity = Variance()
        self.channel_activations = Variance()
        self.feature_activations = Variance()


class FeatureStats(Analyzer):
    """
    Compute basic statistics of feature maps
    """
    def __init__(self):
        super().__init__()

    def process_batch_state(self, name: str, state: AnalyzerState,
                            working_results: _FeatureStatistics | None) \
            -> _FeatureStatistics:
        features = state.outputs

        with torch.no_grad():
            channel_features = features.detach().view(features.shape[0], 1, -1)

            if working_results is None:
                working_results = _FeatureStatistics()

            working_results.channel_sparsity.add(
                channel_features.count_nonzero(dim=-1) / channel_features.shape[-1])
            working_results.channel_activations.add(channel_features.mean(dim=-1))
            working_results.feature_activations.add(features)

        return working_results

    def finalise_result(self, result: _FeatureStatistics) -> dict:
        rec = dict()

        rec['channel_sparsity_mean'] = result.channel_sparsity.mean()
        rec['channel_sparsity_var'] = result.channel_sparsity.variance()
        rec['channel_activations_mean'] = result.channel_activations.mean()
        rec['channel_activations_var'] = result.channel_activations.variance()
        rec['feature_activations_mean'] = result.feature_activations.mean()
        rec['feature_activations_var'] = result.feature_activations.variance()

        return rec
