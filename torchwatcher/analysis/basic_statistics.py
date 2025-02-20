from .analysis import Analyzer, OUTPUTS, AnalyzerState
from .running_stats import Variance


class FeatureStatistics:
    def __init__(self):
        self.channel_sparsity = Variance()
        self.channel_activations = Variance()
        self.feature_activations = Variance()


class FeatureStats(Analyzer):
    def __init__(self):
        super().__init__()

    def process_batch_state(self, name: str, state: AnalyzerState,
                            working_results: FeatureStatistics | None) \
        -> FeatureStatistics:
        features = state[OUTPUTS]
        channel_features = features.view(features.shape[0], 1, -1)

        if working_results is None:
            working_results = FeatureStatistics()

        working_results.channel_sparsity.add(
            channel_features.count_nonzero(dim=-1) / channel_features.shape[-1])
        working_results.channel_activations.add(channel_features.mean(dim=-1))
        working_results.feature_activations.add(features)

        return working_results

    def result_to_dict(self, result: FeatureStatistics) -> dict:
        rec = dict()

        rec['channel_sparsity_mean'] = result.channel_sparsity.mean()
        rec['channel_sparsity_var'] = result.channel_sparsity.variance()
        rec['channel_activations_mean'] = result.channel_activations.mean()
        rec['channel_activations_var'] = result.channel_activations.variance()
        rec['feature_activations_mean'] = result.feature_activations.mean()
        rec['feature_activations_var'] = result.feature_activations.variance()

        return rec
