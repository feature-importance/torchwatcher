import torch
from torch.linalg import LinAlgError, matrix_rank

from .analysis import Analyser, AnalyserState
from .running_stats import Covariance


def estimate_rank(features: torch.Tensor,
                  n: int = 8000,
                  mode: str = 'aTa',
                  threshold: float = 1e-3) -> int:
    """Estimate the rank of mean-centred features matrix (equivalent to rank of
    covariance matrix of either features or samples)

    Args:
        features: the features, one example per row
        n: the amount to sample in order to reduce computation time. -1
            to
        mode: 'aTa' to use features covariance; 'aaT' to use examples x
        threshold: threshold as a percentage of largest s.v. to use to
    disable sampling.
    examples; 'a' to use mean-centred features matrix
    estimate the rank

    Returns:
        the estimated rank
    """
    if mode == 'aTa':
        if n > 0:
            perm = torch.randperm(features.shape[1])
            idx = perm[:n]
            f = features[:, idx]
        else:
            f = features

        cov = torch.cov(f.T)
        return torch.linalg.matrix_rank(cov, hermitian=True,
                                        rtol=threshold).cpu().item()
    elif mode == 'aaT':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        cov = (f - f.mean(dim=0)) @ (f - f.mean(dim=0)).T
        return torch.linalg.matrix_rank(cov, hermitian=True,
                                        rtol=threshold).cpu().item()
    elif mode == 'a':
        if n > 0:
            perm = torch.randperm(features.shape[0])
            idx = perm[:n]
            f = features[idx, :]
        else:
            f = features

        s = torch.linalg.svdvals(f - f.mean(dim=0)) ** 2
        return (s > (s.max() * threshold)).sum().cpu().item()

    raise ValueError('Unknown mode')


def compute_cov_spectrum_stats(covariance: torch.Tensor,
                               threshold=1e-3,
                               taps=10) -> dict:
    s = torch.linalg.svdvals(covariance)

    stats = dict()
    stats['mean'] = s.mean().cpu().item()
    stats['max'] = s[0].item()
    stats['features_rank'] = (s > (s.max() * threshold)).sum().cpu().item()
    stats['features_rank_val'] = s[stats['features_rank'] - 1].item()
    stats['half_rank_val'] = s[(stats['features_rank'] - 1) // 2].item()
    stats['quarter_rank_val'] = s[(stats['features_rank'] - 1) // 4].item()

    spectrum = torch.nn.functional.interpolate(s.view(1, 1, -1), size=taps,
                                               mode='nearest')[0, 0].cpu()
    for i in range(taps):
        stats[f'normalised_spectrum_{i}'] = spectrum[i].item()

    return stats


class RankAnalyser(Analyser):
    def __init__(self, mode='aTa',
                 n=8000,
                 threshold=1e-3):
        super().__init__()

        self.n = n
        self.indices = {}
        self.features_dim = {}
        self.mode = mode
        self.threshold = threshold

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: Covariance | None) -> Covariance:
        if working_results is None:
            working_results = Covariance()

        features = state.outputs.detach()

        f = features.view(features.shape[0], -1)
        if not name in self.indices:
            self.features_dim[name] = f.shape[1]
            self.indices[name] = torch.randperm(f.shape[1])[:self.n]

        f = f[:, self.indices[name]]
        working_results.add(f)

        return working_results

    def finalise_result(self, name: str, result: Covariance) -> dict:
        try:
            covar = result.covariance()
            rank = matrix_rank(covar, hermitian=True,
                               rtol=self.threshold).cpu().item()

            rec = dict()
            rec['features_rank'] = rank
            rec['features_dim'] = self.features_dim[name]

            norm = min(self.features_dim[name], covar.shape[0])
            rec['normalized_features_rank'] = rank / norm

            return rec
        except LinAlgError:
            return {}


class LayerWeightRankAnalyser(Analyser[int]):
    """Compute the rank of a layer's flattened output-filter matrix.

    The first weight dimension is treated as the output-channel dimension and
    every remaining dimension is flattened into one feature dimension. The
    result is computed once per watched layer after each analyser reset.
    """

    def __init__(self, threshold=1e-3):
        super().__init__()
        self.threshold = threshold

    def process_batch_state(self, name: str, state: AnalyserState,
                            working_results: int | None) -> int | None:
        if working_results is None and state.module is not None:
            weight = getattr(state.module, 'weight', None)
            if weight is not None:
                matrix = weight.detach().reshape(weight.shape[0], -1)
                working_results = matrix_rank(
                    matrix,
                    hermitian=False,
                    rtol=self.threshold,
                ).cpu().item()

        return working_results

    def finalise_result(self, name: str, result: int | None) -> dict:
        return {'layer_weight_rank': result}


class PerChannelWeightRankAnalyser(Analyser[torch.Tensor]):
    """Compute one weight-matrix rank per output channel.

    Weights must have at least two dimensions. Dimension zero is interpreted as
    output channels and dimension one as input channels; any remaining kernel
    dimensions are flattened. For a convolutional weight tensor this computes
    the rank of each ``(input_channels, kernel_elements)`` filter matrix. For a
    linear layer each output-channel matrix has one column, so its rank is zero
    or one.
    """

    def __init__(self, threshold=1e-3):
        super().__init__()
        self.threshold = threshold

    def process_batch_state(
            self,
            name: str,
            state: AnalyserState,
            working_results: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if working_results is not None or state.module is None:
            return working_results

        weight = getattr(state.module, 'weight', None)
        if weight is None:
            return None
        if weight.ndim < 2:
            raise ValueError(
                "per-channel weight rank requires a weight tensor with at "
                "least two dimensions"
            )

        matrices = weight.detach().reshape(
            weight.shape[0],
            weight.shape[1],
            -1,
        )
        return matrix_rank(
            matrices,
            hermitian=False,
            rtol=self.threshold,
        ).cpu()

    def finalise_result(
            self,
            name: str,
            result: torch.Tensor | None,
    ) -> dict:
        return {'per_channel_weight_rank': result}


class CovarianceSpectrumStatisticsAnalyser(RankAnalyser):
    def __init__(self, n=8000, threshold=1e-3, taps=10):
        super().__init__(n=n, threshold=threshold)
        self.taps = taps

    def finalise_result(self, name, result) -> dict:
        stats = compute_cov_spectrum_stats(result.covariance(),
                                           threshold=self.threshold,
                                           taps=self.taps)
        stats['features_dim'] = self.features_dim[name]

        norm = min(self.features_dim[name], self.n)
        stats['normalized_features_rank'] = stats['features_rank'] / norm

        return stats
