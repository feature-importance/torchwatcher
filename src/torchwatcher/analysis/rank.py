from typing import Any

import torch
from torch.linalg import LinAlgError, matrix_rank

from .analysis import Analyzer, AnalyzerState
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


class RankAnalyzer(Analyzer):
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
                            state: AnalyzerState,
                            working_results: Covariance | None) -> Covariance:
        if working_results is None:
            working_results = Covariance()

        features = state.outputs

        if not name in self.indices:
            self.features_dim[name] = features.shape[1]
            self.indices[name] = torch.randperm(features.shape[1])[:self.n]

        f = features.view(features.shape[0], -1)[:, self.indices]
        working_results.add(f)

        return working_results

    def finalise_result(self, name: str, result: Covariance) -> dict:
        try:
            covar = result.covariance()
            rank = matrix_rank(covar, hermitian=True,
                               rtol=self.threshold).cpu().item()

            rec = dict()
            rec['features_rank'] = rank
            rec['features_dim'] = self.features_dim

            norm = min(self.features_dim[name], covar.shape[0])
            rec['normalized_features_rank'] = rank / norm

            return rec
        except LinAlgError:
            return {}


class LayerWeightRankAnalyser(RankAnalyzer):
    def __init__(self, mode='aTa', n=8000, threshold=1e-3):
        super().__init__(mode, n, threshold)
        self.w_rank = None

    def process_batch_state(self, name: str, state: AnalyzerState,
                            working_results: Any | None):
        if (state.module is not None and hasattr(state.module, 'weight') and
                working_results is None):
            w = state.module.weight
            w = w.view(w.shape[0], -1)
            working_results = matrix_rank(w, hermitian=False,
                                          rtol=self.threshold).cpu().item()

        return working_results


class CovarianceSpectrumStatisticsAnalyser(RankAnalyzer):
    def __init__(self, n=8000, threshold=1e-3, taps=10):
        super().__init__(n=n, threshold=threshold)
        self.taps = taps

    def finalise_result(self, name, result) -> dict:
        stats = compute_cov_spectrum_stats(self.covar.covariance(),
                                           threshold=self.threshold,
                                           taps=self.taps)
        stats['features_dim'] = self.features_dim

        norm = min(self.features_dim[name], self.n)
        stats['normalized_features_rank'] = stats['features_rank'] / norm

        return stats
