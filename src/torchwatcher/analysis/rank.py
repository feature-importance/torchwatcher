import torch
from torch.linalg import LinAlgError, matrix_rank

from .analysis import Analyser, AnalyserState
from .running_stats import Covariance


def _promote_linalg_input(value: torch.Tensor) -> torch.Tensor:
    if value.dtype in {torch.float16, torch.bfloat16}:
        return value.float()
    return value


class _FeatureRows:
    """Keep either every feature row or a bounded uniform sample."""

    def __init__(self, limit: int):
        self.limit = limit
        self.batches = []
        self.sample = None
        self.priorities = None

    def add(self, features: torch.Tensor):
        if features.shape[0] == 0:
            return

        if self.limit <= 0:
            self.batches.append(features.clone())
            return

        priorities = torch.rand(features.shape[0], device=features.device)
        if self.sample is not None:
            features = torch.cat((self.sample, features))
            priorities = torch.cat((self.priorities, priorities))

        count = min(self.limit, features.shape[0])
        indices = priorities.topk(count).indices
        self.sample = features[indices]
        self.priorities = priorities[indices]

    def value(self) -> torch.Tensor:
        if self.limit > 0:
            return self.sample
        return torch.cat(self.batches)


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
    features = _promote_linalg_input(features)
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
    covariance = _promote_linalg_input(covariance)
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

        if mode not in {'aTa', 'aaT', 'a'}:
            raise ValueError('Unknown mode')

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            name_result: Covariance | _FeatureRows | None
                            ) -> Covariance | _FeatureRows:
        features = state.outputs.detach()
        f = features.reshape(features.shape[0], -1)
        f = _promote_linalg_input(f)

        if name not in self.features_dim:
            self.features_dim[name] = f.shape[1]
        elif self.features_dim[name] != f.shape[1]:
            raise ValueError(
                f"feature dimension for {name!r} changed from "
                f"{self.features_dim[name]} to {f.shape[1]}"
            )

        if self.mode == 'aTa':
            if name_result is None:
                name_result = Covariance()
            if name not in self.indices:
                self.indices[name] = (
                    torch.randperm(f.shape[1])[:self.n]
                    if self.n > 0 else None
                )
            if self.indices[name] is not None:
                f = f[:, self.indices[name].to(f.device)]
            name_result.add(f)
            return name_result

        if name_result is None:
            name_result = _FeatureRows(self.n)
        name_result.add(f)

        return name_result

    def finalise_result(
            self,
            name: str,
            result: Covariance | _FeatureRows,
    ) -> dict:
        try:
            if self.mode == 'aTa':
                covar = result.covariance()
                rank = matrix_rank(
                    covar,
                    hermitian=True,
                    rtol=self.threshold,
                ).cpu().item()
                sampled_dimension = covar.shape[0]
            else:
                features = result.value()
                rank = estimate_rank(
                    features,
                    n=-1,
                    mode=self.mode,
                    threshold=self.threshold,
                )
                sampled_dimension = features.shape[0]

            rec = dict()
            rec['features_rank'] = rank
            rec['features_dim'] = self.features_dim[name]

            norm = min(self.features_dim[name], sampled_dimension)
            rec['normalised_features_rank'] = rank / norm

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
                matrix = _promote_linalg_input(matrix)
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
        matrices = _promote_linalg_input(matrices)
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
        covariance = result.covariance()
        stats = compute_cov_spectrum_stats(covariance,
                                           threshold=self.threshold,
                                           taps=self.taps)
        stats['features_dim'] = self.features_dim[name]

        norm = covariance.shape[0]
        stats['normalised_features_rank'] = stats['features_rank'] / norm

        return stats
