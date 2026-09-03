from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .analysis import (
    AnalysisPoint,
    AnalysisRelation,
    AnalyserState,
    RelationalAnalyser,
)


@dataclass(frozen=True)
class CKAResult:
    """A labelled matrix of pairwise CKA values."""

    row_names: tuple[str, ...]
    column_names: tuple[str, ...]
    values: torch.Tensor

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            row_name: {
                column_name: self.values[row, column].item()
                for column, column_name in enumerate(self.column_names)
            }
            for row, row_name in enumerate(self.row_names)
        }


@dataclass(frozen=True)
class _CKAAccumulator:
    numerator: torch.Tensor
    left_self_similarity: torch.Tensor
    right_self_similarity: torch.Tensor
    batches: int = 0

    def updated(
        self,
        numerator: torch.Tensor,
        left_self_similarity: torch.Tensor,
        right_self_similarity: torch.Tensor,
    ) -> "_CKAAccumulator":
        return _CKAAccumulator(
            numerator=self.numerator + numerator,
            left_self_similarity=(
                self.left_self_similarity + left_self_similarity
            ),
            right_self_similarity=(
                self.right_self_similarity + right_self_similarity
            ),
            batches=self.batches + 1,
        )

    def score(self, eps: float) -> torch.Tensor:
        denominator = (
            self.left_self_similarity * self.right_self_similarity
        ).sqrt()
        if not torch.isfinite(denominator) or denominator <= eps:
            return torch.full_like(denominator, torch.nan)
        return self.numerator / denominator


def _biased_hsic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    n = left.shape[0]
    left = (
        left
        - left.mean(dim=0, keepdim=True)
        - left.mean(dim=1, keepdim=True)
        + left.mean()
    )
    right = (
        right
        - right.mean(dim=0, keepdim=True)
        - right.mean(dim=1, keepdim=True)
        + right.mean()
    )
    normalizer = max((n - 1) ** 2, 1)
    return (left * right).sum() / normalizer


def _unbiased_hsic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    n = left.shape[0]
    if n < 4:
        raise ValueError("debiased minibatch CKA requires at least four samples")

    left = left.clone()
    right = right.clone()
    left.fill_diagonal_(0)
    right.fill_diagonal_(0)

    left_rows = left.sum(dim=1)
    right_rows = right.sum(dim=1)
    value = (
        (left * right).sum()
        + left_rows.sum() * right_rows.sum() / ((n - 1) * (n - 2))
        - 2 * torch.dot(left_rows, right_rows) / (n - 2)
    )
    return value / (n * (n - 3))


class LinearCKAAnalyser(RelationalAnalyser[_CKAAccumulator]):
    """Accumulate minibatch linear centred kernel alignment between layers.

    Activations are flattened to ``(batch, features)`` and converted to linear
    Gram matrices. Only the Gram matrices for the current batch and three
    scalar totals per layer pair are retained.

    With one watched source, the analyser compares its layers with one another.
    With two sources, it compares every layer in the first source with every
    layer in the second. Use ``comparisons`` to select relations explicitly.
    """

    def __init__(
        self,
        comparisons: Sequence[tuple[str, str]] | None = None,
        *,
        debiased: bool = False,
        accumulation_device: torch.device | str = "cpu",
        accumulation_dtype: torch.dtype = torch.float64,
        eps: float = 1e-12,
    ):
        super().__init__(comparisons=comparisons)
        if not accumulation_dtype.is_floating_point:
            raise ValueError("accumulation_dtype must be a floating-point dtype")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.debiased = debiased
        self.accumulation_device = torch.device(accumulation_device)
        self.accumulation_dtype = accumulation_dtype
        self.eps = eps
        self._gram_cache_key = object()

    def _gram(self, state: AnalyserState) -> torch.Tensor:
        cached = state.extras.get(self._gram_cache_key)
        if cached is not None:
            return cached

        activation = state.outputs
        if not torch.is_tensor(activation):
            raise TypeError(
                f"layer {state.name!r} produced {type(activation).__name__}; "
                "use watch(source, transform=...) to select a tensor"
            )
        if activation.ndim == 0:
            raise ValueError(
                f"layer {state.name!r} produced a scalar without a batch dimension"
            )
        if activation.shape[0] == 0:
            raise ValueError("CKA cannot process an empty batch")
        if activation.is_complex():
            raise TypeError("complex-valued activations are not supported")

        features = activation.detach().reshape(activation.shape[0], -1)
        if not features.is_floating_point():
            features = features.float()
        gram = features @ features.t()
        gram = gram.to(
            device=self.accumulation_device,
            dtype=self.accumulation_dtype,
        )
        state.extras[self._gram_cache_key] = gram
        return gram

    def prepare_state(self, point: AnalysisPoint, state: AnalyserState):
        self._gram(state)
        # The Gram matrix is sufficient for minibatch linear CKA. We release
        # the activations immediately to save memory because otherwise we might
        # end up holding them for multiple models at once.
        state._outputs = None

    def validate_batch_states(self, states, comparisons):
        required_sources = {
            source for comparison in comparisons for source in comparison
        }
        batch_sizes: dict[str, int] = {}
        for source in required_sources:
            sizes = {
                self._gram(state).shape[0]
                for state in states[source].values()
            }
            if len(sizes) != 1:
                raise ValueError(
                    f"layers from source {source!r} have inconsistent batch sizes"
                )
            batch_sizes[source] = sizes.pop()
            if self.debiased and batch_sizes[source] < 4:
                raise ValueError(
                    "debiased minibatch CKA requires at least four samples"
                )

        for left, right in comparisons:
            if batch_sizes[left] != batch_sizes[right]:
                raise ValueError(
                    f"sources {left!r} and {right!r} have different batch sizes: "
                    f"{batch_sizes[left]} and {batch_sizes[right]}"
                )

    def process_batch_relation(
        self,
        relation: AnalysisRelation,
        states: tuple[AnalyserState, AnalyserState],
        working_results: _CKAAccumulator | None,
    ) -> _CKAAccumulator:
        left = self._gram(states[0])
        right = self._gram(states[1])
        hsic = _unbiased_hsic if self.debiased else _biased_hsic

        numerator = hsic(left, right)
        left_self_similarity = hsic(left, left)
        right_self_similarity = hsic(right, right)
        if working_results is None:
            zero = torch.zeros_like(numerator)
            working_results = _CKAAccumulator(zero, zero, zero)
        return working_results.updated(
            numerator,
            left_self_similarity,
            right_self_similarity,
        )

    def finalise_result(
        self,
        name: AnalysisRelation,
        result: _CKAAccumulator,
    ) -> float:
        return result.score(self.eps).item()

    def result(
        self,
        left_source: str | None = None,
        right_source: str | None = None,
    ) -> CKAResult:
        """Return one source comparison as a labelled matrix."""
        if left_source is None and right_source is None:
            comparisons = self.comparisons
            if len(comparisons) != 1:
                raise ValueError(
                    "left_source and right_source are required when multiple "
                    "comparisons are configured"
                )
            left_source, right_source = comparisons[0]
        elif left_source is None or right_source is None:
            raise ValueError("left_source and right_source must be provided together")

        comparison = (left_source, right_source)
        if comparison not in self.comparisons:
            raise KeyError(f"comparison {comparison!r} is not configured")
        if left_source not in self._schemas or right_source not in self._schemas:
            raise RuntimeError("no complete CKA batches have been collected")

        left_names = self._schemas[left_source]
        right_names = self._schemas[right_source]
        values = torch.empty(
            (len(left_names), len(right_names)),
            dtype=self.accumulation_dtype,
            device=self.accumulation_device,
        )
        for row, left_name in enumerate(left_names):
            for column, right_name in enumerate(right_names):
                relation = AnalysisRelation(
                    left=AnalysisPoint(left_source, left_name),
                    right=AnalysisPoint(right_source, right_name),
                )
                values[row, column] = self.working_results[relation].score(self.eps)

        return CKAResult(
            row_names=tuple(
                AnalysisPoint(left_source, name).qualified_name
                for name in left_names
            ),
            column_names=tuple(
                AnalysisPoint(right_source, name).qualified_name
                for name in right_names
            ),
            values=values.detach().cpu(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            f"{left}:{right}": self.result(left, right).to_dict()
            for left, right in self.comparisons
            if left in self._schemas and right in self._schemas
        }
