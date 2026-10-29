"""Convert analyser results into structured observations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from torchwatcher.logging import (
    Observation,
    ObservationContext,
    ObservationPathInput,
    flatten_observation,
)


_RESULT_UNSET = object()


def analyser_observations(
        analyser,
        *,
        path: ObservationPathInput | None = None,
        context: ObservationContext | None = None,
        metadata: Mapping[str, Any] | None = None,
        result: Any = _RESULT_UNSET,
) -> Iterator[Observation]:
    """Convert an analyser result into flattened observations.

    Args:
        analyser: An analyser-like object with a callable ``to_dict`` method.
        path: Root path for the analyser result. By default, use
            ``("analysers", type(analyser).__name__)``.
        context: The context associated with the result.
        metadata: Additional metadata copied to each observation.
        result: An already computed analyser result. When omitted,
            ``analyser.to_dict()`` is called.

    Yields:
        One observation for each leaf in the analyser result.
    """
    to_dict = getattr(analyser, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("analyser must provide a callable to_dict method")
    if path is None:
        path = ("analysers", type(analyser).__name__)
    if context is None:
        context = ObservationContext()
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    if result is _RESULT_UNSET:
        result = to_dict()

    analyser_type = type(analyser)
    combined_metadata = {
        "analyser_type": (
            f"{analyser_type.__module__}.{analyser_type.__qualname__}"
        ),
    }
    combined_metadata.update(metadata)
    observation = Observation(
        path=path,
        value=result,
        context=context,
        metadata=combined_metadata,
    )
    yield from flatten_observation(observation)


__all__ = ["analyser_observations"]
