"""Convert structured observations into tabular records and dataframes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product
from typing import Any

import torch

from .observations import Observation, normalise_observation_path
from .processing import flatten_observation, normalise_value


_COLUMNS = (
    "path",
    "global_step",
    "epoch",
    "batch",
    "event",
    "value",
    "metadata",
)
_EXPANDED_COLUMNS = (
    "path",
    "global_step",
    "epoch",
    "batch",
    "event",
    "value_index",
    "value",
    "metadata",
)


def observation_records(
        observations: Iterable[Observation],
        *,
        expand_values: bool = False,
) -> Iterator[dict[str, Any]]:
    """Convert observations into long-form tabular records.

    Nested mapping values are flattened into observation paths. Scalar tensors
    become Python scalars, while non-scalar tensors are detached and copied to
    CPU. With ``expand_values=True``, each tensor element becomes a separate
    record whose ``value_index`` contains its coordinate.

    Args:
        observations: The observations to convert.
        expand_values: Whether to expand tensor values into scalar rows.

    Yields:
        Dictionaries suitable for ``pandas.DataFrame.from_records``.
    """
    if not isinstance(expand_values, bool):
        raise TypeError("expand_values must be a bool")
    try:
        observations = iter(observations)
    except TypeError as error:
        raise TypeError("observations must be iterable") from error

    for observation in observations:
        if not isinstance(observation, Observation):
            raise TypeError(
                "observations must contain only Observation objects"
            )
        metadata = normalise_value(observation.metadata)
        for flattened in flatten_observation(observation):
            base = _observation_record(flattened, metadata)
            if expand_values:
                for index, value in _expanded_values(flattened.value):
                    yield {
                        **base,
                        "value_index": index,
                        "value": value,
                    }
            else:
                yield {
                    **base,
                    "value": _tabular_value(flattened.value),
                }


def observations_to_dataframe(
        observations: Iterable[Observation],
        *,
        expand_values: bool = False,
):
    """Convert observations into a long-form pandas dataframe.

    Pandas is imported lazily and is only required when this function is
    called. Install Torchwatcher with the ``dataframe`` extra to make it
    available.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "observations_to_dataframe requires pandas; install "
            "torchwatcher[dataframe]"
        ) from error

    columns = _EXPANDED_COLUMNS if expand_values else _COLUMNS
    records = list(observation_records(
        observations,
        expand_values=expand_values,
    ))
    return pd.DataFrame.from_records(records, columns=columns)


def _observation_record(
        observation: Observation,
        metadata: dict[str, Any],
) -> dict[str, Any]:
    context = observation.context
    return {
        "path": observation.path,
        "global_step": context.global_step,
        "epoch": context.epoch,
        "batch": context.batch,
        "event": context.event,
        "metadata": metadata,
    }


def _tabular_value(value: Any) -> Any:
    value = normalise_value(value)
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    return value


def _expanded_values(value: Any) -> Iterator[tuple[tuple[int, ...] | None, Any]]:
    value = normalise_value(value)
    if not torch.is_tensor(value):
        yield None, value
        return
    if value.ndim == 0:
        yield (), value.item()
        return
    if value.numel() == 0:
        yield None, value
        return

    for index in product(*(range(size) for size in value.shape)):
        yield index, value[index].item()


def path_starts_with(frame, prefix):
    """Return a subset of a dataframe whose paths start with a given prefix."""
    prefix = normalise_observation_path(prefix)
    return frame.loc[
        frame["path"].map(
            lambda path: path[:len(prefix)] == prefix
        )
    ]

def path_contains(frame, fragment):
    """Return a subset of a dataframe whose paths contain a given fragment."""
    fragment = normalise_observation_path(fragment)
    size = len(fragment)

    return frame.loc[
        frame["path"].map(
            lambda path: any(
                path[index:index + size] == fragment
                for index in range(len(path) - size + 1)
            )
        )
    ]

__all__ = [
    "observation_records",
    "observations_to_dataframe",
    "path_starts_with",
    "path_contains"
]
