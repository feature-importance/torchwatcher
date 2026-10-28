"""Utilities for preparing observations for storage."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from .observations import Observation


def flatten_observation(observation: Observation) -> Iterator[Observation]:
    """Flatten a nested mapping value into observations with extended paths.

    Mapping keys are converted to strings. Empty mappings and all non-mapping
    values are treated as leaves. Context and metadata are preserved on every
    resulting observation.
    """
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")

    value = observation.value
    if not isinstance(value, Mapping) or not value:
        yield observation
        return

    items = [
        (str(key), nested_value)
        for key, nested_value in value.items()
    ]
    components: set[str] = set()
    for component, _ in items:
        if not component:
            raise ValueError("mapping keys must produce non-empty path components")
        if component in components:
            raise ValueError(
                f"mapping keys produce duplicate path component {component!r}"
            )
        components.add(component)

    for component, nested_value in items:
        nested = Observation(
            path=(*observation.path, component),
            value=nested_value,
            context=observation.context,
            metadata=observation.metadata,
        )
        yield from flatten_observation(nested)


def normalise_value(value: Any) -> Any:
    """Detach and copy tensors to CPU, recursively handling basic containers."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            key: normalise_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(normalise_value(item) for item in value)
    if isinstance(value, list):
        return [normalise_value(item) for item in value]
    return value


def normalise_observation(observation: Observation) -> Observation:
    """Create a storage-safe snapshot of an observation.

    Tensors in the value or metadata are detached from autograd, copied to CPU,
    and cloned so later mutation of the source cannot change the observation.
    """
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")

    return Observation(
        path=observation.path,
        value=normalise_value(observation.value),
        context=observation.context,
        metadata=normalise_value(observation.metadata),
    )


__all__ = [
    "flatten_observation",
    "normalise_observation",
    "normalise_value",
]
