"""Structured observations produced by Torchwatcher components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ObservationContext:
    """Identify where an observation was made during training or evaluation.

    All fields are optional so observations made outside a training loop can
    use an empty context.

    Args:
        global_step: The zero-based global training step.
        epoch: The epoch associated with the observation.
        batch: The batch associated with the observation.
        event: The lifecycle event at which the observation was made.
    """

    global_step: int | None = None
    epoch: int | None = None
    batch: int | None = None
    event: str | None = None

    def __post_init__(self) -> None:
        for name in ("global_step", "epoch", "batch"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
            ):
                raise TypeError(f"{name} must be an int or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.event is not None and not isinstance(self.event, str):
            raise TypeError("event must be a str or None")
        if self.event == "":
            raise ValueError("event must be non-empty")


@dataclass(frozen=True, eq=False)
class Observation:
    """A value recorded with a stable hierarchical path.

    The value is intentionally unconstrained. Producers may emit scalars,
    tensors or structured results; sinks decide which values they support and
    how those values are serialized.

    Args:
        path: A non-empty tuple identifying the observation series.
        value: The observed value.
        context: Where the observation was made.
        metadata: Additional descriptive information about the series.
    """

    path: tuple[str, ...]
    value: Any
    context: ObservationContext = field(default_factory=ObservationContext)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple):
            raise TypeError("path must be a tuple of strings")
        if not self.path:
            raise ValueError("path must not be empty")
        if any(not isinstance(component, str) for component in self.path):
            raise TypeError("path components must be strings")
        if any(not component for component in self.path):
            raise ValueError("path components must be non-empty")
        if not isinstance(self.context, ObservationContext):
            raise TypeError("context must be an ObservationContext")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


__all__ = [
    "Observation",
    "ObservationContext",
]
