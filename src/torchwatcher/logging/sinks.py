"""Storage contracts for Torchwatcher observations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from .observations import Observation
from .processing import normalise_observation


@runtime_checkable
class ObservationSink(Protocol):
    """A destination for batches of structured observations."""

    def write(self, observations: Iterable[Observation]) -> None:
        """Write a batch of observations."""
        ...

    def flush(self) -> None:
        """Flush buffered observations to the underlying destination."""
        ...

    def close(self) -> None:
        """Flush and release resources held by the sink."""
        ...


class InMemorySink:
    """Retain observations in memory.

    Args:
        normalise: Whether to snapshot tensors onto CPU as they are written.
    """

    def __init__(self, *, normalise: bool = True) -> None:
        if not isinstance(normalise, bool):
            raise TypeError("normalise must be a bool")
        self.normalise = normalise
        self.observations: list[Observation] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this sink has been closed."""
        return self._closed

    def write(self, observations: Iterable[Observation]) -> None:
        """Append observations to this sink."""
        if self.closed:
            raise RuntimeError("cannot write to a closed sink")

        batch = _observation_batch(observations)
        if self.normalise:
            batch = tuple(normalise_observation(item) for item in batch)
        self.observations.extend(batch)

    def flush(self) -> None:
        """Flush buffered observations; a no-op for an in-memory sink."""

    def close(self) -> None:
        """Close this sink. Existing observations remain available."""
        self._closed = True


class CompositeSink:
    """Write each batch of observations to multiple sinks."""

    def __init__(self, *sinks: ObservationSink) -> None:
        if not sinks:
            raise ValueError("at least one sink is required")
        if any(not isinstance(sink, ObservationSink) for sink in sinks):
            raise TypeError("sinks must implement ObservationSink")
        self.sinks = tuple(sinks)
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this sink has been closed."""
        return self._closed

    def write(self, observations: Iterable[Observation]) -> None:
        """Write the observations to every child sink."""
        if self.closed:
            raise RuntimeError("cannot write to a closed sink")

        batch = _observation_batch(observations)
        for sink in self.sinks:
            sink.write(batch)

    def flush(self) -> None:
        """Flush every child sink."""
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        """Close every child sink."""
        if self.closed:
            return
        for sink in self.sinks:
            sink.close()
        self._closed = True


def _observation_batch(
        observations: Iterable[Observation]
) -> tuple[Observation, ...]:
    try:
        batch = tuple(observations)
    except TypeError as error:
        raise TypeError("observations must be iterable") from error
    if any(not isinstance(item, Observation) for item in batch):
        raise TypeError("observations must contain only Observation objects")
    return batch


__all__ = [
    "CompositeSink",
    "InMemorySink",
    "ObservationSink",
]
