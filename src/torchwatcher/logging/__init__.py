"""Structured logging and storage interfaces."""

from .observations import (
    Observation,
    ObservationContext,
    ObservationPath,
    ObservationPathInput,
    normalise_observation_path,
)
from .processing import (
    flatten_observation,
    normalise_observation,
    normalise_value,
)
from .sinks import CompositeSink, InMemorySink, ObservationSink

__all__ = [
    "CompositeSink",
    "InMemorySink",
    "Observation",
    "ObservationContext",
    "ObservationPath",
    "ObservationPathInput",
    "ObservationSink",
    "flatten_observation",
    "normalise_observation",
    "normalise_observation_path",
    "normalise_value",
]
