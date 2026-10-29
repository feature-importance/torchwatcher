"""Track measurements of selected parameters during training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torchbearer
from torch import nn
from torchbearer.callbacks import Callback

from torchwatcher.logging import (
    Observation,
    ObservationContext,
    ObservationSink,
    flatten_observation,
)
from torchwatcher.parameters import ParameterInfo, ParameterSelector
from torchwatcher.training._parameter_selection import ParameterSelection
from torchwatcher.training.scheduling import (
    EveryNSteps,
    ObservationSchedule,
    TrainingClock,
)


ParameterMeasurement = Callable[[ParameterInfo, Any], Any]
OptimizerStepMeasurement = Callable[[torch.Tensor, ParameterInfo, Any], Any]

_TRACKING_EVENTS = {
    "on_start": "start",
    "on_backward": "backward",
    "on_step_training": "step",
    "optimizer_step": "optimizer_step",
}


class ParameterTracker(Callback):
    """Record measurements of selected parameters during training.

    For ordinary events, measurements receive a :class:`ParameterInfo` and the
    Torchbearer state. For ``optimizer_step``, they receive the applied step,
    the :class:`ParameterInfo` and state. Measurements may return scalars,
    tensors or nested mappings. Nested mappings are flattened into separate
    observations by extending the observation path.

    Args:
        selector_or_parameters: A selector to resolve against the trial model,
            or an iterable of parameters registered on that model.
        measurements: A non-empty mapping from observation names to callables
            receiving ``(parameter_info, state)``. With
            ``event="optimizer_step"``, callables instead receive
            ``(step, parameter_info, state)``.
        sink: The destination for recorded observations.
        schedule: The observation schedule. By default, record every step,
            including step zero when tracking ``on_start``.
        event: The Torchbearer event at which measurements are made. Supported
            values are ``on_start``, ``on_backward`` and
            ``on_step_training``. Use ``optimizer_step`` to pair a pre-step
            snapshot with the post-step parameter and measure their difference.
        clock: A clock to share with other training observers. A new clock is
            created when omitted.
    """

    def __init__(
            self,
            selector_or_parameters: ParameterSelector | Iterable[nn.Parameter],
            measurements: Mapping[
                str,
                ParameterMeasurement | OptimizerStepMeasurement,
            ],
            sink: ObservationSink,
            *,
            schedule: ObservationSchedule | None = None,
            event: str = "on_backward",
            clock: TrainingClock | None = None,
    ) -> None:
        super().__init__()
        self._selection = ParameterSelection(selector_or_parameters)
        self.measurements = _validate_measurements(measurements)

        if not isinstance(sink, ObservationSink):
            raise TypeError("sink must implement ObservationSink")
        if schedule is None:
            schedule = EveryNSteps(1, include_zero=True)
        if not isinstance(schedule, ObservationSchedule):
            raise TypeError("schedule must implement ObservationSchedule")
        if event not in _TRACKING_EVENTS:
            events = ", ".join(sorted(_TRACKING_EVENTS))
            raise ValueError(f"event must be one of: {events}")
        if clock is not None and not isinstance(clock, TrainingClock):
            raise TypeError("clock must be a TrainingClock or None")

        self.sink = sink
        self.schedule = schedule
        self.event = event
        self.clock = clock or TrainingClock()
        self._selected: tuple[ParameterInfo, ...] | None = None
        self._pending_step: tuple[
            ObservationContext,
            dict[int, torch.Tensor],
        ] | None = None

    @property
    def selected(self) -> tuple[ParameterInfo, ...]:
        """Return the selection resolved against the current trial model."""
        if self._selected is None:
            raise RuntimeError(
                "selected parameters are not available before on_start"
            )
        return self._selected

    def on_start(self, state) -> None:
        self.clock.on_start(state)
        self._pending_step = None
        self._selected = self._selection.resolve(state[torchbearer.MODEL])
        if not self._selected:
            raise ValueError("parameter selection matched no parameters")
        if self.event == "on_start":
            self._record(state)

    def on_backward(self, state) -> None:
        if self.event == "optimizer_step":
            self._capture_optimizer_step(state)
        elif self.event == "on_backward":
            self._record(state)

    def on_step_training(self, state) -> None:
        self.clock.on_step_training(state)
        if self.event == "optimizer_step":
            self._record_optimizer_step(state)
        elif self.event == "on_step_training":
            self._record(state)

    def on_end(self, state) -> None:
        self._pending_step = None
        self.sink.flush()

    def state_dict(self) -> dict[str, Any]:
        """Return the tracker state required to resume global-step counting."""
        return {"clock": self.clock.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        """Restore tracker state and return this tracker."""
        self.clock.load_state_dict(state_dict.get("clock", {}))
        return self

    def _record(self, state) -> None:
        selected = self.selected
        context = self.clock.context(
            state,
            _TRACKING_EVENTS[self.event],
            next_step=self.event == "on_backward",
        )
        if not self.schedule.should_record(context):
            return

        self._write_measurements(
            context,
            lambda measurement, parameter_info: measurement(
                parameter_info,
                state,
            ),
        )

    def _capture_optimizer_step(self, state) -> None:
        selected = self.selected
        context = self.clock.context(
            state,
            _TRACKING_EVENTS[self.event],
            next_step=True,
        )
        self._pending_step = None
        if not self.schedule.should_record(context):
            return

        before = {
            id(info.parameter): info.parameter.detach().clone()
            for info in selected
        }
        self._pending_step = context, before

    def _record_optimizer_step(self, state) -> None:
        pending = self._pending_step
        self._pending_step = None
        if pending is None:
            return

        context, before = pending
        completed_context = self.clock.context(
            state,
            _TRACKING_EVENTS[self.event],
        )
        if completed_context.global_step != context.global_step:
            raise RuntimeError(
                "optimizer-step snapshot does not match the completed step"
            )

        steps = {
            id(info.parameter): (
                info.parameter.detach() - before[id(info.parameter)]
            )
            for info in self.selected
        }
        self._write_measurements(
            context,
            lambda measurement, parameter_info: measurement(
                steps[id(parameter_info.parameter)],
                parameter_info,
                state,
            ),
        )

    def _write_measurements(
            self,
            context: ObservationContext,
            evaluate: Callable[[Callable, ParameterInfo], Any],
    ) -> None:
        selected = self.selected

        observations = []
        for parameter_info in selected:
            metadata = {
                "aliases": parameter_info.aliases,
                "shape": tuple(parameter_info.parameter.shape),
                "dtype": str(parameter_info.parameter.dtype),
                "requires_grad": parameter_info.parameter.requires_grad,
            }
            for name, measurement in self.measurements:
                observation = Observation(
                    path=("parameters", parameter_info.full_name, name),
                    value=evaluate(measurement, parameter_info),
                    context=context,
                    metadata=metadata,
                )
                observations.extend(flatten_observation(observation))

        self.sink.write(observations)


def _validate_measurements(
        measurements: Mapping[
            str,
            ParameterMeasurement | OptimizerStepMeasurement,
        ],
) -> tuple[
    tuple[str, ParameterMeasurement | OptimizerStepMeasurement],
    ...,
]:
    if not isinstance(measurements, Mapping):
        raise TypeError("measurements must be a mapping")
    if not measurements:
        raise ValueError("at least one measurement is required")

    items = tuple(measurements.items())
    if any(not isinstance(name, str) for name, _ in items):
        raise TypeError("measurement names must be strings")
    if any(not name for name, _ in items):
        raise ValueError("measurement names must be non-empty")
    if any(not callable(measurement) for _, measurement in items):
        raise TypeError("measurements must be callable")
    return items


__all__ = [
    "OptimizerStepMeasurement",
    "ParameterMeasurement",
    "ParameterTracker",
]
