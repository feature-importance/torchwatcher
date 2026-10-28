"""Shared training clocks and observation schedules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torchbearer
from torchbearer.callbacks import Callback

from torchwatcher.logging import ObservationContext


GLOBAL_STEP = torchbearer.StateKey("torchwatcher.global_step")
"""Torchbearer state key containing the number of completed training steps."""

_CLOCK_POSITION = torchbearer.StateKey("torchwatcher.clock_position")


class TrainingClock(Callback):
    """Publish a shared global training step in Torchbearer state.

    Multiple clock callbacks are safe: the epoch and batch position stored in
    state ensures a training batch is counted only once.

    Args:
        initial_step: The initial number of completed training steps.
    """

    def __init__(self, initial_step: int = 0) -> None:
        super().__init__()
        _validate_step(initial_step, "initial_step")
        self.global_step = initial_step

    def on_start(self, state) -> None:
        existing = state.get(GLOBAL_STEP, 0)
        _validate_step(existing, "state global step")
        self.global_step = max(self.global_step, existing)
        state[GLOBAL_STEP] = self.global_step
        state[_CLOCK_POSITION] = None

    def on_step_training(self, state) -> None:
        position = (
            state.get(torchbearer.EPOCH),
            state.get(torchbearer.BATCH),
        )
        if state.get(_CLOCK_POSITION) != position:
            existing = state.get(GLOBAL_STEP, self.global_step)
            _validate_step(existing, "state global step")
            self.global_step = max(self.global_step, existing) + 1
            state[GLOBAL_STEP] = self.global_step
            state[_CLOCK_POSITION] = position
        else:
            self.global_step = state[GLOBAL_STEP]

    def context(
            self,
            state: Mapping,
            event: str,
            *,
            next_step: bool = False,
    ) -> ObservationContext:
        """Return an observation context for the current training position."""
        return observation_context(state, event, next_step=next_step)

    def state_dict(self) -> dict[str, int]:
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        global_step = state_dict.get("global_step", 0)
        _validate_step(global_step, "global_step")
        self.global_step = global_step
        return self


def observation_context(
        state: Mapping,
        event: str,
        *,
        next_step: bool = False,
) -> ObservationContext:
    """Construct an observation context from Torchbearer state.

    Args:
        state: The current Torchbearer state.
        event: The lifecycle event associated with the observation.
        next_step: Associate the observation with the upcoming optimizer step
            rather than the number of steps already completed. This is useful
            for gradients and other pre-step observations.
    """
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    if not isinstance(next_step, bool):
        raise TypeError("next_step must be a bool")

    global_step = state.get(GLOBAL_STEP)
    if global_step is not None:
        _validate_step(global_step, "state global step")
        if next_step:
            global_step += 1

    if event == "start":
        epoch = 0
        batch = 0
    else:
        epoch = _one_based_index(state.get(torchbearer.EPOCH), "epoch")
        batch = (
            None
            if event == "end"
            else _one_based_index(state.get(torchbearer.BATCH), "batch")
        )

    return ObservationContext(
        global_step=global_step,
        epoch=epoch,
        batch=batch,
        event=event,
    )


@runtime_checkable
class ObservationSchedule(Protocol):
    """Determine whether an observation should be recorded in a context."""

    def should_record(self, context: ObservationContext) -> bool:
        """Return whether an observation should be recorded."""
        ...


@dataclass(frozen=True)
class EveryNSteps:
    """Record every ``period`` global steps."""

    period: int
    include_zero: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.period, int) or isinstance(self.period, bool):
            raise TypeError("period must be an int")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if not isinstance(self.include_zero, bool):
            raise TypeError("include_zero must be a bool")

    def should_record(self, context: ObservationContext) -> bool:
        step = context.global_step
        if step is None:
            return False
        if step == 0 and not self.include_zero:
            return False
        return step % self.period == 0


@dataclass(frozen=True, init=False)
class AtSteps:
    """Record at explicitly selected global steps."""

    steps: frozenset[int]

    def __init__(self, *steps: int | Iterable[int]) -> None:
        if len(steps) == 1 and not isinstance(steps[0], int):
            values = tuple(steps[0])
        else:
            values = tuple(steps)
        if not values:
            raise ValueError("at least one step is required")
        for step in values:
            _validate_step(step, "step")
        object.__setattr__(self, "steps", frozenset(values))

    def should_record(self, context: ObservationContext) -> bool:
        return context.global_step in self.steps


@dataclass(frozen=True)
class PredicateSchedule:
    """Adapt an observation-context predicate to a schedule."""

    predicate: Callable[[ObservationContext], bool]

    def __post_init__(self) -> None:
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")

    def should_record(self, context: ObservationContext) -> bool:
        return bool(self.predicate(context))


@dataclass(frozen=True, init=False)
class PiecewiseSchedule:
    """Use different recording periods across ranges of global steps.

    Each phase is ``(last_step, period)`` and bounds are inclusive. The final
    bound can be ``None`` to continue indefinitely. Periods are aligned to the
    global step rather than the beginning of each phase.
    """

    phases: tuple[tuple[int | None, int], ...]
    include_zero: bool

    def __init__(
            self,
            phases: Iterable[tuple[int | None, int]],
            *,
            include_zero: bool = False,
    ) -> None:
        phases = tuple(phases)
        if not phases:
            raise ValueError("at least one phase is required")
        if not isinstance(include_zero, bool):
            raise TypeError("include_zero must be a bool")

        previous_bound = -1
        for index, (bound, period) in enumerate(phases):
            if bound is None:
                if index != len(phases) - 1:
                    raise ValueError("only the final phase can have no bound")
            else:
                _validate_step(bound, "phase bound")
                if bound <= previous_bound:
                    raise ValueError("phase bounds must be strictly increasing")
                previous_bound = bound
            if not isinstance(period, int) or isinstance(period, bool):
                raise TypeError("phase periods must be ints")
            if period <= 0:
                raise ValueError("phase periods must be positive")

        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "include_zero", include_zero)

    def should_record(self, context: ObservationContext) -> bool:
        step = context.global_step
        if step is None:
            return False
        if step == 0 and not self.include_zero:
            return False
        for bound, period in self.phases:
            if bound is None or step <= bound:
                return step % period == 0
        return False


def as_iteration_predicate(
        schedule: ObservationSchedule,
        *,
        event: str = "batch",
) -> Callable[[int, Mapping], bool]:
    """Adapt an observation schedule for model-utilities callback scheduling.

    The iteration index is accepted for compatibility with
    ``at_training_iterations``. The shared clock in ``state`` remains the
    source of the observation's global step.
    """
    if not isinstance(schedule, ObservationSchedule):
        raise TypeError("schedule must implement ObservationSchedule")
    if not isinstance(event, str):
        raise TypeError("event must be a str")
    if not event:
        raise ValueError("event must be non-empty")

    def predicate(_, state) -> bool:
        context = observation_context(state, event=event)
        return bool(schedule.should_record(context))

    return predicate


def _one_based_index(value, name: str) -> int | None:
    if value is None:
        return None
    _validate_step(value, name)
    return value + 1


def _validate_step(value, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


__all__ = [
    "AtSteps",
    "EveryNSteps",
    "GLOBAL_STEP",
    "ObservationSchedule",
    "PiecewiseSchedule",
    "PredicateSchedule",
    "TrainingClock",
    "as_iteration_predicate",
    "observation_context",
]
