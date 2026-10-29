from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torchbearer
from model_utilities.utils.callbacks import at_training_iterations
from torch import nn
from torchbearer.callbacks import Callback

from torchwatcher.analysis.observations import analyser_observations
from torchwatcher.logging import (
    ObservationContext,
    ObservationPathInput,
    ObservationSink,
)
from torchwatcher.parameters import (
    ParameterInfo,
    ParameterSelector,
)
from torchwatcher.training._parameter_selection import ParameterSelection
from torchwatcher.training.scheduling import (
    GLOBAL_STEP,
    EveryNSteps,
    ObservationSchedule,
    TrainingClock,
    as_iteration_predicate,
)

_PARAMETER_CALLBACK_EVENTS = {
    "on_start",
    "on_backward",
    "on_step_training",
}


class SelectedParametersCallback(Callback):
    """Apply an action to selected model parameters during training.

    The action receives a :class:`ParameterInfo` object and the Torchbearer
    state. It is called once for each selected parameter, in selection order.
    The action is responsible for making any mutation explicitly; its return
    value is ignored.

    Args:
        selector_or_parameters: A selector to resolve against the trial's model,
            or an iterable of parameters registered on that model.
        action: A callable receiving ``(parameter_info, state)``.
        event: The Torchbearer event at which to call the action. Supported
            values are ``on_start``, ``on_backward`` and ``on_step_training``.
    """

    def __init__(
        self,
        selector_or_parameters: ParameterSelector | Iterable[nn.Parameter],
        action: Callable[[ParameterInfo, Any], Any],
        *,
        event: str = "on_backward",
    ):
        super().__init__()
        if not callable(action):
            raise TypeError("action must be callable")
        if event not in _PARAMETER_CALLBACK_EVENTS:
            events = ", ".join(sorted(_PARAMETER_CALLBACK_EVENTS))
            raise ValueError(f"event must be one of: {events}")

        self.action = action
        self.event = event
        self._selection = ParameterSelection(selector_or_parameters)
        self._selected: tuple[ParameterInfo, ...] | None = None

    @property
    def selected(self) -> tuple[ParameterInfo, ...]:
        """Return the selection resolved against the current trial model."""
        if self._selected is None:
            raise RuntimeError(
                "selected parameters are not available before on_start"
            )
        return self._selected

    def on_start(self, state):
        self._selected = self._selection.resolve(state[torchbearer.MODEL])
        if not self._selected:
            raise ValueError("parameter selection matched no parameters")
        if self.event == "on_start":
            self._apply(state)

    def on_backward(self, state):
        if self.event == "on_backward":
            self._apply(state)

    def on_step_training(self, state):
        if self.event == "on_step_training":
            self._apply(state)

    def _apply(self, state):
        if self._selected is None:
            raise RuntimeError(
                "SelectedParametersCallback received an event before on_start"
            )
        for parameter_info in self._selected:
            self.action(parameter_info, state)


class AnalyserEvaluation(Callback):
    """Evaluate an analyser over a loader during Torchbearer training.

    Use ``callbacks`` to attach the evaluation on a fixed period or any
    ``model_utilities.utils.callbacks.at_training_iterations`` schedule. By
    default, snapshot passes run with gradient computation disabled. Set
    ``compute_gradients=True`` and optionally provide ``backward`` for analysers
    that need a backward pass. Results can be retained in ``records``, written
    to an observation sink, or both.

    Args:
        analyser: The analyser to evaluate.
        loader: The data loader used for each evaluation.
        prepare_inputs: An optional function converting a batch and device to
            model inputs.
        compute_gradients: Whether evaluation should track gradients.
        backward: An optional function performing or returning a backward
            computation for each batch.
        records: A list in which legacy result records are retained.
        sink: An optional destination for flattened analyser observations.
        path: The root observation path. By default, use the analyser class
            name below ``analysers``.
        retain_records: Whether to retain results in ``records`` in addition
            to writing them to the sink.
    """

    def __init__(
        self,
        analyser,
        loader,
        *,
        prepare_inputs: Callable[[Any, torch.device], Any] | None = None,
        compute_gradients: bool = False,
        backward: Callable[[Any, Any], Any] | None = None,
        records: list[dict] | None = None,
        sink: ObservationSink | None = None,
        path: ObservationPathInput | None = None,
        retain_records: bool = True,
    ):
        super().__init__()
        if backward is not None and not compute_gradients:
            raise ValueError(
                "compute_gradients must be True when backward is provided"
            )
        if sink is not None and not isinstance(sink, ObservationSink):
            raise TypeError("sink must implement ObservationSink or be None")
        if not isinstance(retain_records, bool):
            raise TypeError("retain_records must be a bool")
        if not retain_records and records is not None:
            raise ValueError(
                "records cannot be provided when retain_records is False"
            )

        self.analyser = analyser
        self.loader = loader
        self.prepare_inputs = prepare_inputs or _default_inputs
        self.compute_gradients = compute_gradients
        self.backward = backward
        self.records = [] if records is None else records
        self.sink = sink
        self.path = path
        self.retain_records = retain_records
        self._last_recorded_step: int | None = None

    def callbacks(
        self,
        *,
        every_n_batches: int | None = None,
        schedule=None,
        include_start: bool = True,
        include_end: bool = True,
        clock: TrainingClock | None = None,
    ) -> list[Callback]:
        if every_n_batches is None and schedule is None:
            raise ValueError("every_n_batches or schedule must be provided")
        if every_n_batches is not None and schedule is not None:
            raise ValueError("only one of every_n_batches or schedule can be provided")
        if every_n_batches is not None and every_n_batches <= 0:
            raise ValueError("every_n_batches must be a positive integer")

        if every_n_batches is not None:
            schedule = EveryNSteps(every_n_batches)
        if clock is not None and not isinstance(clock, TrainingClock):
            raise TypeError("clock must be a TrainingClock or None")

        callbacks = []
        if include_start:
            callbacks.append(_RecordAnalyserEvaluation(self, "start"))

        callbacks.append(clock or TrainingClock())
        if isinstance(schedule, ObservationSchedule):
            schedule = as_iteration_predicate(schedule)
        callbacks.append(at_training_iterations(schedule)(self))

        if include_end:
            callbacks.append(_RecordAnalyserEvaluation(self, "end"))
        if self.sink is not None:
            callbacks.append(_FlushAnalyserEvaluationSink(self))

        return callbacks

    def on_step_training(self, state):
        self.record(
            state,
            event="batch",
            global_step=state.get(GLOBAL_STEP),
        )

    def record(self, state, *, event, global_step=None) -> dict:
        model = state[torchbearer.MODEL]
        device = next(model.parameters()).device
        was_training = model.training
        was_enabled = self.analyser.enabled
        saved_gradients = (
            _save_gradients(model) if self.compute_gradients else None
        )

        self.analyser.reset()
        self.analyser.enabled = True
        model.eval()

        try:
            with torch.set_grad_enabled(self.compute_gradients):
                for batch in self.loader:
                    if self.compute_gradients:
                        model.zero_grad(set_to_none=True)
                    inputs = self.prepare_inputs(batch, device)
                    batch_context = getattr(self.analyser, "batch", nullcontext)
                    with batch_context():
                        outputs = _forward(model, inputs)
                        if self.backward is not None:
                            _backward(self.backward, outputs, batch)
            result = self.analyser.to_dict()
        finally:
            if self.compute_gradients:
                model.zero_grad(set_to_none=True)
                _restore_gradients(model, saved_gradients)
            self.analyser.enabled = was_enabled
            model.train(was_training)

        if global_step is None:
            global_step = state.get(GLOBAL_STEP, self._last_recorded_step)

        context = ObservationContext(
            global_step=global_step,
            epoch=_epoch(state, event),
            batch=_batch(state, event),
            event=event,
        )
        record = {
            "event": event,
            "epoch": context.epoch,
            "batch": context.batch,
            "global_step": global_step,
            "result": result,
        }
        if self.retain_records:
            self.records.append(record)
        if self.sink is not None:
            self.sink.write(analyser_observations(
                self.analyser,
                path=self.path,
                context=context,
                result=result,
            ))
        self._last_recorded_step = global_step
        return record


class _RecordAnalyserEvaluation(Callback):
    def __init__(self, evaluation: AnalyserEvaluation, event: str):
        super().__init__()
        self.evaluation = evaluation
        self.event = event

    def on_start(self, state):
        if self.event == "start":
            self.evaluation.record(state, event="start", global_step=0)

    def on_end(self, state):
        if self.event == "end":
            global_step = state.get(GLOBAL_STEP)
            if self.evaluation._last_recorded_step != global_step:
                self.evaluation.record(
                    state,
                    event="end",
                    global_step=global_step,
                )


class _FlushAnalyserEvaluationSink(Callback):
    def __init__(self, evaluation: AnalyserEvaluation):
        super().__init__()
        self.evaluation = evaluation

    def on_end(self, state):
        self.evaluation.sink.flush()


def _default_inputs(batch, device):
    inputs = batch[0] if isinstance(batch, Sequence) else batch
    return _to_device(inputs, device)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _forward(model, inputs):
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


def _backward(backward, outputs, batch):
    result = backward(outputs, batch)
    if torch.is_tensor(result):
        result.backward()


def _save_gradients(model):
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]


def _restore_gradients(model, gradients):
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = gradient


def _epoch(state, event):
    if event == "start":
        return 0
    return state.get(torchbearer.EPOCH, 0) + 1


def _batch(state, event):
    if event == "start":
        return 0
    if event == "end":
        return None
    return state.get(torchbearer.BATCH, 0) + 1
