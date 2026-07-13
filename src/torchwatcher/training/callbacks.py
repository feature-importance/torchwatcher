from collections.abc import Callable, Sequence
from typing import Any

import torch
import torchbearer
from model_utilities.utils.callbacks import at_training_iterations
from torchbearer.callbacks import Callback

_GLOBAL_STEP = torchbearer.StateKey("torchwatcher.global_step")


class AnalyzerEvaluation(Callback):
    """Evaluate an analyzer over a loader during Torchbearer training.

    Use ``callbacks`` to attach the evaluation on a fixed period or any
    ``model_utilities.utils.callbacks.at_training_iterations`` schedule.
    """

    def __init__(
        self,
        analyzer,
        loader,
        *,
        prepare_inputs: Callable[[Any, torch.device], Any] | None = None,
        records: list[dict] | None = None,
    ):
        super().__init__()
        self.analyzer = analyzer
        self.loader = loader
        self.prepare_inputs = prepare_inputs or _default_inputs
        self.records = [] if records is None else records
        self._last_recorded_step: int | None = None

    def callbacks(
        self,
        *,
        every_n_batches: int | None = None,
        schedule=None,
        include_start: bool = True,
        include_end: bool = True,
    ) -> list[Callback]:
        if every_n_batches is None and schedule is None:
            raise ValueError("every_n_batches or schedule must be provided")
        if every_n_batches is not None and schedule is not None:
            raise ValueError("only one of every_n_batches or schedule can be provided")
        if every_n_batches is not None and every_n_batches <= 0:
            raise ValueError("every_n_batches must be a positive integer")

        if every_n_batches is not None:
            schedule = lambda index: (index + 1) % every_n_batches == 0

        callbacks = []
        if include_start:
            callbacks.append(_RecordAnalyzerEvaluation(self, "start"))

        callbacks.append(_TrainingIterationCounter())
        callbacks.append(at_training_iterations(schedule)(self))

        if include_end:
            callbacks.append(_RecordAnalyzerEvaluation(self, "end"))

        return callbacks

    def on_step_training(self, state):
        self.record(
            state,
            event="batch",
            global_step=state.get(_GLOBAL_STEP),
        )

    def record(self, state, *, event, global_step=None) -> dict:
        model = state[torchbearer.MODEL]
        device = next(model.parameters()).device
        was_training = model.training
        was_enabled = self.analyzer.enabled

        self.analyzer.reset()
        self.analyzer.enabled = True
        model.eval()

        try:
            with torch.no_grad():
                for batch in self.loader:
                    inputs = self.prepare_inputs(batch, device)
                    _forward(model, inputs)
            result = self.analyzer.to_dict()
        finally:
            self.analyzer.enabled = was_enabled
            model.train(was_training)

        if global_step is None:
            global_step = state.get(_GLOBAL_STEP, self._last_recorded_step)

        record = {
            "event": event,
            "epoch": _epoch(state, event),
            "batch": _batch(state, event),
            "global_step": global_step,
            "result": result,
        }
        self.records.append(record)
        self._last_recorded_step = global_step
        return record


class _RecordAnalyzerEvaluation(Callback):
    def __init__(self, evaluation: AnalyzerEvaluation, event: str):
        super().__init__()
        self.evaluation = evaluation
        self.event = event

    def on_start(self, state):
        if self.event == "start":
            self.evaluation.record(state, event="start", global_step=0)

    def on_end(self, state):
        if self.event == "end":
            global_step = state.get(_GLOBAL_STEP)
            if self.evaluation._last_recorded_step != global_step:
                self.evaluation.record(
                    state,
                    event="end",
                    global_step=global_step,
                )


class _TrainingIterationCounter(Callback):
    def on_start(self, state):
        state[_GLOBAL_STEP] = 0

    def on_step_training(self, state):
        state[_GLOBAL_STEP] = state.get(_GLOBAL_STEP, 0) + 1


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
