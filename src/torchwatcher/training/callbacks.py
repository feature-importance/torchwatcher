from collections.abc import Callable, MutableMapping
from typing import Any

import torch
import torchbearer
from torchbearer.callbacks import Callback


class PeriodicEvaluation(Callback):
    """Run an evaluator while Torchbearer training is in progress.

    The callback temporarily switches the model to eval mode, runs the
    evaluator under ``torch.no_grad()``, stores the returned record, and then
    restores the model's previous training mode.
    """

    def __init__(
        self,
        evaluator: Callable[..., Any],
        *,
        loader=None,
        every_n_batches: int | None = None,
        schedule: Callable[[int], bool] | None = None,
        include_start: bool = True,
        include_end: bool = True,
        records: list[Any] | None = None,
    ):
        super().__init__()

        if every_n_batches is None and schedule is None:
            raise ValueError("every_n_batches or schedule must be provided")
        if every_n_batches is not None and schedule is not None:
            raise ValueError("only one of every_n_batches or schedule can be provided")
        if every_n_batches is not None and every_n_batches <= 0:
            raise ValueError("every_n_batches must be a positive integer")

        self.evaluator = evaluator
        self.loader = loader
        self.every_n_batches = every_n_batches
        self.schedule = schedule
        self.include_start = include_start
        self.include_end = include_end
        self.records = [] if records is None else records
        self.global_step = 0
        self._last_recorded_step: int | None = None

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "last_recorded_step": self._last_recorded_step,
            "records": self.records,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("global_step", 0)
        self._last_recorded_step = state_dict.get("last_recorded_step")
        self.records = state_dict.get("records", [])
        return self

    def on_start(self, state):
        if self.include_start:
            self._record(state, event="start", global_step=0)

    def on_step_training(self, state):
        self.global_step += 1
        if self._should_record(self.global_step):
            self._record(
                state,
                event="batch",
                global_step=self.global_step,
            )

    def on_end(self, state):
        if self.include_end and self._last_recorded_step != self.global_step:
            self._record(state, event="end", global_step=self.global_step)

    def _should_record(self, global_step):
        if self.every_n_batches is not None:
            return global_step % self.every_n_batches == 0
        return self.schedule(global_step)

    def _record(self, state, *, event, global_step):
        model = state[torchbearer.MODEL]
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                record = self.evaluator(
                    model,
                    self.loader,
                    state,
                    global_step=global_step,
                    event=event,
                )
        finally:
            model.train(was_training)

        record = self._with_metadata(record, state, event, global_step)
        self.records.append(record)
        self._last_recorded_step = global_step
        return record

    def _with_metadata(self, record, state, event, global_step):
        metadata = {
            "event": event,
            "epoch": _epoch(state, event),
            "batch": _batch(state, event),
            "global_step": global_step,
        }

        if isinstance(record, MutableMapping):
            return {**metadata, **record}

        return {
            **metadata,
            "result": record,
        }


def _epoch(state, event):
    if event == "start":
        return 0
    epoch = state.get(torchbearer.EPOCH, 0)
    return epoch + 1


def _batch(state, event):
    if event == "start":
        return 0
    if event == "end":
        return None
    return state.get(torchbearer.BATCH, 0) + 1
