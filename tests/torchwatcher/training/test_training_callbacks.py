import torch
import torchbearer
from torch import nn

from torchwatcher.training import PeriodicEvaluation


def test_periodic_evaluation_records_metadata_and_restores_training_mode():
    model = nn.Linear(2, 1)
    model.train()
    calls = []

    def evaluator(model, loader, state, *, global_step, event):
        calls.append((model.training, torch.is_grad_enabled(), loader))
        return {"seen_step": global_step, "seen_event": event}

    callback = PeriodicEvaluation(
        evaluator,
        loader="loader",
        every_n_batches=2,
        include_start=True,
        include_end=True,
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    callback.on_start(state)
    callback.on_step_training(state)
    state[torchbearer.BATCH] = 1
    callback.on_step_training(state)
    callback.on_end(state)

    assert model.training
    assert calls == [
        (False, False, "loader"),
        (False, False, "loader"),
    ]
    assert callback.records == [
        {
            "event": "start",
            "epoch": 0,
            "batch": 0,
            "global_step": 0,
            "seen_step": 0,
            "seen_event": "start",
        },
        {
            "event": "batch",
            "epoch": 1,
            "batch": 2,
            "global_step": 2,
            "seen_step": 2,
            "seen_event": "batch",
        },
    ]


def test_periodic_evaluation_records_final_step_when_not_already_recorded():
    model = nn.Linear(2, 1)

    def evaluator(*_args, global_step, event):
        return f"{event}:{global_step}"

    callback = PeriodicEvaluation(
        evaluator,
        every_n_batches=3,
        include_start=False,
        include_end=True,
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 1,
        torchbearer.BATCH: 1,
    }

    callback.on_step_training(state)
    callback.on_step_training(state)
    callback.on_end(state)

    assert callback.records == [
        {
            "event": "end",
            "epoch": 2,
            "batch": None,
            "global_step": 2,
            "result": "end:2",
        }
    ]
