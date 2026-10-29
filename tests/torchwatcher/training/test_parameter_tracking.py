import pytest
import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchwatcher.logging import InMemorySink
from torchwatcher.parameters import selectors
from torchwatcher.training import (
    AtSteps,
    EveryNSteps,
    ParameterTracker,
    TrainingClock,
)


class RecordingSink:
    def __init__(self):
        self.observations = []
        self.flush_count = 0
        self.closed = False

    def write(self, observations):
        self.observations.extend(observations)

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class IncompleteSink:
    def write(self, observations):
        pass


def test_parameter_tracker_records_measurements_at_backward():
    model = nn.Linear(2, 1)
    sink = InMemorySink()
    tracker = ParameterTracker(
        selectors.matches_full_name("weight"),
        {"gradient_norm": lambda info, _: info.parameter.grad.norm()},
        sink,
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    tracker.on_start(state)
    model.weight.grad = torch.full_like(model.weight, 2)
    tracker.on_backward(state)

    assert tracker.selected[0].parameter is model.weight
    assert len(sink.observations) == 1
    observation = sink.observations[0]
    assert observation.path == ("parameters", "weight", "gradient_norm")
    assert observation.value == pytest.approx(8 ** 0.5)
    assert observation.context.global_step == 1
    assert observation.context.epoch == 1
    assert observation.context.batch == 1
    assert observation.context.event == "backward"
    assert observation.metadata["aliases"] == ("weight",)
    assert observation.metadata["shape"] == (1, 2)


def test_parameter_tracker_applies_schedule_once_per_event():
    model = nn.Linear(2, 1)
    sink = InMemorySink()
    calls = []
    tracker = ParameterTracker(
        selectors.all,
        {"value": lambda info, _: calls.append(info.full_name)},
        sink,
        schedule=AtSteps(2),
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    tracker.on_start(state)
    tracker.on_backward(state)
    tracker.on_step_training(state)
    state[torchbearer.BATCH] = 1
    tracker.on_backward(state)

    assert calls == ["weight", "bias"]
    assert [item.context.global_step for item in sink.observations] == [2, 2]


def test_parameter_tracker_flattens_nested_measurements():
    model = nn.Conv2d(3, 2, 3, bias=False)
    sink = InMemorySink()
    tracker = ParameterTracker(
        [model.weight],
        {
            "norm": lambda info, _: {
                "whole": torch.linalg.vector_norm(info.parameter),
                "per_output_channel": torch.linalg.vector_norm(
                    info.parameter,
                    dim=(1, 2, 3),
                ),
            }
        },
        sink,
        event="on_start",
    )

    tracker.on_start({torchbearer.MODEL: model})

    assert [item.path for item in sink.observations] == [
        ("parameters", "weight", "norm", "whole"),
        ("parameters", "weight", "norm", "per_output_channel"),
    ]
    assert sink.observations[0].context.global_step == 0
    assert sink.observations[0].context.event == "start"
    assert sink.observations[1].value.shape == (2,)


def test_parameter_tracker_records_post_optimizer_values():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )
    sink = InMemorySink()
    tracker = ParameterTracker(
        [model.weight],
        {"value": lambda info, _: info.parameter},
        sink,
        event="on_step_training",
    )
    trial = torchbearer.Trial(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        nn.MSELoss(),
        callbacks=[tracker],
    )

    trial.with_generators(train_generator=loader).to("cpu")
    trial.run(1, verbose=0)

    assert len(sink.observations) == 1
    assert sink.observations[0].value.item() == pytest.approx(0.8)
    assert sink.observations[0].context.global_step == 1
    assert sink.observations[0].context.event == "step"


def test_parameter_tracker_records_actual_optimizer_step():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1), torch.ones(1, 1)),
        batch_size=1,
    )
    sink = InMemorySink()
    tracker = ParameterTracker(
        [model.weight],
        {
            "value": lambda step, info, state: step,
            "norm": lambda step, info, state: torch.linalg.vector_norm(step),
        },
        sink,
        event="optimizer_step",
    )
    trial = torchbearer.Trial(
        model,
        torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            weight_decay=0.5,
        ),
        nn.MSELoss(),
        callbacks=[tracker],
    )

    trial.with_generators(train_generator=loader).to("cpu")
    trial.run(1, verbose=0)

    assert model.weight.item() == pytest.approx(0.95)
    assert [item.path[-1] for item in sink.observations] == ["value", "norm"]
    assert sink.observations[0].value.item() == pytest.approx(-0.05)
    assert sink.observations[1].value.item() == pytest.approx(0.05)
    assert all(
        item.context.global_step == 1
        for item in sink.observations
    )
    assert all(
        item.context.event == "optimizer_step"
        for item in sink.observations
    )


def test_optimizer_step_schedule_is_checked_before_snapshotting():
    model = nn.Linear(1, 1, bias=False)
    sink = InMemorySink()
    tracker = ParameterTracker(
        [model.weight],
        {"value": lambda step, info, state: step},
        sink,
        schedule=AtSteps(2),
        event="optimizer_step",
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    tracker.on_start(state)
    tracker.on_backward(state)
    assert tracker._pending_step is None
    tracker.on_step_training(state)

    state[torchbearer.BATCH] = 1
    tracker.on_backward(state)
    assert tracker._pending_step is not None
    with torch.no_grad():
        model.weight.add_(3)
    tracker.on_step_training(state)

    assert len(sink.observations) == 1
    assert sink.observations[0].value.item() == pytest.approx(3)
    assert sink.observations[0].context.global_step == 2


def test_optimizer_step_requires_paired_backward_event():
    model = nn.Linear(1, 1)
    sink = InMemorySink()
    tracker = ParameterTracker(
        [model.weight],
        {"value": lambda step, info, state: step},
        sink,
        event="optimizer_step",
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    tracker.on_start(state)
    tracker.on_step_training(state)

    assert sink.observations == []


def test_parameter_tracker_can_share_a_clock():
    model = nn.Linear(1, 1)
    clock = TrainingClock(initial_step=4)
    left = ParameterTracker(
        [model.weight],
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
        event="on_step_training",
        clock=clock,
    )
    right = ParameterTracker(
        [model.bias],
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
        event="on_step_training",
        clock=clock,
    )
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    left.on_start(state)
    right.on_start(state)
    left.on_step_training(state)
    right.on_step_training(state)

    assert left.sink.observations[0].context.global_step == 5
    assert right.sink.observations[0].context.global_step == 5


def test_parameter_tracker_flushes_but_does_not_close_sink():
    model = nn.Linear(1, 1)
    sink = RecordingSink()
    tracker = ParameterTracker(
        [model.weight],
        {"value": lambda info, _: info.parameter},
        sink,
    )

    tracker.on_start({torchbearer.MODEL: model})
    tracker.on_end({})

    assert sink.flush_count == 1
    assert not sink.closed


def test_parameter_tracker_restores_clock_state():
    tracker = ParameterTracker(
        selectors.all,
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
        clock=TrainingClock(initial_step=3),
    )
    restored = ParameterTracker(
        selectors.all,
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
    ).load_state_dict(tracker.state_dict())

    assert restored.clock.global_step == 3


@pytest.mark.parametrize(
    ("measurements", "error", "message"),
    [
        ([], TypeError, "mapping"),
        ({}, ValueError, "at least one"),
        ({1: lambda *_: None}, TypeError, "names"),
        ({"": lambda *_: None}, ValueError, "non-empty"),
        ({"value": None}, TypeError, "callable"),
    ],
)
def test_parameter_tracker_validates_measurements(
        measurements, error, message
):
    with pytest.raises(error, match=message):
        ParameterTracker(selectors.all, measurements, InMemorySink())


def test_parameter_tracker_validates_configuration():
    measurement = {"value": lambda info, _: info.parameter}

    with pytest.raises(TypeError, match="ObservationSink"):
        ParameterTracker(selectors.all, measurement, IncompleteSink())
    with pytest.raises(TypeError, match="ObservationSchedule"):
        ParameterTracker(
            selectors.all,
            measurement,
            InMemorySink(),
            schedule=lambda _: True,
        )
    with pytest.raises(ValueError, match="event"):
        ParameterTracker(
            selectors.all,
            measurement,
            InMemorySink(),
            event="missing",
        )
    with pytest.raises(TypeError, match="clock"):
        ParameterTracker(
            selectors.all,
            measurement,
            InMemorySink(),
            clock=object(),
        )


def test_parameter_tracker_rejects_empty_selection():
    model = nn.Linear(1, 1)
    tracker = ParameterTracker(
        selectors.matches_full_name("missing"),
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
    )

    with pytest.raises(ValueError, match="matched no parameters"):
        tracker.on_start({torchbearer.MODEL: model})


def test_parameter_tracker_requires_start_before_tracking_event():
    tracker = ParameterTracker(
        selectors.all,
        {"value": lambda info, _: info.parameter},
        InMemorySink(),
        schedule=EveryNSteps(1),
    )

    with pytest.raises(RuntimeError, match="before on_start"):
        tracker.on_backward({})
