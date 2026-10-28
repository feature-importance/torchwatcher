import pytest
import torchbearer

from torchwatcher.logging import ObservationContext
from torchwatcher.training import (
    AtSteps,
    EveryNSteps,
    GLOBAL_STEP,
    ObservationSchedule,
    PiecewiseSchedule,
    PredicateSchedule,
    TrainingClock,
    as_iteration_predicate,
    observation_context,
)


def _context(step):
    return ObservationContext(global_step=step)


def test_training_clock_publishes_completed_global_steps():
    clock = TrainingClock()
    state = {
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    clock.on_start(state)
    assert state[GLOBAL_STEP] == 0

    clock.on_step_training(state)
    assert state[GLOBAL_STEP] == 1

    state[torchbearer.BATCH] = 1
    clock.on_step_training(state)
    assert state[GLOBAL_STEP] == 2


def test_multiple_training_clocks_count_each_batch_once():
    left = TrainingClock()
    right = TrainingClock()
    state = {
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 0,
    }

    left.on_start(state)
    right.on_start(state)
    left.on_step_training(state)
    right.on_step_training(state)

    assert state[GLOBAL_STEP] == 1
    assert left.global_step == right.global_step == 1


def test_training_clock_state_can_be_restored():
    clock = TrainingClock(initial_step=4)
    restored = TrainingClock().load_state_dict(clock.state_dict())
    state = {
        torchbearer.EPOCH: 1,
        torchbearer.BATCH: 0,
    }

    restored.on_start(state)
    restored.on_step_training(state)

    assert state[GLOBAL_STEP] == 5


def test_observation_context_uses_training_position():
    state = {
        GLOBAL_STEP: 3,
        torchbearer.EPOCH: 1,
        torchbearer.BATCH: 4,
    }

    context = observation_context(state, "batch")

    assert context == ObservationContext(
        global_step=3,
        epoch=2,
        batch=5,
        event="batch",
    )
    assert observation_context(state, "backward", next_step=True).global_step == 4
    assert observation_context(state, "end").batch is None
    assert observation_context(state, "start") == ObservationContext(
        global_step=3,
        epoch=0,
        batch=0,
        event="start",
    )


def test_training_clock_exposes_observation_context():
    clock = TrainingClock()
    state = {GLOBAL_STEP: 2}

    assert clock.context(state, "backward", next_step=True) == (
        ObservationContext(global_step=3, event="backward")
    )


def test_every_records_at_fixed_periods():
    schedule = EveryNSteps(2)

    assert isinstance(schedule, ObservationSchedule)
    assert not schedule.should_record(_context(None))
    assert not schedule.should_record(_context(0))
    assert not schedule.should_record(_context(1))
    assert schedule.should_record(_context(2))
    assert schedule.should_record(_context(4))
    assert EveryNSteps(2, include_zero=True).should_record(_context(0))


def test_at_records_explicit_steps():
    schedule = AtSteps(1, 3, 5)

    assert schedule.should_record(_context(1))
    assert schedule.should_record(_context(5))
    assert not schedule.should_record(_context(2))
    assert AtSteps([2, 4]).should_record(_context(4))


def test_predicate_schedule_uses_complete_context():
    schedule = PredicateSchedule(
        lambda context: context.event == "backward"
    )

    assert schedule.should_record(ObservationContext(event="backward"))
    assert not schedule.should_record(ObservationContext(event="batch"))


def test_piecewise_schedule_changes_period_at_inclusive_bounds():
    schedule = PiecewiseSchedule([
        (3, 1),
        (10, 2),
        (None, 5),
    ])

    recorded = [
        step
        for step in range(16)
        if schedule.should_record(_context(step))
    ]

    assert recorded == [1, 2, 3, 4, 6, 8, 10, 15]


@pytest.mark.parametrize("period", [0, -1])
def test_every_requires_a_positive_period(period):
    with pytest.raises(ValueError, match="positive"):
        EveryNSteps(period)


def test_schedules_validate_their_configuration():
    with pytest.raises(ValueError, match="at least one step"):
        AtSteps()
    with pytest.raises(ValueError, match="non-negative"):
        AtSteps(-1)
    with pytest.raises(ValueError, match="at least one phase"):
        PiecewiseSchedule([])
    with pytest.raises(ValueError, match="final phase"):
        PiecewiseSchedule([(None, 1), (10, 2)])
    with pytest.raises(ValueError, match="strictly increasing"):
        PiecewiseSchedule([(10, 1), (5, 1)])
    with pytest.raises(ValueError, match="positive"):
        PiecewiseSchedule([(None, 0)])


def test_training_clock_validates_steps():
    with pytest.raises(ValueError, match="non-negative"):
        TrainingClock(-1)

    with pytest.raises(TypeError, match="next_step"):
        observation_context({}, "batch", next_step=1)


def test_observation_schedule_adapts_to_iteration_predicate():
    predicate = as_iteration_predicate(EveryNSteps(2))
    state = {
        GLOBAL_STEP: 2,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 1,
    }

    assert predicate(1, state)
    state[GLOBAL_STEP] = 3
    assert not predicate(2, state)


def test_iteration_predicate_adapter_validates_its_arguments():
    with pytest.raises(TypeError, match="ObservationSchedule"):
        as_iteration_predicate(lambda _: True)

    with pytest.raises(ValueError, match="event"):
        as_iteration_predicate(EveryNSteps(1), event="")
