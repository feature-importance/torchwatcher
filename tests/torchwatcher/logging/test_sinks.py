from collections.abc import Iterable

import pytest
import torch

from torchwatcher.logging import (
    CompositeSink,
    InMemorySink,
    Observation,
    ObservationSink,
)


class ExampleSink:
    def __init__(self):
        self.observations = []

    def write(self, observations: Iterable[Observation]) -> None:
        self.observations.extend(observations)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class IncompleteSink:
    def write(self, observations):
        pass


def test_observation_sink_is_runtime_checkable():
    assert isinstance(ExampleSink(), ObservationSink)
    assert not isinstance(IncompleteSink(), ObservationSink)


def test_structural_sink_can_write_observations():
    sink: ObservationSink = ExampleSink()
    observation = Observation(("metric",), 1.0)

    sink.write([observation])

    assert sink.observations == [observation]


def test_in_memory_sink_stores_normalised_snapshots():
    sink = InMemorySink()
    value = torch.tensor([1.0], requires_grad=True)

    sink.write(Observation(("metric",), value) for _ in range(1))
    value.data.zero_()

    assert len(sink.observations) == 1
    assert torch.equal(sink.observations[0].value, torch.tensor([1.0]))
    assert not sink.observations[0].value.requires_grad


def test_in_memory_sink_can_retain_original_observations():
    sink = InMemorySink(normalise=False)
    observation = Observation(("metric",), torch.tensor([1.0]))

    sink.write([observation])

    assert sink.observations == [observation]


def test_in_memory_sink_rejects_writes_after_close():
    sink = InMemorySink()
    sink.close()

    assert sink.closed
    with pytest.raises(RuntimeError, match="closed"):
        sink.write([])


def test_composite_sink_materialises_one_shot_iterables_once():
    left = InMemorySink(normalise=False)
    right = InMemorySink(normalise=False)
    composite = CompositeSink(left, right)
    observations = (
        Observation(("metric", str(index)), index)
        for index in range(2)
    )

    composite.write(observations)

    assert left.observations == right.observations
    assert [item.value for item in left.observations] == [0, 1]


def test_composite_sink_flushes_and_closes_children():
    left = ExampleSink()
    right = ExampleSink()
    composite = CompositeSink(left, right)

    composite.flush()
    composite.close()
    composite.close()

    assert composite.closed


def test_sinks_validate_inputs():
    with pytest.raises(TypeError, match="bool"):
        InMemorySink(normalise=1)

    with pytest.raises(ValueError, match="at least one"):
        CompositeSink()

    with pytest.raises(TypeError, match="ObservationSink"):
        CompositeSink(IncompleteSink())

    with pytest.raises(TypeError, match="Observation"):
        InMemorySink().write([object()])
