from types import MappingProxyType

import pytest
import torch

from torchwatcher.logging import (
    Observation,
    ObservationContext,
    ObservationPath,
    normalise_observation_path,
)


def test_observation_stores_value_path_and_context():
    value = torch.tensor([1.0, 2.0])
    path: ObservationPath = (
        "parameters",
        "linear.weight",
        "gradient",
        "l2",
    )
    context = ObservationContext(
        global_step=12,
        epoch=2,
        batch=4,
        event="backward",
    )

    observation = Observation(
        path=path,
        value=value,
        context=context,
        metadata={"units": "norm"},
    )

    assert observation.path == path
    assert observation.value is value
    assert observation.context is context
    assert observation.metadata == {"units": "norm"}
    assert isinstance(observation.metadata, MappingProxyType)


def test_observation_copies_metadata():
    metadata = {"source": "gradient"}

    observation = Observation(("metric",), 1.0, metadata=metadata)
    metadata["source"] = "value"

    assert observation.metadata == {"source": "gradient"}
    with pytest.raises(TypeError):
        observation.metadata["source"] = "value"


@pytest.mark.parametrize("path", ["", (), [], ("",), ("parameters", "")])
def test_observation_rejects_empty_paths(path):
    with pytest.raises(ValueError, match="path"):
        Observation(path, 1.0)


@pytest.mark.parametrize("path", [1, b"metric", {"metric"}, ("metric", 1)])
def test_observation_requires_strings_or_a_string_sequence(path):
    with pytest.raises(TypeError, match="path"):
        Observation(path, 1.0)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("metric", ("metric",)),
        ("classifier.weight", ("classifier.weight",)),
        (["analysers", "rank"], ("analysers", "rank")),
        (("parameters", "weight"), ("parameters", "weight")),
    ],
)
def test_observation_normalises_supported_path_forms(path, expected):
    assert Observation(path, 1.0).path == expected
    assert normalise_observation_path(path) == expected


def test_observation_copies_mutable_path_input():
    path = ["analysers", "rank"]

    observation = Observation(path, 1.0)
    path.append("changed")

    assert observation.path == ("analysers", "rank")


def test_observation_validates_context_and_metadata():
    with pytest.raises(TypeError, match="context"):
        Observation(("metric",), 1.0, context={})

    with pytest.raises(TypeError, match="metadata"):
        Observation(("metric",), 1.0, metadata=[])

    with pytest.raises(TypeError, match="metadata keys"):
        Observation(("metric",), 1.0, metadata={1: "value"})


def test_observation_context_fields_are_optional():
    assert ObservationContext() == ObservationContext()


@pytest.mark.parametrize("field", ["global_step", "epoch", "batch"])
def test_observation_context_requires_non_negative_integer_indices(field):
    with pytest.raises(TypeError, match=field):
        ObservationContext(**{field: 1.5})

    with pytest.raises(TypeError, match=field):
        ObservationContext(**{field: True})

    with pytest.raises(ValueError, match=field):
        ObservationContext(**{field: -1})


def test_observation_context_validates_event():
    with pytest.raises(TypeError, match="event"):
        ObservationContext(event=1)

    with pytest.raises(ValueError, match="event"):
        ObservationContext(event="")
