import pytest
import torch

from torchwatcher.logging import (
    Observation,
    ObservationContext,
    flatten_observation,
    normalise_observation,
    normalise_value,
)


def test_flatten_observation_extends_paths_for_nested_mappings():
    context = ObservationContext(global_step=2)
    observation = Observation(
        ("analysis", "FeatureStatistics"),
        {
            "layer": {
                "mean": 1.0,
                "variance": 2.0,
            },
            3: {"mean": 4.0},
        },
        context=context,
        metadata={"split": "validation"},
    )

    flattened = list(flatten_observation(observation))

    assert [item.path for item in flattened] == [
        ("analysis", "FeatureStatistics", "layer", "mean"),
        ("analysis", "FeatureStatistics", "layer", "variance"),
        ("analysis", "FeatureStatistics", "3", "mean"),
    ]
    assert [item.value for item in flattened] == [1.0, 2.0, 4.0]
    assert all(item.context is context for item in flattened)
    assert all(
        item.metadata == {"split": "validation"}
        for item in flattened
    )


def test_flatten_observation_keeps_empty_mappings_as_leaves():
    observation = Observation(("analysis",), {})

    assert list(flatten_observation(observation)) == [observation]


def test_flatten_observation_rejects_path_component_collisions():
    observation = Observation(("analysis",), {1: "one", "1": "string one"})

    with pytest.raises(ValueError, match="duplicate path component"):
        list(flatten_observation(observation))


def test_flatten_observation_validates_its_argument():
    with pytest.raises(TypeError, match="Observation"):
        list(flatten_observation({}))


def test_normalise_value_recursively_snapshots_tensors():
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    value = {
        "tensor": source,
        "tuple": (source,),
        "list": [source],
    }

    normalised = normalise_value(value)
    source.data.zero_()

    assert torch.equal(normalised["tensor"], torch.tensor([1.0, 2.0]))
    assert torch.equal(normalised["tuple"][0], torch.tensor([1.0, 2.0]))
    assert torch.equal(normalised["list"][0], torch.tensor([1.0, 2.0]))
    assert not normalised["tensor"].requires_grad
    assert normalised["tensor"].device.type == "cpu"


def test_normalise_observation_snapshots_value_and_metadata():
    value = torch.tensor([1.0], requires_grad=True)
    metadata_value = torch.tensor([2.0], requires_grad=True)
    context = ObservationContext(event="backward")
    observation = Observation(
        ("metric",),
        value,
        context=context,
        metadata={"tensor": metadata_value},
    )

    normalised = normalise_observation(observation)
    value.data.zero_()
    metadata_value.data.zero_()

    assert normalised is not observation
    assert normalised.path == observation.path
    assert normalised.context is context
    assert torch.equal(normalised.value, torch.tensor([1.0]))
    assert torch.equal(
        normalised.metadata["tensor"],
        torch.tensor([2.0]),
    )


def test_normalise_observation_validates_its_argument():
    with pytest.raises(TypeError, match="Observation"):
        normalise_observation({})
