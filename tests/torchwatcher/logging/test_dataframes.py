import pytest
import torch

from torchwatcher.logging import (
    Observation,
    ObservationContext,
    observation_records,
    observations_to_dataframe,
)


def test_observation_records_create_long_form_rows():
    context = ObservationContext(
        global_step=4,
        epoch=2,
        batch=3,
        event="backward",
    )
    observation = Observation(
        ["analysers", "features"],
        {
            "mean": torch.tensor(1.5),
            "per_channel": torch.tensor([2.0, 3.0]),
        },
        context=context,
        metadata={"dataset": "validation"},
    )

    records = list(observation_records([observation]))

    assert [record["path"] for record in records] == [
        ("analysers", "features", "mean"),
        ("analysers", "features", "per_channel"),
    ]
    assert records[0] == {
        "path": ("analysers", "features", "mean"),
        "global_step": 4,
        "epoch": 2,
        "batch": 3,
        "event": "backward",
        "value": 1.5,
        "metadata": {"dataset": "validation"},
    }
    assert torch.equal(records[1]["value"], torch.tensor([2.0, 3.0]))
    assert records[1]["value"].device.type == "cpu"
    assert not records[1]["value"].requires_grad


def test_observation_records_snapshot_tensor_values():
    value = torch.tensor([1.0, 2.0], requires_grad=True)

    records = list(observation_records([Observation("value", value)]))
    value.detach().add_(10)

    assert torch.equal(records[0]["value"], torch.tensor([1.0, 2.0]))


def test_observation_records_expand_tensor_values():
    observations = [
        Observation("matrix", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        Observation("scalar_tensor", torch.tensor(5.0)),
        Observation("python_scalar", 6.0),
    ]

    records = list(observation_records(
        observations,
        expand_values=True,
    ))

    assert [record["value_index"] for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (),
        None,
    ]
    assert [record["value"] for record in records] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_observation_records_preserve_empty_tensor():
    records = list(observation_records(
        [Observation("empty", torch.empty(0, 2))],
        expand_values=True,
    ))

    assert len(records) == 1
    assert records[0]["value_index"] is None
    assert records[0]["value"].shape == (0, 2)


def test_observation_records_validate_arguments():
    with pytest.raises(TypeError, match="iterable"):
        list(observation_records(None))
    with pytest.raises(TypeError, match="Observation"):
        list(observation_records([object()]))
    with pytest.raises(TypeError, match="expand_values"):
        list(observation_records([], expand_values=1))


def test_observations_to_dataframe_returns_expected_columns():
    pytest.importorskip("pandas")
    observations = [
        Observation(
            ("metric", "loss"),
            torch.tensor(1.25),
            context=ObservationContext(global_step=2, event="batch"),
        )
    ]

    frame = observations_to_dataframe(observations)

    assert list(frame.columns) == [
        "path",
        "global_step",
        "epoch",
        "batch",
        "event",
        "value",
        "metadata",
    ]
    assert frame.to_dict("records") == [{
        "path": ("metric", "loss"),
        "global_step": 2,
        "epoch": None,
        "batch": None,
        "event": "batch",
        "value": 1.25,
        "metadata": {},
    }]


def test_observations_to_dataframe_expands_values():
    pytest.importorskip("pandas")

    frame = observations_to_dataframe(
        [Observation("channels", torch.tensor([1.0, 2.0]))],
        expand_values=True,
    )

    assert list(frame["value_index"]) == [(0,), (1,)]
    assert list(frame["value"]) == [1.0, 2.0]


def test_observations_to_dataframe_preserves_columns_when_empty():
    pytest.importorskip("pandas")

    frame = observations_to_dataframe([])

    assert list(frame.columns) == [
        "path",
        "global_step",
        "epoch",
        "batch",
        "event",
        "value",
        "metadata",
    ]
    assert frame.empty
