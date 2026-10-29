import pytest

from torchwatcher.analysis import analyser_observations
from torchwatcher.logging import ObservationContext


class FakeAnalyser:
    def __init__(self):
        self.calls = 0

    def to_dict(self):
        self.calls += 1
        return {
            "layer": {
                "mean": 1.0,
                "variance": 2.0,
            }
        }


def test_analyser_observations_flattens_analyser_result():
    analyser = FakeAnalyser()
    context = ObservationContext(global_step=3, event="batch")

    observations = list(analyser_observations(
        analyser,
        path=("analysers", "features"),
        context=context,
        metadata={"dataset": "validation"},
    ))

    assert analyser.calls == 1
    assert [item.path for item in observations] == [
        ("analysers", "features", "layer", "mean"),
        ("analysers", "features", "layer", "variance"),
    ]
    assert [item.value for item in observations] == [1.0, 2.0]
    assert all(item.context is context for item in observations)
    assert all(
        item.metadata["dataset"] == "validation"
        for item in observations
    )
    assert all(
        item.metadata["analyser_type"].endswith(".FakeAnalyser")
        for item in observations
    )


def test_analyser_observations_accepts_precomputed_result():
    analyser = FakeAnalyser()

    observations = list(analyser_observations(
        analyser,
        result={"value": 4},
    ))

    assert analyser.calls == 0
    assert observations[0].path == (
        "analysers",
        "FakeAnalyser",
        "value",
    )
    assert observations[0].value == 4
    assert observations[0].context == ObservationContext()


def test_analyser_observations_validates_arguments():
    with pytest.raises(TypeError, match="to_dict"):
        list(analyser_observations(object()))
    with pytest.raises(TypeError, match="metadata"):
        list(analyser_observations(FakeAnalyser(), metadata=[]))
    with pytest.raises(TypeError, match="path"):
        list(analyser_observations(FakeAnalyser(), path=1))
