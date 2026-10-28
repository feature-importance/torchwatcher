import pytest
import torch
from torch import nn

from torchwatcher.parameters import selectors
from torchwatcher.training import ParameterGroup, build_parameter_groups


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.classifier = nn.Linear(2, 1)


def test_build_parameter_groups_adds_unmatched_parameters_as_default_group():
    model = ExampleModel()

    groups = build_parameter_groups(
        model,
        ParameterGroup(selectors.is_bias, weight_decay=0.0),
    )

    assert groups[0]["weight_decay"] == 0.0
    assert groups[0]["params"] == [
        model.encoder.bias,
        model.classifier.bias,
    ]
    assert groups[1] == {
        "params": [
            model.encoder.weight,
            model.classifier.weight,
        ],
    }


def test_build_parameter_groups_constructs_a_pytorch_optimizer():
    model = ExampleModel()
    groups = build_parameter_groups(
        model,
        ParameterGroup(selectors.is_bias, weight_decay=0.0),
    )

    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.01)

    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["weight_decay"] == 0.01


def test_build_parameter_groups_preserves_group_declaration_order():
    model = ExampleModel()
    groups = build_parameter_groups(
        model,
        ParameterGroup(
            selectors.matches_module_name("classifier"),
            lr=1e-2,
        ),
        ParameterGroup(
            selectors.matches_module_name("encoder"),
            lr=1e-3,
        ),
        include_remaining=False,
    )

    assert groups[0]["params"] == [
        model.classifier.weight,
        model.classifier.bias,
    ]
    assert groups[0]["lr"] == 1e-2
    assert groups[1]["params"] == [
        model.encoder.weight,
        model.encoder.bias,
    ]
    assert groups[1]["lr"] == 1e-3


def test_build_parameter_groups_rejects_overlapping_groups():
    model = ExampleModel()

    with pytest.raises(ValueError, match="classifier.weight"):
        build_parameter_groups(
            model,
            ParameterGroup(selectors.is_weight),
            ParameterGroup(
                selectors.matches_module_name("classifier")
            ),
        )


def test_build_parameter_groups_rejects_empty_groups():
    model = ExampleModel()

    with pytest.raises(ValueError, match="matched no parameters"):
        build_parameter_groups(
            model,
            ParameterGroup(selectors.matches_full_name("missing")),
        )


def test_build_parameter_groups_can_exclude_unmatched_parameters():
    model = ExampleModel()

    groups = build_parameter_groups(
        model,
        ParameterGroup(selectors.is_bias),
        include_remaining=False,
    )

    assert len(groups) == 1
    assert groups[0]["params"] == [
        model.encoder.bias,
        model.classifier.bias,
    ]


def test_build_parameter_groups_includes_shared_parameters_once():
    model = nn.Module()
    model.left = nn.Linear(2, 2, bias=False)
    model.right = nn.Linear(2, 2, bias=False)
    model.right.weight = model.left.weight

    groups = build_parameter_groups(
        model,
        ParameterGroup(selectors.matches_full_name("right.weight")),
    )

    assert groups == [{"params": [model.left.weight]}]


def test_parameter_group_validates_its_arguments():
    with pytest.raises(TypeError, match="ParameterSelector"):
        ParameterGroup(lambda _: True)

    with pytest.raises(ValueError, match="params"):
        ParameterGroup(selectors.all, **{"params": []})


def test_build_parameter_groups_validates_its_arguments():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        build_parameter_groups(object())

    with pytest.raises(TypeError, match="ParameterGroup"):
        build_parameter_groups(ExampleModel(), selectors.all)

    with pytest.raises(TypeError, match="bool"):
        build_parameter_groups(ExampleModel(), include_remaining=1)
