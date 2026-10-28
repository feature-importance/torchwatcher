import pytest
import torch
from torch import nn

from torchwatcher.parameters import selectors
from torchwatcher.parameters.selectors import (
    ParameterRegistration,
    ParameterSelector,
    matches_full_name,
    matches_full_name_glob,
    matches_module_class,
    matches_module_name,
    matches_module_name_glob,
    matches_parameter_name,
    parameters,
    select_parameters,
)


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_parameter = nn.Parameter(torch.ones(1))
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, bias=True),
            nn.BatchNorm2d(4),
        )
        self.classifier = nn.Linear(4, 2, bias=False)


def _names(model, selector=None):
    return [
        info.full_name
        for info in select_parameters(model, selector)
    ]


def test_select_parameters_returns_parameter_context_in_model_order():
    model = ExampleModel()

    selected = select_parameters(model)

    assert [info.full_name for info in selected] == [
        "root_parameter",
        "features.0.weight",
        "features.0.bias",
        "features.1.weight",
        "features.1.bias",
        "classifier.weight",
    ]
    conv_weight = selected[1]
    assert conv_weight.module_name == "features.0"
    assert conv_weight.parameter_name == "weight"
    assert conv_weight.module is model.features[0]
    assert conv_weight.parameter is model.features[0].weight
    assert conv_weight.aliases == ("features.0.weight",)
    assert conv_weight.canonical is conv_weight.registrations[0]
    assert conv_weight.registrations == (
        ParameterRegistration(
            full_name="features.0.weight",
            module_name="features.0",
            parameter_name="weight",
            module=model.features[0],
        ),
    )


def test_parameter_selectors_compose_with_boolean_operators():
    model = ExampleModel()
    selector = (
        matches_parameter_name("weight")
        & ~matches_module_class(nn.BatchNorm2d)
    )

    assert _names(model, selector) == [
        "features.0.weight",
        "classifier.weight",
    ]

    weight_or_bias = (
        matches_parameter_name("weight")
        | matches_parameter_name("bias")
    )
    assert "root_parameter" not in _names(model, weight_or_bias)


def test_all_selector_matches_every_parameter_and_composes_as_identity():
    model = ExampleModel()

    assert _names(model, selectors.all) == _names(model)
    assert _names(
        model,
        selectors.all & selectors.is_weight,
    ) == _names(model, selectors.is_weight)


def test_name_selectors_support_exact_and_glob_matches():
    model = ExampleModel()

    assert _names(model, matches_full_name("classifier.weight")) == [
        "classifier.weight"
    ]
    assert _names(model, matches_full_name_glob("features.*.weight")) == [
        "features.0.weight",
        "features.1.weight",
    ]
    assert _names(model, matches_module_name("features.0")) == [
        "features.0.weight",
        "features.0.bias",
    ]
    assert _names(model, matches_module_name_glob("features.*")) == [
        "features.0.weight",
        "features.0.bias",
        "features.1.weight",
        "features.1.bias",
    ]


def test_module_class_selector_only_matches_directly_owned_parameters():
    model = ExampleModel()

    assert _names(model, matches_module_class(nn.Conv2d)) == [
        "features.0.weight",
        "features.0.bias",
    ]
    assert _names(
        model,
        matches_module_class((nn.Conv2d, nn.Linear)),
    ) == [
        "features.0.weight",
        "features.0.bias",
        "classifier.weight",
    ]


def test_requires_grad_selector_excludes_frozen_parameters():
    model = ExampleModel()
    model.features[0].weight.requires_grad_(False)

    selected = selectors.requires_grad.select(model)

    assert "features.0.weight" not in [info.full_name for info in selected]
    assert "features.0.bias" in [info.full_name for info in selected]


def test_shared_parameters_are_deduplicated_and_keep_their_registrations():
    model = nn.Module()
    model.left = nn.Linear(2, 2, bias=False)
    model.right = nn.Linear(2, 2, bias=False)
    model.right.weight = model.left.weight

    selected = select_parameters(model, matches_full_name("right.weight"))

    assert len(selected) == 1
    assert selected[0].full_name == "left.weight"
    assert selected[0].aliases == ("left.weight", "right.weight")
    assert selected[0].parameter is model.left.weight
    assert [
        registration.module_name
        for registration in selected[0].registrations
    ] == ["left", "right"]
    assert selected[0].registrations[0].module is model.left
    assert selected[0].registrations[1].module is model.right


def test_composed_selectors_match_the_same_shared_parameter_occurrence():
    class ParameterHolder(nn.Module):
        def __init__(self, parameter):
            super().__init__()
            self.weight = parameter

    model = nn.Module()
    model.left = nn.Linear(2, 2, bias=False)
    model.right = ParameterHolder(model.left.weight)
    selector = (
        matches_full_name("right.weight")
        & matches_module_class(nn.Linear)
    )

    assert select_parameters(model, selector) == []


def test_selector_select_is_a_convenience_for_select_parameters():
    model = ExampleModel()
    selector = matches_parameter_name("bias")

    assert [info.full_name for info in selector.select(model)] == [
        "features.0.bias",
        "features.1.bias",
    ]


def test_parameters_returns_selected_parameters_directly():
    model = ExampleModel()
    selector = matches_parameter_name("bias")

    assert all(
        actual is expected
        for actual, expected in zip(
            parameters(model),
            model.parameters(),
            strict=True,
        )
    )

    expected = [
        model.features[0].bias,
        model.features[1].bias,
    ]
    assert all(
        actual is expected_parameter
        for actual, expected_parameter in zip(
            parameters(model, selector),
            expected,
            strict=True,
        )
    )
    assert all(
        actual is expected_parameter
        for actual, expected_parameter in zip(
            selector.parameters(model),
            expected,
            strict=True,
        )
    )


def test_selector_parameters_can_construct_an_optimizer_group():
    model = ExampleModel()

    optimizer = torch.optim.SGD(
        selectors.is_weight.parameters(model),
        lr=0.1,
    )

    assert all(
        actual is expected
        for actual, expected in zip(
            optimizer.param_groups[0]["params"],
            [
                model.features[0].weight,
                model.features[1].weight,
                model.classifier.weight,
            ],
            strict=True,
        )
    )


def test_selector_cannot_be_used_as_a_boolean():
    with pytest.raises(ValueError, match="bitwise symbols"):
        bool(ParameterSelector())


def test_select_parameters_validates_its_arguments():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        select_parameters(object())

    with pytest.raises(TypeError, match="ParameterSelector"):
        select_parameters(ExampleModel(), lambda info: True)
