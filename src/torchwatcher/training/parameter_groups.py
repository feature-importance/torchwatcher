"""Utilities for constructing optimizer parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from torch import nn

from torchwatcher.parameters import ParameterSelector, select_parameters


@dataclass(frozen=True, init=False)
class ParameterGroup:
    """Describe an optimizer parameter group using a selector.

    Args:
        selector: The selector defining which parameters belong to the group.
        **options: Options to include in the optimizer parameter group, such as
            ``lr`` or ``weight_decay``.
    """

    selector: ParameterSelector
    options: Mapping[str, Any]

    def __init__(self, selector: ParameterSelector, **options: Any) -> None:
        if not isinstance(selector, ParameterSelector):
            raise TypeError("selector must be a ParameterSelector")
        if "params" in options:
            raise ValueError("params is managed by build_parameter_groups")

        object.__setattr__(self, "selector", selector)
        object.__setattr__(
            self,
            "options",
            MappingProxyType(dict(options)),
        )


def build_parameter_groups(
        model: nn.Module,
        *groups: ParameterGroup,
        include_remaining: bool = True,
) -> list[dict[str, Any]]:
    """Build optimizer parameter groups for ``model``.

    Each explicitly configured group is returned in declaration order. A
    parameter may belong to at most one explicit group. By default, parameters
    not matched by an explicit group are appended as a final group using the
    optimizer's default options.

    Args:
        model: The model whose parameters should be grouped.
        *groups: Parameter group descriptions to resolve against ``model``.
        include_remaining: Whether to append parameters not matched by an
            explicit group.

    Returns:
        A list of parameter group dictionaries accepted by PyTorch optimizers.

    Raises:
        ValueError: If a group matches no parameters or explicit groups overlap.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(include_remaining, bool):
        raise TypeError("include_remaining must be a bool")
    if any(not isinstance(group, ParameterGroup) for group in groups):
        raise TypeError("groups must be ParameterGroup objects")

    all_parameters = select_parameters(model)
    assigned: set[int] = set()
    resolved = []

    for index, group in enumerate(groups):
        selected = group.selector.select(model)
        if not selected:
            raise ValueError(f"parameter group {index} matched no parameters")

        overlapping = [
            parameter_info
            for parameter_info in selected
            if id(parameter_info.parameter) in assigned
        ]
        if overlapping:
            names = ", ".join(
                parameter_info.full_name
                for parameter_info in overlapping
            )
            raise ValueError(
                f"parameter group {index} overlaps an earlier group: {names}"
            )

        parameters = [
            parameter_info.parameter
            for parameter_info in selected
        ]
        assigned.update(id(parameter) for parameter in parameters)
        resolved.append({
            "params": parameters,
            **group.options,
        })

    if include_remaining:
        remaining = [
            parameter_info.parameter
            for parameter_info in all_parameters
            if id(parameter_info.parameter) not in assigned
        ]
        if remaining:
            resolved.append({"params": remaining})

    return resolved


__all__ = [
    "ParameterGroup",
    "build_parameter_groups",
]
