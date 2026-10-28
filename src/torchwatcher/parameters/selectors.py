"""Functions and classes for selecting parameters from a model."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from os import linesep
from typing import Callable, Self, Type

from torch import nn

from torchwatcher import utils


@dataclass(frozen=True)
class ParameterRegistration:
    """One registration of a parameter on a module."""

    full_name: str
    module_name: str
    parameter_name: str
    module: nn.Module


@dataclass(frozen=True, eq=False)
class ParameterInfo:
    """A unique parameter and every place it is registered in a model.

    Parameters can be shared by multiple modules or attributes. Each such
    occurrence is represented by a :class:`ParameterRegistration`. The first
    registration encountered during model traversal is canonical; convenience
    properties such as ``full_name`` and ``module`` delegate to it.
    """

    parameter: nn.Parameter
    registrations: tuple[ParameterRegistration, ...]

    def __post_init__(self):
        if not self.registrations:
            raise ValueError("ParameterInfo requires at least one registration")

    @property
    def canonical(self) -> ParameterRegistration:
        """Return the first registration encountered during model traversal."""
        return self.registrations[0]

    @property
    def full_name(self) -> str:
        """Return the canonical qualified parameter name."""
        return self.canonical.full_name

    @property
    def module_name(self) -> str:
        """Return the canonical owning module's qualified name."""
        return self.canonical.module_name

    @property
    def parameter_name(self) -> str:
        """Return the parameter name local to its canonical module."""
        return self.canonical.parameter_name

    @property
    def module(self) -> nn.Module:
        """Return the canonical owning module."""
        return self.canonical.module

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return every qualified name under which the parameter is registered."""
        return tuple(registration.full_name for registration in self.registrations)


SelectorFunction = Callable[[ParameterInfo], bool]


class ParameterSelector:
    """Select parameters using their names, modules and properties.

    Selectors can be composed using ``&`` (and), ``|`` (or), ``^`` (exclusive
    or) and ``~`` (not). For example::

        selector = (
            matches_module_class(nn.Conv2d)
            & matches_parameter_name("weight")
        )
    """

    def __init__(
            self,
            selector_function: SelectorFunction | None = None
    ) -> None:
        if selector_function is not None and not callable(selector_function):
            raise TypeError("selector_function must be callable or None")

        self._selector_function = (
            utils.true if selector_function is None else selector_function
        )

    @property
    def fn(self) -> SelectorFunction:
        """Return the underlying selector function."""
        return self._selector_function

    def __and__(self, other: ParameterSelector) -> ParameterSelector:
        """Return a selector matching parameters selected by both operands."""
        return ParameterSelector(
            lambda info: self.fn(info) and other.fn(info)
        )

    def __or__(self, other: ParameterSelector) -> ParameterSelector:
        """Return a selector matching parameters selected by either operand."""
        return ParameterSelector(
            lambda info: self.fn(info) or other.fn(info)
        )

    def __xor__(self, other: ParameterSelector) -> ParameterSelector:
        """Return a selector matching exactly one operand."""
        return ParameterSelector(
            lambda info: self.fn(info) ^ other.fn(info)
        )

    def __invert__(self) -> ParameterSelector:
        """Return a selector which inverts this selector."""
        return ParameterSelector(lambda info: not self.fn(info))

    def __bool__(self) -> bool:
        """Prevent accidental use of Python's ``and`` and ``or`` operators."""
        raise ValueError(
            "Attempt to convert a ParameterSelector into a Boolean value. "
            f"This is likely a result of trying to use Python primitive logical "
            f"operators (and, or, not, etc.).{linesep}"
            "If this was desired, please use the respective bitwise symbols "
            "(&, |, ~, etc.) instead."
        )

    def __call__(self) -> Self:
        """Return this selector, allowing variable-like and function-like use."""
        return self

    def select(self, model: nn.Module) -> list[ParameterInfo]:
        """Select parameters from ``model`` using this selector."""
        return select_parameters(model, self)


def parameter_lambda(fn: SelectorFunction) -> ParameterSelector:
    """Construct a parameter selector from an arbitrary predicate.

    Args:
        fn: A function receiving a :class:`ParameterInfo` for one parameter
            registration and returning whether that parameter should be
            selected.

    Returns:
        A parameter selector which calls ``fn``.
    """
    return ParameterSelector(fn)


def matches_parameter_name(name: str) -> ParameterSelector:
    """Select parameters whose local name exactly matches ``name``."""
    return parameter_lambda(lambda info: info.parameter_name == name)


def matches_parameter_name_glob(pattern: str) -> ParameterSelector:
    """Select parameters whose local name matches a shell-style glob."""
    return parameter_lambda(
        lambda info: fnmatchcase(info.parameter_name, pattern)
    )


def matches_full_name(name: str) -> ParameterSelector:
    """Select parameters whose qualified name or alias exactly matches ``name``."""
    return parameter_lambda(lambda info: info.full_name == name)


def matches_full_name_glob(pattern: str) -> ParameterSelector:
    """Select parameters with a qualified name matching a shell-style glob."""
    return parameter_lambda(
        lambda info: fnmatchcase(info.full_name, pattern)
    )


def matches_module_name(name: str) -> ParameterSelector:
    """Select parameters owned by the module named ``name``."""
    return parameter_lambda(lambda info: info.module_name == name)


def matches_module_name_glob(pattern: str) -> ParameterSelector:
    """Select parameters owned by modules matching a shell-style glob."""
    return parameter_lambda(
        lambda info: fnmatchcase(info.module_name, pattern)
    )


def matches_module_class(
        clz: Type[nn.Module] | tuple[Type[nn.Module], ...]
) -> ParameterSelector:
    """Select parameters directly owned by instances of ``clz``.

    A tuple of module classes can be provided, matching the behaviour of
    :func:`isinstance`.
    """
    return parameter_lambda(lambda info: isinstance(info.module, clz))


def matches_module_lambda(
        fn: Callable[[nn.Module], bool]
) -> ParameterSelector:
    """Select parameters whose owning module satisfies ``fn``."""
    return parameter_lambda(lambda info: fn(info.module))


def select_parameters(
        model: nn.Module,
        selector: ParameterSelector | None = None
) -> list[ParameterInfo]:
    """Return the parameters in ``model`` matched by ``selector``.

    Parameters are returned in model traversal order and deduplicated by
    identity. For shared parameters, selectors are evaluated independently
    against every registration. The returned :class:`ParameterInfo` uses the
    first occurrence as its canonical registration and exposes every occurrence
    through ``registrations``.

    Args:
        model: The model whose parameters should be selected.
        selector: The selector to apply. If omitted, all parameters are
            selected.

    Returns:
        A list of information about the selected parameters.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if selector is None:
        selector = ParameterSelector()
    if not isinstance(selector, ParameterSelector):
        raise TypeError("selector must be a ParameterSelector or None")

    occurrences = _parameter_occurrences(model)
    parameters: dict[int, nn.Parameter] = {}
    registrations: dict[int, list[ParameterRegistration]] = {}

    for parameter, registration in occurrences:
        key = id(parameter)
        parameters.setdefault(key, parameter)
        registered = registrations.setdefault(key, [])
        if not any(
                existing.full_name == registration.full_name
                for existing in registered
        ):
            registered.append(registration)

    selected: set[int] = set()
    for parameter, registration in occurrences:
        key = id(parameter)
        info = ParameterInfo(parameter, (registration,))
        if selector.fn(info):
            selected.add(key)

    return [
        ParameterInfo(parameter, tuple(registrations[key]))
        for key, parameter in parameters.items()
        if key in selected
    ]


def _parameter_occurrences(
        model: nn.Module
) -> list[tuple[nn.Parameter, ParameterRegistration]]:
    occurrences = []
    for module_name, module in model.named_modules(remove_duplicate=False):
        for parameter_name, parameter in module.named_parameters(
                recurse=False,
                remove_duplicate=False
        ):
            full_name = (
                f"{module_name}.{parameter_name}"
                if module_name else parameter_name
            )
            occurrences.append((parameter, ParameterRegistration(
                full_name=full_name,
                module_name=module_name,
                parameter_name=parameter_name,
                module=module,
            )))
    return occurrences


requires_grad = ParameterSelector(lambda info: info.parameter.requires_grad)
"""A selector which is true for parameters that require gradients."""

is_weight = matches_parameter_name("weight")
"""A selector which is true for parameters locally named ``weight``."""

is_bias = matches_parameter_name("bias")
"""A selector which is true for parameters locally named ``bias``."""


__all__ = [
    "ParameterInfo",
    "ParameterRegistration",
    "ParameterSelector",
    "SelectorFunction",
    "is_bias",
    "is_weight",
    "matches_full_name",
    "matches_full_name_glob",
    "matches_module_class",
    "matches_module_lambda",
    "matches_module_name",
    "matches_module_name_glob",
    "matches_parameter_name",
    "matches_parameter_name_glob",
    "parameter_lambda",
    "requires_grad",
    "select_parameters",
]
