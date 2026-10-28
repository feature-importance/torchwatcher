"""Utilities for selecting and working with model parameters."""

from . import selectors
from .selectors import (
    ParameterInfo,
    ParameterRegistration,
    ParameterSelector,
    SelectorFunction,
    is_bias,
    is_weight,
    matches_full_name,
    matches_full_name_glob,
    matches_module_class,
    matches_module_lambda,
    matches_module_name,
    matches_module_name_glob,
    matches_parameter_name,
    matches_parameter_name_glob,
    parameter_lambda,
    requires_grad,
    select_parameters,
)

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
    "selectors",
]
