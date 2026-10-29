"""Internal helpers for resolving training parameter selections."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from torchwatcher.parameters import (
    ParameterInfo,
    ParameterSelector,
    select_parameters,
)


class ParameterSelection:
    """Store a parameter selection and resolve it against a model."""

    def __init__(
            self,
            selector_or_parameters: ParameterSelector | Iterable[nn.Parameter],
    ) -> None:
        self.selector = None
        self.parameters = None

        if isinstance(selector_or_parameters, ParameterSelector):
            self.selector = selector_or_parameters
            return

        try:
            parameters = tuple(selector_or_parameters)
        except TypeError as error:
            raise TypeError(
                "selector_or_parameters must be a ParameterSelector or an "
                "iterable of torch.nn.Parameter objects"
            ) from error

        if any(not isinstance(parameter, nn.Parameter)
               for parameter in parameters):
            raise TypeError(
                "selector_or_parameters must contain only "
                "torch.nn.Parameter objects"
            )

        parameters_by_id = {}
        for parameter in parameters:
            parameters_by_id.setdefault(id(parameter), parameter)
        self.parameters = tuple(parameters_by_id.values())

    def resolve(self, model: nn.Module) -> tuple[ParameterInfo, ...]:
        """Resolve this selection against ``model``."""
        if self.selector is not None:
            return tuple(self.selector.select(model))

        model_parameters = {
            id(info.parameter): info
            for info in select_parameters(model)
        }
        selected = []
        missing = []
        for parameter in self.parameters:
            info = model_parameters.get(id(parameter))
            if info is None:
                missing.append(parameter)
            else:
                selected.append(info)

        if missing:
            raise ValueError(
                f"{len(missing)} selected parameter(s) are not registered "
                "on the trial's model"
            )
        return tuple(selected)


__all__ = ["ParameterSelection"]
