import abc
from typing import Callable

import torch
import torch.nn as nn


def x_if_xp_is_none(x, xp):
    return x if xp is None else xp


def unpack(result):
    return result[0] if isinstance(result, tuple) and len(result) == 1 else result


class Interjection(nn.Module, metaclass=abc.ABCMeta):
    pass


class ForwardInterjection(Interjection):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def process(self, name, input):
        pass

    def forward(self, name, *args):
        return unpack(x_if_xp_is_none(args, self.process(name, unpack(args))))


class WrappedForwardInterjection(Interjection):
    def __init__(self):
        super().__init__()
        self._wrapped: dict[str, torch.fx.GraphModule] = {}

    def wrap(self, name: str, module: torch.fx.GraphModule):
        self._wrapped[name] = module

    @abc.abstractmethod
    def process(self, name, input, output):
        pass

    def forward(self, name, *args, **kwargs):
        y = self._wrapped[name](*args, **kwargs)

        return unpack(x_if_xp_is_none(y, self.process(name, unpack(args), unpack(y))))


class WrappedForwardBackwardInterjection(WrappedForwardInterjection):
    def __init__(self):
        super().__init__()
        self._handles: dict[str, Callable] = {}

    @abc.abstractmethod
    def process_backward(self,
                         name,
                         grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                         grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> [tuple[torch.Tensor] |
                                                                                     torch.Tensor | None]:
        pass

    def wrap(self, name: str, module: torch.fx.GraphModule):
        super().wrap(name, module)

        def hook(_: nn.Module,
                 grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                 grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> [tuple[torch.Tensor] | torch.Tensor | None]:
            return self.process_backward(name, unpack(grad_input), unpack(grad_output))
        self._handles[name] = module.register_full_backward_hook(hook)

