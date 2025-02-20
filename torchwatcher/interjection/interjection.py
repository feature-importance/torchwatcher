import abc
from typing import Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from torchwatcher.utils import unpack, x_if_xp_is_none


class Interjection(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for all interjection types.
    """
    pass


class ForwardInterjection(Interjection):
    """
    Forward interjection that can be inserted _after_ particular nodes.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def process(self, name: str, module: [None | nn.Module], inputs):
        pass

    def forward(self, name, module: [None | nn.Module], *args):
        return unpack(x_if_xp_is_none(args,
                                      self.process(name, module, unpack(args))))


class WrappedForwardInterjection(Interjection):
    def __init__(self):
        super().__init__()
        self._wrapped: dict[str, torch.fx.GraphModule] = {}

    def wrap(self, name: str, module: torch.fx.GraphModule):
        self._wrapped[name] = module

    @abc.abstractmethod
    def process(self, name: str, module: [None | nn.Module], inputs, outputs):
        pass

    def forward(self, name, module: [None | nn.Module], *args, **kwargs):
        y = self._wrapped[name](*args, **kwargs)

        return unpack(
            x_if_xp_is_none(y, self.process(name, module, unpack(args),
                                            unpack(y))))


class WrappedForwardBackwardInterjection(WrappedForwardInterjection):
    def __init__(self):
        super().__init__()
        self._handles: dict[str, RemovableHandle] = {}

    def process(self, name, module: [None | nn.Module], inputs, outputs):
        # concrete implementation for convenience if subclasses only care
        # about backward. Just override if you want to hook both forward
        # and backward passes.
        return

    @abc.abstractmethod
    def process_backward(self,
                         name,
                         module: [None | nn.Module],
                         grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                         grad_output: [
                             tuple[torch.Tensor, ...] | torch.Tensor]) -> [
        tuple[torch.Tensor] | torch.Tensor | None]:
        pass

    def wrap(self, name: str, module: torch.fx.GraphModule):
        super().wrap(name, module)

        def hook(_: nn.Module,
                 grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                 grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> \
                [tuple[torch.Tensor] | torch.Tensor | None]:
            return self.process_backward(name, module, unpack(grad_input),
                                         unpack(grad_output))

        self._handles[name] = module.register_full_backward_hook(hook)

    def __del__(self):
        for handle in self._handles.values():
            handle.remove()
