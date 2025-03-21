import abc

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from torchwatcher.utils import unpack, x_if_xp_is_none


class Interjection(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all interjection types."""
    def register(self, name: str, node: torch.fx.GraphModule | None):
        """Called when an interjection is added to the graph. Subclasses can
        override if they need to perform actions before the forward method
        and subsequent process methods are called.

        Args:
            name: Name of the interjection.
            node: the module
        """
        pass


class ForwardInterjection(Interjection):
    """Forward interjection that can be inserted _after_ particular nodes."""

    def __init__(self):
        super().__init__()
        self._prev_node: dict[str, torch.fx.GraphModule | None] = {}

    def register(self, name: str, node: torch.fx.GraphModule | None):
        self._prev_node[name] = node

    @abc.abstractmethod
    def process(self, name: str, module: [None | nn.Module], inputs):
        """Process the output of the interjected node.

        You must not change the input inplace, but you can optionally return
        a modified version of the inputs to pass along in the graph.

        Args:
            name: the name of the interjected node.
            module: the actual module that was interjected. This will be
                None if the interjection was not inserted after a
                module, but rather after a function call or similar.
            inputs: the outputs of the interjected node.

        Returns:
            None, or an object of the same type(s) and shape(s) as
            inputs.
        """
        pass

    def forward(self, name, *args):
        return unpack(
            x_if_xp_is_none(args, self
                            .process(name, self._prev_node[name], unpack(args))))


class WrappedForwardInterjection(Interjection):
    def __init__(self):
        super().__init__()
        self._wrapped: dict[str, torch.fx.GraphModule] = {}

    def register(self, name: str, module: torch.fx.GraphModule):
        self._wrapped[name] = module

    @abc.abstractmethod
    def process(self, name: str, module: [None | nn.Module], inputs, outputs):
        pass

    def forward(self, name, *args, **kwargs):
        y = self._wrapped[name](*args, **kwargs)

        return unpack(
            x_if_xp_is_none(y,
                            self.process(name, self._wrapped[name],
                                         unpack(args), unpack(y))))


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
