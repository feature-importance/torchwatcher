import abc

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from torchwatcher.utils import unpack, x_if_xp_is_none


class Interjection(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all interjection types."""
    def register(self, name: str, node: torch.fx.GraphModule | nn.Module | None):
        """Called when an interjection is added to the graph. Subclasses can
        override if they need to perform actions before the forward method
        and subsequent process methods are called.

        Args:
            name: Name of the interjection.
            node: the module
        """
        pass

    def register_original_module(self, name: str, module: nn.Module | None):
        """Register the module represented by a wrapped graph node, if any.

        Wrapped interjections execute an extracted ``GraphModule``. When the
        selected node was a module call, this hook separately provides the
        underlying module copied into that graph. Non-module nodes provide
        ``None``. Subclasses which do not need this distinction can ignore it.
        """
        pass

    def _register_for_mode(self, name, module, training: bool):
        self.register(name, module)

    def _register_original_module_for_mode(
            self, name, module, training: bool):
        self.register_original_module(name, module)


class ForwardInterjection(Interjection):
    """Forward interjection that can be inserted _after_ particular nodes."""

    def __init__(self):
        super().__init__()
        self._prev_node: dict[str, torch.fx.GraphModule | None] = {}

    def register(self, name: str, node: torch.fx.GraphModule | nn.Module | None):
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
        self._wrapped = {}
        self._original_modules = {}

    def _registration_key(self, name):
        training = getattr(self, "_registration_training", None)
        return name if training is None else (name, training)

    def _register_for_mode(self, name, module, training: bool):
        self._registration_training = training
        try:
            self.register(name, module)
        finally:
            del self._registration_training

    def _register_original_module_for_mode(
            self, name, module, training: bool):
        self._registration_training = training
        try:
            self.register_original_module(name, module)
        finally:
            del self._registration_training

    def register(self, name: str, module: torch.fx.GraphModule | nn.Module):
        key = self._registration_key(name)
        self._wrapped[key] = module
        # Native interjections wrap the original module directly. FX-based
        # interjections replace this value through register_original_module.
        self._original_modules[key] = module
        module_name = name.replace(".", "_")
        if isinstance(key, tuple):
            module_name += "_train" if key[1] else "_eval"
        self.add_module(module_name, module)

    def register_original_module(self, name: str, module: nn.Module | None):
        self._original_modules[self._registration_key(name)] = module

    def _registered_key(self, name: str, training: bool | None = None):
        if training is None:
            training = self.training
        key = (name, training)
        return key if key in self._wrapped else name

    def wrapped_module(self, name: str, training: bool | None = None):
        """Return the wrapped module registered for the requested mode."""
        return self._wrapped[self._registered_key(name, training)]

    @abc.abstractmethod
    def process(self, name: str, module: [None | nn.Module], inputs, outputs):
        pass

    def forward(
            self,
            name,
            *args,
            _torchwatcher_training: bool | None = None,
            **kwargs):
        key = self._registered_key(name, _torchwatcher_training)
        y = self._wrapped[key](*args, **kwargs)

        return unpack(
            x_if_xp_is_none(y,
                            self.process(name, self._original_modules[key],
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

    def register(self, name: str, module: torch.fx.GraphModule | nn.Module):
        super().register(name, module)
        key = self._registration_key(name)

        def hook(_: nn.Module,
                 grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                 grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> \
                [tuple[torch.Tensor] | torch.Tensor | None]:
            return self.process_backward(name, self._original_modules[key],
                                         unpack(grad_input),
                                         unpack(grad_output))

        self._handles[key] = module.register_full_backward_hook(hook)

    def __del__(self):
        for handle in self._handles.values():
            handle.remove()
