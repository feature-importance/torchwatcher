import abc
from functools import partial

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from torchwatcher.utils import unpack, unpack_call


def _wrapped_backward_hook(
        interjection,
        name,
        key,
        _module,
        grad_input,
        grad_output,
):
    """Dispatch a registered full-backward hook to its interjection."""
    processed = interjection.process_backward(
        name,
        interjection._original_modules[key],
        unpack(grad_input),
        unpack(grad_output),
    )
    if processed is None or isinstance(processed, tuple):
        return processed
    return (processed,)


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
        inputs = unpack(args)
        processed = self.process(name, self._prev_node[name], inputs)
        return inputs if processed is None else processed


class WrappedForwardInterjection(Interjection):
    def __init__(self):
        super().__init__()
        self._wrapped = {}
        self._original_modules = {}
        self._wrapped_module_names = {}

    def _wrapped_module_name(self, key, name: str) -> str:
        """Return a unique, private registration name for a public node name."""
        if key in self._wrapped_module_names:
            return self._wrapped_module_names[key]

        mode = "train" if key[1] else "eval"
        preferred = f"{name.replace('.', '_')}_{mode}"
        module_name = preferred
        if module_name in self._modules:
            # ``a.b`` and ``a_b`` have the same readable form. Keep public
            # names untouched and disambiguate only the private nn.Module key.
            module_name = f"{preferred}__{name.encode('utf-8').hex()}"
            suffix = 1
            while module_name in self._modules:
                module_name = (
                    f"{preferred}__{name.encode('utf-8').hex()}_{suffix}"
                )
                suffix += 1

        self._wrapped_module_names[key] = module_name
        return module_name

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
        if isinstance(key, tuple):
            # Extracted FX operations belong to the interjection. Native
            # operations are already owned by their _NameWrapper; registering
            # them here as well would let the analyser's mode change the model.
            module_name = self._wrapped_module_name(key, name)
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

    def train(self, mode: bool = True):
        super().train(mode)

        # The train and eval graphs select these wrappers explicitly during
        # forward. Keep each extracted operation in the mode it represents so
        # independently training an analyser cannot change model behaviour.
        for key, module in self._wrapped.items():
            if isinstance(key, tuple):
                module.train(key[1])

        return self

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

        processed = self.process(
            name,
            self._original_modules[key],
            unpack_call(args, kwargs),
            y,
        )
        return y if processed is None else processed


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
        self._register_backward_hook(name, key, module)

    def _register_backward_hook(self, name, key, module):
        hook = partial(_wrapped_backward_hook, self, name, key)
        self._handles[key] = module.register_full_backward_hook(hook)

    def __getstate__(self):
        state = super().__getstate__().copy()
        # RemovableHandle stores weak references to the module hook mappings;
        # serialising those mappings through the handle does not preserve their
        # identity. Hooks are restored explicitly in __setstate__ instead.
        state["_handles"] = {}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._handles = {}
        for key, module in self._wrapped.items():
            for hook_id, hook in tuple(module._backward_hooks.items()):
                if (
                        isinstance(hook, partial)
                        and hook.func is _wrapped_backward_hook
                        and hook.args
                        and hook.args[0] is self
                ):
                    del module._backward_hooks[hook_id]
            name = key[0] if isinstance(key, tuple) else key
            self._register_backward_hook(name, key, module)

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.remove()
