import abc

import torch
import torch.nn as nn


def x_if_xp_is_none(x, xp):
    return x if xp is None else xp


def unpack(result):
    return result[0] if isinstance(result, tuple) and len(result) == 1 else result


class AbstractInterjection(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, *args, **kwargs):
        pass


class ForwardInterjection(AbstractInterjection):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def process(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return unpack(x_if_xp_is_none(args, self.process(*args, **kwargs)))


class WrappedForwardInterjection(AbstractInterjection):
    def __init__(self):
        super().__init__()
        self.wrapped = None

    @property
    def wrapped(self) -> [torch.fx.GraphModule | None]:
        return self._wrapped

    @wrapped.setter
    def wrapped(self, value: torch.fx.GraphModule):
        self._wrapped = value

    @abc.abstractmethod
    def process(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        y = self.wrapped(*args, **kwargs)

        return unpack(x_if_xp_is_none(y, self.process(*args, **kwargs)))


class WrappedForwardBackwardInterjection(WrappedForwardInterjection):
    def __init__(self):
        super().__init__()
        self.wrapped: [torch.fx.GraphModule | None] = None
        self.hook = None

    @abc.abstractmethod
    def process_backward(self,
                         grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                         grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> [tuple[torch.Tensor] |
                                                                                     torch.Tensor | None]:
        pass

    @WrappedForwardInterjection.wrapped.setter
    def wrapped(self, value: torch.fx.GraphModule):
        super().wrapped(value)

        def hook(module: nn.Module,
                 grad_input: [tuple[torch.Tensor, ...] | torch.Tensor],
                 grad_output: [tuple[torch.Tensor, ...] | torch.Tensor]) -> [tuple[torch.Tensor] | torch.Tensor | None]:
            self.process_backward(grad_input, grad_output)
        self.hook = value.register_full_backward_hook(hook)

