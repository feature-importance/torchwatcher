import abc

import torch


def x_if_ix_is_none(x, xp):
    return x if xp is None else xp

class AbstractInterjection(abc.ABC):
    @abc.abstractmethod
    def process(self, x, *args, **kwargs):
        pass

class BasicForwardInterjection(AbstractInterjection):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def process(self, x, y=None, *args, **kwargs):
        pass

    def forward(self, x):
        return x_if_ix_is_none(x, self.process(x))


class WrappedForwardInterjection(AbstractInterjection):
    def __init__(self):
        super().__init__()
        self.wrapped: [torch.fx.GraphModule | None] = None

    @abc.abstractmethod
    def process(self, x, y=None, *args, **kwargs):
        pass

    def forward(self, x):
        y = self.wrapped(x)

        return x_if_ix_is_none(y, self.process(x, y))


