import abc
import copy
from typing import Callable, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchwatcher.interjection import WrappedForwardBackwardInterjection, \
    WrappedForwardInterjection
from torchwatcher.interjection.interjection import Interjection


class AnalyzerState(dict):
    pass

INPUTS = "inputs"
OUTPUTS = "outputs"
INPUT_GRADIENTS = "input_gradients"
OUTPUT_GRADIENTS = "output_gradients"
MODULE = "module"
NAME = "name"
TARGETS = "targets"


class FInter(WrappedForwardInterjection):
    def __init__(self, analyzer):
        super().__init__()
        self._analyzer = analyzer

    def process(self, name, module, inputs, outputs):
        self._analyzer.log_forward(name, module, inputs, outputs)


class FBInter(WrappedForwardBackwardInterjection):
    def __init__(self, analyzer):
        super().__init__()
        self._analyzer = analyzer

    def process(self, name, module, inputs, outputs):
        self._analyzer.log_forward(name, module, inputs, outputs)

    def process_backward(self, name, module, grad_input, grad_output):
        self._analyzer.log_backward(name, module, grad_input, grad_output)


class Analyzer(WrappedForwardInterjection):
    def __init__(self, gradient=False):
        super().__init__()

        self.current_states: dict[str, AnalyzerState] = {}
        self.working_results: dict[str, Any] = {}

        self.gradient = gradient
        if gradient:
            self.interjection = (FBInter(self),)
        else:
            self.interjection = (FInter(self), )

        self._targets = None

    def forward(self, name, *args):
        return self.interjection[0](name, *args)

    def wrap(self, name: str, module: torch.fx.GraphModule):
        self.interjection[0].wrap(name, module)

    def process(self, name: str, module: [None | nn.Module], inputs, outputs):
        self.interjection[0].process(name, module, inputs, outputs)

    def log_forward(self, name, module, inputs, outputs):
        s = self.current_states[name] = AnalyzerState()
        s[NAME] = name
        s[MODULE] = module
        s[INPUTS] = inputs
        s[OUTPUTS] = outputs
        s[TARGETS] = self.targets

        if not self.gradient:
            self.finalize_state(s)

    def log_backward(self, name, _, grad_input, grad_output):
        s = self.current_states[name]
        s[INPUT_GRADIENTS] = grad_input
        s[OUTPUT_GRADIENTS] = grad_output

        if not self.gradient:
            self.finalize_state(s)

    @property
    def targets(self):
        return self._targets

    def finalize_state(self, state):
        name = state[NAME]
        del self.current_states[name]
        if name in self.working_results:
            working = self.working_results[name]
        else:
            working = None
        self.working_results[name] = self.process_batch_state(name, state, working)

    @abc.abstractmethod
    def process_batch_state(self, name, state, working_results):
        pass

    @abc.abstractmethod
    def result_to_dict(self, result) -> dict:
        pass

    def to_dict(self) -> dict:
        return {k: self.result_to_dict(v) for k, v in
                self.working_results.items()}


class NameAnalyser(Analyzer):
    """
    Just logs the layer name(s)
    """
    def process_batch_state(self, name, state, result):
        return name

    def result_to_dict(self, result) -> dict:
        return {'name': result}
