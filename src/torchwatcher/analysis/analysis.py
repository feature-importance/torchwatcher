import abc
import copy
from typing import Any

import torch
import torch.nn as nn

from torchwatcher.interjection import WrappedForwardBackwardInterjection, \
    WrappedForwardInterjection


class TargetException(Exception):
    def __init__(self):
        super().__init__("targets has not been set. In your "
                         "training/evaluation loop you need to set the targets "
                         "for each batch before calling the model's forward "
                         "method.")


class NoGradException(Exception):
    def __init__(self):
        super().__init__("Gradients has not been set; either you are trying "
                         "to access them before calling the model's backwards "
                         "or the Analyzer instance you're using did not enable "
                         "gradient tracking.")


class AnalyzerState():
    """State held by an analyzer and used to update the results of the
    analysis."""

    def __init__(self):
        super().__init__()

        self._name = None
        self._module = None
        self._output_gradients = None
        self._input_gradients = None
        self._output_gradients_set = False
        self._input_gradients_set = False
        self._outputs = None
        self._inputs = None
        self._targets = None
        self._targets_set = False

        self.extras = dict()

    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def input_gradients(self):
        if self._input_gradients is None:
            return self._input_gradients
        raise NoGradException()

    @property
    def output_gradients(self):
        if self._output_gradients_set:
            return self._output_gradients
        raise NoGradException()

    @property
    def module(self):
        return self._module

    @property
    def name(self):
        return self._name

    @property
    def targets(self):
        if self._targets_set:
            return self._targets
        raise TargetException()


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


class Analyzer[T](WrappedForwardInterjection):
    """Abstract base class for analyzer implementations."""

    def __init__(self, gradient=False):
        super().__init__()

        self.current_states: dict[str, AnalyzerState] = {}
        self.working_results: dict[str, Any] = {}

        self.gradient = gradient
        # store the ref to the inter in a tuple to stop it being registered
        # otherwise we'll have cyclic dependencies
        if gradient:
            self.interjection = (FBInter(self),)
        else:
            self.interjection = (FInter(self),)

        self._targets = None
        self._targets_set = False

    def reset(self):
        self.current_states = {}
        self.working_results = {}

    def forward(self, name, *args):
        return self.interjection[0](name, *args)

    def register(self,
                 name: str,
                 module: torch.fx.GraphModule):
        self.interjection[0].register(name, module)

    def process(self,
                name: str,
                module: None | nn.Module,
                inputs,
                outputs):
        self.interjection[0].process(name, module, inputs, outputs)

    def log_forward(self, name, module, inputs, outputs):
        s = self.current_states[name] = AnalyzerState()
        s._name = name
        s._module = module
        s._inputs = inputs
        s._outputs = outputs
        if self._targets_set:
            s._targets = self._targets
            s._targets_set = True

        if not self.gradient:
            self.finalize_state(s)

    def log_backward(self, name, _, grad_input, grad_output):
        s = self.current_states[name]

        s._input_gradients = grad_input
        s._input_gradients_set = True
        s._output_gradients = grad_output
        s._output_gradients_set = True

        self.finalize_state(s)

    @property
    def targets(self):
        if self._targets_set:
            return self._targets
        raise TargetException()

    @targets.setter
    def targets(self, targets):
        self._targets = targets
        self._targets_set = True

    def finalize_state(self, state: AnalyzerState):
        name = state.name
        del self.current_states[name]

        if name in self.working_results:
            working = self.working_results[name]
        else:
            working = None

        self.working_results[name] = self.process_batch_state(name, state,
                                                              working)

    @abc.abstractmethod
    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: T | None) -> T | None:
        pass

    def finalise_result(self, name: str, result: T) -> T:
        return result

    def to_dict(self) -> dict:
        return {
            k: self.finalise_result(k, v)
            for k, v in self.working_results.items()
        }


class AnalyzerList(Analyzer[Any]):
    """Wraps multiple analyzers into a single analyzer."""

    def __init__(self, *args: Analyzer):
        super().__init__()
        self.analyzers = args

    def log_forward(self, name, module, inputs, outputs):
        for analyzer in self.analyzers:
            # just do this here rather than changing the setter. Don't think
            # it will cause problems.
            if self._targets_set:
                analyzer.targets = self.targets

            analyzer.log_forward(name, module, inputs, outputs)

    def log_backward(self, name, module, grad_input, grad_output):
        for analyzer in self.analyzers:
            analyzer.log_backward(name, module, grad_input, grad_output)

    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: Any | None):
        pass

    def to_dict(self) -> dict:
        result = dict()

        for analyzer in self.analyzers:
            clz = type(analyzer).__name__

            if (isinstance(analyzer, PerClassAnalyzer) and
                    hasattr(analyzer, 'analyzer')):
                clz = 'PerClass' + type(analyzer.analyzer).__name__

            for k, v in analyzer.to_dict().items():
                result[f"{clz}.{k}"] = v

        return result


class PerClassAnalyzer(Analyzer[Any]):
    """Wraps an Analyzer so that it tracks statistics separately for each
     class."""

    def __init__(self, analyzer):
        super().__init__()

        self.analyzer = analyzer
        self.analyzers = {}

    def log_forward(self, name, module, inputs, outputs):
        classes = self.targets

        for c in classes.unique():
            if c not in self.analyzers:
                self.analyzers[c] = copy.deepcopy(self.analyzer)

            analyzer = self.analyzers[c]
            analyzer.targets = self.targets[classes == c]
            analyzer.log_forward(name,
                                 module,
                                 inputs[classes == c],
                                 outputs[classes == c])

    def log_backward(self, name, module, grad_input, grad_output):
        classes = self.targets

        for c, analyzer in self.analyzers.items():
            analyzer.log_backward(name, module, grad_input[classes == c],
                                  grad_output[classes == c])

    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: Any | None):
        pass

    def to_dict(self) -> dict:
        result = dict()

        for c in self.analyzers.keys():
            r = self.analyzers[c].to_dict()
            for k, v in r.items():
                result[f"{k}_{c}"] = v

        return result


class PerClassVersusAnalyzer(PerClassAnalyzer):
    """Wraps an Analyzer so that it tracks statistics separately for each class
    and "not" each class.
    """

    def __init__(self, analyzer):
        super().__init__(analyzer)

    def log_forward(self, name, module, inputs, outputs):
        classes = self.targets

        for c in classes.unique():
            if c not in self.analyzers:
                self.analyzers[c] = copy.deepcopy(self.analyzer)
                self.analyzers[f"~{c}"] = copy.deepcopy(self.analyzer)

            analyzer = self.analyzers[c]
            analyzer.targets = self.targets[classes == c]
            analyzer.log_forward(name, module, inputs[classes == c],
                                 outputs[classes == c])

            analyzer = self.analyzers[f"~{c}"]
            analyzer.targets = self.targets[classes != c]
            analyzer.log_forward(name, module, inputs[classes != c],
                                 outputs[classes != c])

    def log_backward(self, name, module, grad_input, grad_output):
        classes = self.targets

        for c, analyzer in self.analyzers.items():
            if "~" in str(c):
                analyzer.log_backward(name, module, grad_input[classes != c],
                                      grad_output[classes != c])
            else:
                analyzer.log_backward(name, module, grad_input[classes == c],
                                      grad_output[classes == c])


class NameAnalyzer(Analyzer[str]):
    """Just logs the layer name(s)"""

    def process_batch_state(self, name, state, result):
        return name
