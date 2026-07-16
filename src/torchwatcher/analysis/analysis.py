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
                         "or the Analyser instance you're using did not enable "
                         "gradient tracking.")


class AnalyserState():
    """State held by an analyser and used to update the results of the
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

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def input_gradients(self):
        if self._input_gradients_set:
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
    def __init__(self, analyser):
        super().__init__()
        # store the ref to the inter in a tuple to stop it being registered
        # otherwise we'll have cyclic dependencies
        self._analyser = (analyser,)

    def process(self, name, module, inputs, outputs):
        self._analyser[0].log_forward(name, module, inputs, outputs)


class FBInter(WrappedForwardBackwardInterjection):
    def __init__(self, analyser):
        super().__init__()
        # store the ref to the inter in a tuple to stop it being registered
        # otherwise we'll have cyclic dependencies
        self._analyser = (analyser,)

    def process(self, name, module, inputs, outputs):
        self._analyser[0].log_forward(name, module, inputs, outputs)

    def process_backward(self, name, module, grad_input, grad_output):
        self._analyser[0].log_backward(name, module, grad_input, grad_output)


class Analyser[T](WrappedForwardInterjection):
    """Abstract base class for analyser implementations."""

    def __init__(self, gradient=False):
        super().__init__()

        self.current_states: dict[str, AnalyserState] = {}
        self.working_results: dict[str, Any] = {}

        self.gradient = gradient
        if gradient:
            self.interjection = FBInter(self)
        else:
            self.interjection = FInter(self)

        self._targets = None
        self._targets_set = False

        self._enabled = True  # allow Analyser to be disabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def reset(self):
        self.current_states = {}
        self.working_results = {}

    def forward(self, name, *args):
        return self.interjection(name, *args)

    def register(self,
                 name: str,
                 module: torch.fx.GraphModule):
        self.interjection.register(name, module)

    def process(self,
                name: str,
                module: None | nn.Module,
                inputs,
                outputs):
        self.interjection.process(name, module, inputs, outputs)

    def log_forward(self, name, module, inputs, outputs):
        s = self.current_states[name] = AnalyserState()
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

    def finalize_state(self, state: AnalyserState):
        # only if enabled do the updates
        if not self.enabled:
            return

        print(state.name)
        name = state.name
        del self.current_states[name]

        if name in self.working_results:
            working = self.working_results[name]
        else:
            working = None

        self.working_results[name] = self.process_batch_state(name, state, working)

    @abc.abstractmethod
    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: T | None) -> T | None:
        pass

    def finalise_result(self, name: str, result: T) -> T:
        return result

    def to_dict(self) -> dict:
        return {
            k: self.finalise_result(k, v)
            for k, v in self.working_results.items()
        }


class AnalyserList(Analyser[Any]):
    """Wraps multiple analysers into a single analyser."""

    def __init__(self, *args: Analyser):
        super().__init__()
        self.analysers = nn.ModuleList(args)

    def log_forward(self, name, module, inputs, outputs):
        for analyser in self.analysers:
            # just do this here rather than changing the setter. Don't think
            # it will cause problems.
            if self._targets_set:
                analyser.targets = self.targets

            analyser.log_forward(name, module, inputs, outputs)

    def log_backward(self, name, module, grad_input, grad_output):
        for analyser in self.analysers:
            analyser.log_backward(name, module, grad_input, grad_output)

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: Any | None):
        pass

    @Analyser.enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        # also update children so their own logic respects the flag
        for analyser in self.analysers:
            analyser.enabled = value

    def to_dict(self) -> dict:
        result = dict()

        for analyser in self.analysers:
            clz = type(analyser).__name__

            if (isinstance(analyser, PerClassAnalyser) and
                    hasattr(analyser, 'analyser')):
                clz = 'PerClass' + type(analyser.analyser).__name__

            for k, v in analyser.to_dict().items():
                result[f"{clz}.{k}"] = v

        return result

    def register(self, name, module):
        super().register(name, module)
        for analyser in self.analysers:
            analyser.register(name, module)

    def reset(self):
        super().reset()
        for analyser in self.analysers:
            analyser.reset()


class PerClassAnalyser(Analyser[Any]):
    """Wraps an Analyser so that it tracks statistics separately for each
     class."""

    def __init__(self, analyser):
        super().__init__(gradient=analyser.gradient)

        self.analyser = analyser
        self.analysers = {}

    def log_forward(self, name, module, inputs, outputs):
        if not self._targets_set or torch._subclasses.fake_tensor.is_fake(self.targets):
            return

        classes = self.targets

        for c in classes.unique():
            if isinstance(c, torch.Tensor) and c.numel() == 1:
                c = c.cpu().item()

            if c not in self.analysers:
                self.analysers[c] = copy.deepcopy(self.analyser)

            analyser = self.analysers[c]
            analyser.targets = self.targets[classes == c]
            analyser.log_forward(name,
                                 module,
                                 inputs[classes == c],
                                 outputs[classes == c])

    def log_backward(self, name, module, grad_input, grad_output):
        classes = self.targets

        for c, analyser in self.analysers.items():
            if torch.any(classes == c):  # only call if there is data in this batch for this class
                analyser.log_backward(name, module, grad_input[classes == c],
                                      grad_output[classes == c])

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: Any | None):
        pass

    def to_dict(self) -> dict:
        result = dict()
        for c in self.analysers.keys():
            r = self.analysers[c].to_dict()
            result[c] = r

        return result


class PerClassVersusAnalyser(PerClassAnalyser):
    """Wraps an Analyser so that it tracks statistics separately for each class
    and "not" each class.
    """

    def __init__(self, analyser):
        super().__init__(analyser)

    def log_forward(self, name, module, inputs, outputs):
        classes = self.targets

        for c in classes.unique():
            if c not in self.analysers:
                self.analysers[c] = copy.deepcopy(self.analyser)
                self.analysers[f"~{c}"] = copy.deepcopy(self.analyser)

            analyser = self.analysers[c]
            analyser.targets = self.targets[classes == c]
            analyser.log_forward(name, module, inputs[classes == c],
                                 outputs[classes == c])

            analyser = self.analysers[f"~{c}"]
            analyser.targets = self.targets[classes != c]
            analyser.log_forward(name, module, inputs[classes != c],
                                 outputs[classes != c])

    def log_backward(self, name, module, grad_input, grad_output):
        classes = self.targets

        for c, analyser in self.analysers.items():
            if "~" in str(c):
                analyser.log_backward(name, module, grad_input[classes != c],
                                      grad_output[classes != c])
            else:
                analyser.log_backward(name, module, grad_input[classes == c],
                                      grad_output[classes == c])


class NameAnalyser(Analyser[str]):
    """Just logs the layer name(s)"""

    def process_batch_state(self, name, state, result):
        return name
