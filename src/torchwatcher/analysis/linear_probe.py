import copy
from typing import Any, Callable, List, Union

import torchbearer
from torch import nn, Tensor
from torch.optim.optimizer import ParamsT, Optimizer
from torchbearer import Metric, MetricList
from torchbearer.callbacks import Callback

from .analysis import Analyser, AnalyserState

DEFAULT_METRICS = ['acc']


class LinearProbeCallback(Callback):
    """Torchbearer callback for driving a :class:`LinearProbe`.

    The callback copies the current batch targets from Torchbearer's state onto
    the probe before each forward pass. During training it can also step the
    probe optimisers after the watched model has populated the probe logits.
    """

    def __init__(self,
                 probe: "LinearProbe",
                 *,
                 train_probes: bool = True,
                 target_key=torchbearer.TARGET,
                 keep_model_eval: bool = False):
        super().__init__()
        if train_probes and probe.criterion is None:
            raise ValueError(
                "LinearProbeCallback requires a criterion to train probes"
            )

        self.probe = probe
        self.train_probes = train_probes
        self.target_key = target_key
        self.keep_model_eval = keep_model_eval

    def on_start(self, state):
        if self.keep_model_eval:
            state[torchbearer.MODEL].eval()

    def on_start_training(self, state):
        if self.keep_model_eval:
            state[torchbearer.MODEL].eval()
        self.probe.train()

    def on_sample(self, state):
        self.probe.reset()
        self.probe.targets = state[self.target_key]

    def on_step_training(self, state):
        if self.train_probes:
            self.probe.train_step()

    def on_start_validation(self, state):
        self.probe.eval()
        self.probe.reset()

    def on_sample_validation(self, state):
        self.probe.targets = state[self.target_key]


class LinearProbe(Analyser):
    def __init__(self,
                 num_classes: int,
                 partial_optim: Callable[[ParamsT], Optimizer] = None,
                 criterion: Callable|None = None,
                 metrics: List[Union[str, Metric]] = None):
        super().__init__()

        self.num_classes = num_classes
        self.probes = nn.ModuleDict()
        self.optimisers = {}
        self.partial_optim = partial_optim
        self.criterion = criterion
        self.metrics = MetricList(metrics if metrics is not None
                                  else DEFAULT_METRICS)

    def callback(self,
                 *,
                 train_probes: bool = True,
                 target_key=torchbearer.TARGET,
                 keep_model_eval: bool = False) -> LinearProbeCallback:
        return LinearProbeCallback(
            self,
            train_probes=train_probes,
            target_key=target_key,
            keep_model_eval=keep_model_eval,
        )

    def register(self, name, module):
        super().register(name, module)

        probe_name = f"{name.replace(".", "_")}_probe"
        self.probes[probe_name] = nn.LazyLinear(self.num_classes)
        self.optimisers[name] = self.partial_optim(
            self.probes[probe_name].parameters())

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: Metric | None):

        probe_name = f"{name.replace(".", "_")}_probe"
        x = state.outputs.view(state.outputs.shape[0], -1)

        probe = self.probes[probe_name]
        if self.training:
            # forward on the relevant linear layer and return the logits
            # Note that we detach here; we don't want the grads of the probe
            # to flow backwards; it's just a function of the representation
            # itself
            return probe(x.detach())
        else:
            tstate = torchbearer.State()
            tstate[torchbearer.PREDICTION] = probe(x)
            tstate[torchbearer.TARGET] = state.targets
            tstate[torchbearer.CRITERION] = self.criterion

            if self.criterion is not None:
                tstate[torchbearer.LOSS] = self.criterion(
                    tstate[torchbearer.PREDICTION],
                    tstate[torchbearer.TARGET])

            if working_results is None or isinstance(working_results, Tensor):
                working_results = copy.deepcopy(self.metrics)
                working_results.reset(tstate)

            working_results.process(tstate)

            return working_results

    def finalise_result(self,
                        name: str,
                        result: Metric | Tensor) -> (dict | Metric | Tensor):
        if isinstance(result, Metric):
            return result.process_final()
        return result

    def train_step(self):
        for name in self.working_results.keys():
            optimiser = self.optimisers[name]
            optimiser.zero_grad()
            predictions = self.working_results[name]
            loss = self.criterion(predictions, self.targets)
            loss.backward()
            optimiser.step()
