import copy
from typing import Any, Callable, List, Union

import torchbearer
from torch import nn, Tensor
from torch.optim.optimizer import ParamsT, Optimizer
from torchbearer import Metric, MetricList

from .analysis import Analyzer, AnalyzerState

DEFAULT_METRICS = ['acc']

class LinearProbe(Analyzer):
    def __init__(self,
                 num_classes: int,
                 partial_optim: Callable[[ParamsT], Optimizer] = None,
                 criterion: Callable|None = None,
                 metrics: List[Union[str, Metric]] = None):
        super().__init__()

        self.num_classes = num_classes
        self.probes = nn.ModuleDict()
        self.optimizers = {}
        self.partial_optim = partial_optim
        self.criterion = criterion
        self.metrics = MetricList(metrics if metrics is not None
                                  else DEFAULT_METRICS)

    def register(self, name, module):
        super().register(name, module)

        probe_name = f"{name.replace(".", "_")}_probe"
        self.probes[probe_name] = nn.LazyLinear(self.num_classes)
        self.optimizers[name] = self.partial_optim(
            self.probes[probe_name].parameters())

    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
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

    def finalise_result(self, name: str, result: Metric | Tensor) -> (dict |
                                                                    Metric |
                                                                      Tensor):
        if isinstance(result, Metric):
            return result.process_final()
        return result

    def train_step(self):
        for name in self.working_results.keys():
            optimizer = self.optimizers[name]
            optimizer.zero_grad()
            predictions = self.working_results[name]
            loss = self.criterion(predictions, self.targets)
            loss.backward()
            optimizer.step()

