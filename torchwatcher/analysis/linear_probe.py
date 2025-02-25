from typing import Any

import torch
from torch import nn

from .analysis import Analyzer, AnalyzerState


class LinearProbe(Analyzer):
    def __init__(self, num_classes, partial_optim=None, loss_fcn=None):
        super().__init__()

        self.num_classes = num_classes
        self.probes = nn.ModuleDict()
        self.optimizers = {}
        self.partial_optim = partial_optim
        self.loss_fcn = loss_fcn

    def register(self, name, module):
        super().register(name, module)

        probe_name = f"{name.replace(".", "_")}_probe"
        self.probes[probe_name] = nn.LazyLinear(self.num_classes)
        self.optimizers[name] = self.partial_optim(
            self.probes[probe_name].parameters())

    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: Any | None):

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
            predictions: torch.Tensor = probe(x)
            targets: torch.Tensor = state.targets

            # compute acc, predictions, etc
            # TODO: use torchbearer metrics to do this instead, and pass the
            # ones you want into the ctor as arguments
            acc = (predictions.argmax(1) == targets).float().mean()
            count = targets.numel()
            if working_results is not None and isinstance(working_results,
                                                          dict):
                old_acc = working_results['acc']
                new_acc = old_acc * working_results['count'] + acc * count
                working_results['acc'] = new_acc
                working_results['count'] += count
            else:
                return {'acc': acc, 'count': count}

    def train_step(self):
        for name in self.working_results.keys():
            optimizer = self.optimizers[name]
            optimizer.zero_grad()
            predictions = self.working_results[name]
            loss = self.loss_fcn(predictions, self.targets)
            loss.backward()
            optimizer.step()

    def finalise_result(self, result) -> dict:
        return self.working_results