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
        """Create a callback for using a probe inside a Torchbearer trial.

        Args:
            probe: The probe analyser to drive.
            train_probes: If ``True``, call :meth:`LinearProbe.train_step`
                after each Torchbearer training step. Set this to ``False``
                for evaluation-only passes.
            target_key: Torchbearer state key containing the current batch
                targets.
            keep_model_eval: If ``True``, keep the watched model in eval mode
                during the training pass. This is useful when training probes
                over a frozen backbone.
        """
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
    """Train and evaluate linear classifiers on watched activations.

    A ``LinearProbe`` attaches one lazy linear classifier to each registered
    layer. In training mode, each watched forward pass stores the detached probe
    logits for that batch; calling :meth:`train_step` then updates every probe
    against the current ``targets``. In eval mode, the probes do not update and
    instead accumulate Torchbearer metrics such as accuracy.

    Targets can be set manually via ``probe.targets = targets`` before the
    watched model's forward pass, or automatically by using
    :meth:`callback` inside a Torchbearer ``Trial``.
    """

    def __init__(self,
                 num_classes: int,
                 partial_optim: Callable[[ParamsT], Optimizer] = None,
                 criterion: Callable|None = None,
                 metrics: List[Union[str, Metric]] = None):
        """Create a linear-probe analyser.

        Args:
            num_classes: Number of output classes for each probe classifier.
            partial_optim: Callable that receives a probe's parameters and
                returns its optimiser, for example
                ``lambda params: torch.optim.Adam(params, lr=1e-3)``.
            criterion: Loss function used by :meth:`train_step` and optional
                evaluation loss reporting.
            metrics: Torchbearer metric names or metric instances to accumulate
                during eval mode. Defaults to accuracy.
        """
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
        """Return a Torchbearer callback that drives this probe.

        The callback sets ``targets`` from Torchbearer's state before each
        watched forward pass. During training it can also update the probe
        optimisers, allowing a standard ``Trial.run`` / ``Trial.evaluate``
        workflow to replace manual probe loops.

        Args:
            train_probes: If ``True``, update probes during the training pass.
            target_key: Torchbearer state key containing batch targets.
            keep_model_eval: If ``True``, keep the watched model in eval mode
                while the probes train.
        """
        return LinearProbeCallback(
            self,
            train_probes=train_probes,
            target_key=target_key,
            keep_model_eval=keep_model_eval,
        )

    def register(self, name, module):
        """Register a watched layer and create its corresponding probe."""
        super().register(name, module)

        probe_name = f"{name.replace(".", "_")}_probe"
        self.probes[probe_name] = nn.LazyLinear(self.num_classes)
        self.optimisers[name] = self.partial_optim(
            self.probes[probe_name].parameters())

    def process_batch_state(self,
                            name: str,
                            state: AnalyserState,
                            working_results: Metric | None):
        """Process one watched layer's activations for the current batch.

        In training mode this returns probe logits, which are later consumed by
        :meth:`train_step`. In eval mode it updates and returns the metric list
        for this layer.
        """

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
        """Convert accumulated eval metrics into their final dictionary form."""
        if isinstance(result, Metric):
            return result.process_final()
        return result

    def train_step(self):
        """Update every registered probe from the current batch logits.

        This expects ``targets`` to have been set before the watched model's
        forward pass, either manually or via :class:`LinearProbeCallback`.
        """
        for name in self.working_results.keys():
            optimiser = self.optimisers[name]
            optimiser.zero_grad()
            predictions = self.working_results[name]
            loss = self.criterion(predictions, self.targets)
            loss.backward()
            optimiser.step()
