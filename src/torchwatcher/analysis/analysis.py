import abc
import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Group observations which belong to the same input batch.

        Ordinary analysers process each watched node independently, so their
        batch context is deliberately a no-op. Relational analysers override
        this method to delay processing until all related observations have
        been collected.
        """
        yield

    def run(
        self,
        model: nn.Module | Mapping[str, nn.Module],
        data,
        *,
        batch_size: int | None = None,
        prepare_inputs: (
            Callable[[Any, torch.device], Any]
            | Mapping[str, Callable[[Any, torch.device], Any]]
            | None
        ) = None,
        prepare_targets: Callable[[Any, torch.device], Any] | None = None,
        device: (
            torch.device
            | str
            | Mapping[str, torch.device | str]
            | None
        ) = None,
        reset: bool = True,
        eval_mode: bool = True,
        loader_kwargs: Mapping[str, Any] | None = None,
    ) -> dict:
        """Run watched model(s) over a dataset or dataloader.

        This convenience method owns the analyser's batch boundary and
        temporarily enables it. A regular analyser accepts one model. A
        relational analyser additionally accepts a mapping from source names
        to watched models, for example ``{"student": student,
        "teacher": teacher}``.

        Args:
            model: A watched model, or a source-to-model mapping for a
                relational analyser.
            data: A ``DataLoader`` or a dataset from which one will be built.
            batch_size: Batch size used when ``data`` is a dataset. Defaults to
                32. It must not be supplied with an existing dataloader.
            prepare_inputs: Converts ``(batch, device)`` into model inputs. A
                source-to-callable mapping can prepare different inputs for
                each relational source. By default, the first item of a tuple
                or list batch is used and tensors are moved recursively.
            prepare_targets: Optionally converts ``(batch, device)`` into
                analyser targets. By default, the second item of a tuple or
                list batch is used when present.
            device: Input device, or a source-to-device mapping. By default it
                is inferred from each model's parameters or buffers.
            reset: Reset existing analyser results before running.
            eval_mode: Temporarily put each model into evaluation mode.
            loader_kwargs: Extra ``DataLoader`` arguments used only when
                ``data`` is a dataset.

        Returns:
            The analyser's final ``to_dict()`` result.
        """
        if self.gradient:
            raise ValueError(
                "Analyser.run currently supports forward-only analysers; "
                "gradient analysers require an explicit backward pass"
            )

        models = self._normalise_run_models(model)
        loader = _analysis_loader(data, batch_size, loader_kwargs)
        devices = _analysis_devices(models, device)
        preparers = _analysis_preparers(models, prepare_inputs)
        target_preparer = prepare_targets or _default_analysis_targets
        target_device = next(iter(devices.values()))

        was_enabled = self.enabled
        analyser_was_training = self.training
        training_modes = {
            source: watched_model.training
            for source, watched_model in models.items()
        }
        if reset:
            self.reset()
        self.enabled = True
        if eval_mode:
            self.eval()
            for watched_model in models.values():
                watched_model.eval()

        try:
            with torch.no_grad():
                for raw_batch in loader:
                    targets = target_preparer(raw_batch, target_device)
                    if targets is not _NO_ANALYSIS_TARGET:
                        self.targets = targets
                    with self.batch():
                        for source, watched_model in models.items():
                            inputs = preparers[source](
                                raw_batch,
                                devices[source],
                            )
                            _analysis_forward(watched_model, inputs)
            return self.to_dict()
        finally:
            self.enabled = was_enabled
            for source, watched_model in models.items():
                watched_model.train(training_modes[source])
            self.train(analyser_was_training)

    def _normalise_run_models(
        self,
        model: nn.Module | Mapping[str, nn.Module],
    ) -> dict[str, nn.Module]:
        if isinstance(model, Mapping):
            models = dict(model)
            if len(models) != 1:
                raise ValueError(
                    "regular analysers accept one model; multiple models "
                    "require a RelationalAnalyser"
                )
        else:
            models = {"model": model}
        _validate_analysis_models(models)
        return models

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


@dataclass(frozen=True)
class AnalysisPoint:
    """A watched graph node qualified by its source model or stream."""

    source: str
    name: str

    @property
    def qualified_name(self) -> str:
        return f"{self.source}.{self.name}"


@dataclass(frozen=True)
class AnalysisRelation:
    """A directed relation between two watched graph nodes."""

    left: AnalysisPoint
    right: AnalysisPoint


class _RelationalInterjection(WrappedForwardInterjection):
    """Bind one model/stream source to a relational analyser."""

    def __init__(self, analyser: "RelationalAnalyser", source: str,
                 transform: Callable[[Any], Any] | None = None):
        super().__init__()
        # Avoid registering the analyser as a child module, which would create
        # a cycle once this interjection is inserted into a watched model.
        self._analyser = (analyser,)
        self.source = source
        self._transform = (transform,)

    @property
    def transform(self):
        return self._transform[0]

    def register(self, name, module):
        super().register(name, module)
        self._analyser[0]._register_point(self.source, name)

    def process(self, name, module, inputs, outputs):
        if self.transform is not None:
            outputs = self.transform(outputs)
        self._analyser[0]._log_relational_forward(
            self.source, name, module, inputs, outputs
        )


class RelationalAnalyser[T](Analyser[T]):
    """Base class for analyses which combine observations from multiple nodes.

    Unlike a regular :class:`Analyser`, a relational analyser is not inserted
    directly into a model. Instead, :meth:`watch` creates a source-bound
    interjection. All model forwards which correspond to one input batch must
    run inside :meth:`batch`; processing occurs only when that context exits.

    Subclasses implement :meth:`process_batch_relation`, which is called for
    the Cartesian product of watched layers in each configured source pair.
    """

    def __init__(
        self,
        comparisons: Sequence[tuple[str, str]] | None = None,
    ):
        super().__init__(gradient=False)
        if comparisons is None:
            self._configured_comparisons = None
        else:
            configured = tuple((left, right) for left, right in comparisons)
            if any(not left or not right for left, right in configured):
                raise ValueError("comparison source names must be non-empty")
            if len(set(configured)) != len(configured):
                raise ValueError("comparisons must not contain duplicates")
            self._configured_comparisons = configured
        self._observers: dict[str, _RelationalInterjection] = {}
        self._registered_points: dict[str, set[str]] = {}
        self._schemas: dict[str, tuple[str, ...]] = {}
        self._batch_states: dict[str, dict[str, AnalyserState]] = {}
        self._batch_active = False

    def watch(
        self,
        source: str,
        *,
        transform: Callable[[Any], Any] | None = None,
    ) -> WrappedForwardInterjection:
        """Create the interjection used to observe one model or stream.

        Calling ``watch`` repeatedly for a source returns the same observer.
        A transform must therefore be supplied on its first call only.
        """
        if not source:
            raise ValueError("source must be a non-empty string")
        if self._batch_active or self.working_results:
            raise RuntimeError("sources cannot be added after collection has started")
        if source in self._observers:
            observer = self._observers[source]
            if transform is not None and transform is not observer.transform:
                raise ValueError(
                    f"source {source!r} already has a different transform"
                )
            return observer

        observer = _RelationalInterjection(self, source, transform)
        self._observers[source] = observer
        self._registered_points[source] = set()
        return observer

    @property
    def comparisons(self) -> tuple[tuple[str, str], ...]:
        """The configured or inferred source comparisons."""
        if self._configured_comparisons is not None:
            return self._configured_comparisons
        sources = tuple(self._observers)
        if len(sources) == 1:
            return ((sources[0], sources[0]),)
        if len(sources) == 2:
            return ((sources[0], sources[1]),)
        if not sources:
            return ()
        raise ValueError(
            "comparisons must be specified when watching more than two sources"
        )

    @property
    def registered_points(self) -> dict[str, frozenset[str]]:
        """Return the graph node names registered for each watched source."""
        return {
            source: frozenset(names)
            for source, names in self._registered_points.items()
        }

    def _register_point(self, source: str, name: str):
        self._registered_points[source].add(name)

    def _normalise_run_models(
        self,
        model: nn.Module | Mapping[str, nn.Module],
    ) -> dict[str, nn.Module]:
        required_sources = {
            source for comparison in self.comparisons for source in comparison
        }
        if isinstance(model, Mapping):
            models = dict(model)
        elif len(required_sources) == 1:
            models = {next(iter(required_sources)): model}
        else:
            raise ValueError(
                "cross-source relational analysis requires a "
                "{source: model} mapping"
            )

        _validate_analysis_models(models)
        missing = required_sources.difference(models)
        extra = set(models).difference(required_sources)
        if missing or extra:
            raise ValueError(
                "run model sources do not match configured comparisons; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return models

    def register(self, name, module):
        raise RuntimeError(
            "RelationalAnalyser cannot be interjected directly; use "
            "analyser.watch(source)"
        )

    def forward(self, name, *args):
        raise RuntimeError(
            "RelationalAnalyser cannot be interjected directly; use "
            "analyser.watch(source)"
        )

    def log_forward(self, name, module, inputs, outputs):
        raise RuntimeError(
            "RelationalAnalyser observations must come from analyser.watch(source)"
        )

    def _log_relational_forward(self, source, name, module, inputs, outputs):
        if not self.enabled:
            return
        if not self._batch_active:
            raise RuntimeError(
                "relational observations must be enclosed by analyser.batch()"
            )

        source_states = self._batch_states.setdefault(source, {})
        if name in source_states:
            point = AnalysisPoint(source, name).qualified_name
            raise RuntimeError(f"{point!r} was observed more than once in one batch")

        state = AnalyserState()
        state._name = name
        state._module = module
        state._inputs = inputs
        state._outputs = outputs
        if self._targets_set:
            state._targets = self._targets
            state._targets_set = True
        self.prepare_state(AnalysisPoint(source, name), state)
        source_states[name] = state

    def prepare_state(self, point: AnalysisPoint, state: AnalyserState):
        """Prepare an observation before it is retained for the batch.

        Subclasses can use this hook to cache a compact representation and
        release a large activation before the rest of the models run.
        """
        pass

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Collect and atomically relate observations from one input batch."""
        if not self.enabled:
            yield
            return
        if self._batch_active:
            raise RuntimeError("relational analyser batch contexts cannot be nested")

        self._batch_active = True
        self._batch_states = {}
        try:
            yield
            self._finalize_relational_batch()
        finally:
            self._batch_states = {}
            self._batch_active = False

    def _finalize_relational_batch(self):
        comparisons = self.comparisons
        if not comparisons:
            raise RuntimeError("no watched sources have been registered")

        required_sources = {
            source for comparison in comparisons for source in comparison
        }
        unknown_sources = required_sources.difference(self._observers)
        if unknown_sources:
            names = ", ".join(sorted(unknown_sources))
            raise RuntimeError(f"comparison references unwatched sources: {names}")

        next_schemas = dict(self._schemas)
        for source in required_sources:
            states = self._batch_states.get(source)
            if not states:
                raise RuntimeError(
                    f"source {source!r} produced no observations in this batch"
                )
            observed = tuple(states)
            if source not in self._schemas:
                next_schemas[source] = observed
            elif set(observed) != set(self._schemas[source]):
                expected = set(self._schemas[source])
                actual = set(observed)
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise RuntimeError(
                    f"source {source!r} changed its observed layer set; "
                    f"missing={missing}, extra={extra}"
                )

        self.validate_batch_states(self._batch_states, comparisons)

        next_results = dict(self.working_results)
        for left_source, right_source in comparisons:
            left_states = self._batch_states[left_source]
            right_states = self._batch_states[right_source]
            for left_name in next_schemas[left_source]:
                for right_name in next_schemas[right_source]:
                    relation = AnalysisRelation(
                        AnalysisPoint(left_source, left_name),
                        AnalysisPoint(right_source, right_name),
                    )
                    working = next_results.get(relation)
                    next_results[relation] = self.process_batch_relation(
                        relation,
                        (left_states[left_name], right_states[right_name]),
                        working,
                    )
        self.working_results = next_results
        self._schemas = next_schemas

    def validate_batch_states(
        self,
        states: dict[str, dict[str, AnalyserState]],
        comparisons: tuple[tuple[str, str], ...],
    ):
        """Validate a complete batch before any relation is accumulated.

        Subclasses can override this hook to reject incompatible observations
        or cache batch-local derived values in ``AnalyserState.extras``.
        """
        pass

    @abc.abstractmethod
    def process_batch_relation(
        self,
        relation: AnalysisRelation,
        states: tuple[AnalyserState, AnalyserState],
        working_results: T | None,
    ) -> T | None:
        """Update the result for one relation using a paired batch."""
        pass

    def process_batch_state(self, name, state, working_results):
        raise RuntimeError(
            "RelationalAnalyser processes paired states through "
            "process_batch_relation"
        )

    def reset(self):
        if getattr(self, "_batch_active", False):
            raise RuntimeError("cannot reset a relational analyser during a batch")
        super().reset()
        self._schemas = {}
        self._batch_states = {}


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


def _validate_analysis_models(models: Mapping[str, nn.Module]):
    if not models:
        raise ValueError("at least one model is required")
    invalid = {
        source: type(model).__name__
        for source, model in models.items()
        if not isinstance(source, str) or not isinstance(model, nn.Module)
    }
    if invalid:
        raise TypeError(
            "models must map string source names to nn.Module instances; "
            f"invalid={invalid}"
        )


def _analysis_loader(data, batch_size, loader_kwargs):
    kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    if isinstance(data, DataLoader):
        if batch_size is not None or kwargs:
            raise ValueError(
                "batch_size and loader_kwargs cannot be used with a DataLoader"
            )
        return data

    if "batch_size" in kwargs:
        if batch_size is not None:
            raise ValueError("batch_size was provided twice")
        batch_size = kwargs.pop("batch_size")
    if batch_size is None:
        batch_size = 32
    kwargs.setdefault("shuffle", False)
    return DataLoader(data, batch_size=batch_size, **kwargs)


def _analysis_devices(models, device):
    if isinstance(device, Mapping):
        missing = set(models).difference(device)
        extra = set(device).difference(models)
        if missing or extra:
            raise ValueError(
                "device sources do not match model sources; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return {
            source: torch.device(device[source])
            for source in models
        }
    if device is not None:
        resolved = torch.device(device)
        return {source: resolved for source in models}
    return {
        source: _analysis_model_device(model)
        for source, model in models.items()
    }


def _analysis_model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _analysis_preparers(models, prepare_inputs):
    if isinstance(prepare_inputs, Mapping):
        missing = set(models).difference(prepare_inputs)
        extra = set(prepare_inputs).difference(models)
        if missing or extra:
            raise ValueError(
                "prepare_inputs sources do not match model sources; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return dict(prepare_inputs)

    preparer = prepare_inputs or _default_analysis_inputs
    return {source: preparer for source in models}


def _default_analysis_inputs(batch, device):
    inputs = batch[0] if isinstance(batch, Sequence) else batch
    return _analysis_to_device(inputs, device)


_NO_ANALYSIS_TARGET = object()


def _default_analysis_targets(batch, device):
    if isinstance(batch, Sequence) and len(batch) > 1:
        return _analysis_to_device(batch[1], device)
    return _NO_ANALYSIS_TARGET


def _analysis_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_analysis_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_analysis_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _analysis_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _analysis_forward(model, inputs):
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)
