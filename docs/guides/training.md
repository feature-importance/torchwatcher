# Training integration

## Evaluating analysers during training

`AnalyserEvaluation` is a Torchbearer callback helper for taking analyser
snapshots during training. It temporarily resets and enables the analyser, runs
a fixed loader with the model in eval mode, stores the analyser result, and then
restores the previous analyser and model training state. Snapshot passes run
with gradient computation disabled by default.

```python
import torchbearer
from torchwatcher.analysis.rank import RankAnalyser
from torchwatcher.interjection import interject_by_match, node_selector
from torchwatcher.training import AnalyserEvaluation

rank_analyser = RankAnalyser(n=4096, threshold=1e-3)
rank_analyser.enabled = False

watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    rank_analyser,
)

rank_evaluation = AnalyserEvaluation(
    rank_analyser,
    rank_loader,
)

trial = torchbearer.Trial(
    watched_model,
    optimiser,
    criterion,
    metrics=["loss", "acc"],
    callbacks=[
        *rank_evaluation.callbacks(
            every_n_batches=50,
            include_start=True,
            include_end=True,
        ),
    ],
)

trial.with_generators(train_generator=train_loader).to(device)
history = trial.run(epochs)

snapshots = rank_evaluation.records
```

Evaluation results can also be written directly to an observation sink:

```python
from torchwatcher.logging import InMemorySink

sink = InMemorySink()
rank_evaluation = AnalyserEvaluation(
    rank_analyser,
    rank_loader,
    sink=sink,
    path=("analysers", "rank"),
    retain_records=False,
)
```

`retain_records` defaults to `True` for compatibility. Set it to `False` when
the sink is the authoritative store so large results are not also retained in
memory. The sink receives flattened observations whose context identifies the
training step that triggered the evaluation; its batch is not the batch index
within the evaluation loader. Evaluation callbacks flush the sink at training
end but do not close it.

Scheduling and global-step tracking are shared training utilities. Built-in
schedules receive an `ObservationContext`, allowing the same schedule to be
used by analyser evaluation and parameter tracking:

```python
from torchwatcher.training import PiecewiseSchedule, TrainingClock

clock = TrainingClock()
schedule = PiecewiseSchedule([
    (100, 1),
    (1000, 10),
    (None, 100),
])

callbacks = rank_evaluation.callbacks(
    schedule=schedule,
    clock=clock,
)
```

`EveryNSteps(n)` records at fixed intervals, `AtSteps(...)` records explicit
global steps, and `PredicateSchedule` adapts an arbitrary context predicate.
Global step zero represents the initial model state; the first completed
optimizer step is one. Pre-step consumers can associate gradients with that
same step by requesting the upcoming-step context from `TrainingClock`.

Each record has the following structure:

```text
{
    "event": "start" or "batch" or "end",
    "epoch": <int>,
    "batch": <int or None>,
    "global_step": <int>,
    "result": <analyser.to_dict()>,
}
```

For custom batches, pass `prepare_inputs` to control how batches are moved to
the model device and passed into the model.

```python
evaluation = AnalyserEvaluation(
    analyser,
    eval_loader,
    prepare_inputs=lambda batch, device: {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    },
)
```

For analysers that need gradients, pass `compute_gradients=True`. If the
analyser needs a backward pass, also pass a `backward` callback. The callback is
called with the model outputs and the original batch. It can either perform
backward itself or return a tensor to be backpropagated.

```python
gradient_evaluation = AnalyserEvaluation(
    gradient_analyser,
    eval_loader,
    compute_gradients=True,
    backward=lambda outputs, batch: outputs.sum(),
)
```

`AnalyserEvaluation` preserves any parameter gradients that existed before a
gradient-enabled snapshot and restores them afterward, so evaluation gradients
do not leak into the surrounding training loop.

## Applying functions to selected parameters

`SelectedParametersCallback` adapts parameter selectors to the Torchbearer
training lifecycle. It resolves the selection against the trial model at
startup and calls an action once for each selected `ParameterInfo`, in selection
order. By default, the action runs at `on_backward`, after gradients have been
computed and before the optimizer step.

```python
import torch
from torchwatcher.parameters import selectors
from torchwatcher.training import SelectedParametersCallback

conv_weights = (
    selectors.matches_module_class(torch.nn.Conv2d)
    & selectors.is_weight
)

clip_convolution_gradients = SelectedParametersCallback(
    conv_weights,
    lambda parameter_info, state: torch.nn.utils.clip_grad_value_(
        parameter_info.parameter,
        clip_value=1.0,
    ),
)
```

An explicit iterable of parameters can be supplied instead of a selector. The
callback resolves those parameters to `ParameterInfo` objects, preserves their
input order, removes duplicates, and reports parameters which are not
registered on the trial model.

Set `event` to `on_start`, `on_backward`, or `on_step_training` to apply the
action at model startup, between backward and the optimizer step, or after the
optimizer step respectively. The action's return value is ignored; any
mutation must be performed explicitly.

## Tracking selected parameters

`ParameterTracker` records named measurements of selected parameters. Each
measurement receives a `ParameterInfo` and the Torchbearer state, and can
return a scalar, tensor, or nested mapping. A schedule controls which training
steps are sent to the observation sink:

```python
from torchwatcher.logging import InMemorySink
from torchwatcher.parameters import selectors
from torchwatcher.training import EveryNSteps, ParameterTracker

sink = InMemorySink()
tracker = ParameterTracker(
    selectors.is_weight,
    {
        "gradient_norm": lambda info, state: info.parameter.grad.norm(),
    },
    sink,
    schedule=EveryNSteps(10),
)
```

The default event is `on_backward`, after gradients have been computed and
before the optimizer update. Backward observations are associated with the
upcoming optimizer step. Use `event="on_step_training"` to record parameter
values after the optimizer update, or `event="on_start"` to record the initial
state.

Measurements can return nested mappings for related aggregations. They are
flattened into separate observation paths:

```python
import torch

channel_norms = ParameterTracker(
    selectors.matches_module_class(torch.nn.Conv2d) & selectors.is_weight,
    {
        "gradient_norm": lambda info, state: {
            "whole": torch.linalg.vector_norm(info.parameter.grad),
            "per_output_channel": torch.linalg.vector_norm(
                info.parameter.grad,
                dim=(1, 2, 3),
            ),
        },
    },
    sink,
)
```

The tracker includes a `TrainingClock`; pass the same clock to multiple
trackers or analyser-evaluation callbacks when they should share restored
state. At training end it flushes the sink but leaves closing caller-owned
sinks to the caller.

To measure the update actually applied by the optimizer, use the synthetic
`optimizer_step` event. The tracker snapshots selected parameters after
backward, then subtracts that snapshot from the post-optimizer value. These
measurement functions receive `(step, parameter_info, state)`, where
`step = parameter_after - parameter_before`:

```python
applied_steps = ParameterTracker(
    selectors.is_weight,
    {
        "norm": lambda step, info, state: torch.linalg.vector_norm(step),
        "per_output_channel": lambda step, info, state: (
            torch.linalg.vector_norm(step, dim=(1, 2, 3))
        ),
    },
    sink,
    schedule=EveryNSteps(10),
    event="optimizer_step",
)
```

This records the realized update, including learning rate, momentum, adaptive
scaling, weight decay and prior gradient mutation. The schedule is evaluated
before the optimizer runs, so parameter snapshots are only retained for steps
that will be recorded. Callback ordering remains significant: parameter
mutations made after the optimizer but before the tracker's
`on_step_training` callback are included in the measured step.

