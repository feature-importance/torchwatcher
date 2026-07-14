# TorchWatcher

`torchwatcher` uses `torch.fx` to trace and manipulate compute graphs of
`PyTorch` models. Primarily, it has two purposes: modifying the forward and/or
backward passes at specific points in a model; and extracting information
from the forward/backward passes at specific points in order to do analysis
of features, gradients, etc. These tasks are achieved by adding
_interjections_ to the compute graph.

By using `torch.fx` we have the ability to work with models that already exist
without the need to modify their source code to add hooks. We also gain the
ability to add interjections at points which would normally be difficult to
access, such as calls to functions in `torch.nn.functional` which cannot be
hooked with forward or backward hooks.

## Low level APIs: Interjections

`torch.fx` creates a representation of the compute graph of a module through
a process called tracing. The graph itself is represented by a collection of
node objects which are linked together. Nodes in the graph represent
operations such as calling a `nn.Module`s `forward` method, calling a
function (e.g. something like `torch.relu` or `torch.nn.functional.relu`),
calling an arbitrary method, etc.

`torchwatcher`s interjection package allows you to:

- easily select node(s) in the graph where you want to place interjection
- insert interjection either after each desired node, or, by wrapping the
  desired node. The latter allows access to both the forward and backward
  passes on the nodes, whilst the former only lets you access each node's
  forward output.
- construct a new model that will run the interjection automatically on each
  batch when you call it (and train it).

### Interjections

Interjections are subclasses of one of:

- `torchwatcher.interjection.ForwardInterjection`
- `torchwatcher.interjection.WrappedForwardInterjection`
- `torchwatcher.interjection.WrappedForwardBackwardInterjection`

#### Forward Interjections

`ForwardInterjection`s are inserted after a node in the compute graph. They
are passed the selected node's output and are free to return a new value
which will then be passed along to all the following nodes that originally
consumed the selected node's output. In the case that the interjection does not
return anything, the selected node's output is passed along unchanged.

Consider the following compute graph:

      ↓
    node_1
      ↓
    node_2
      ↓
    node_3
      ↓

If an instance of a subclass of `ForwardInterjection` (`my_interjection`) is
inserted at`node_2`, the graph is transformed to

           ↓
         node_1
           ↓
         node_2
           ↓
    my_interjection
           ↓
         node_3
           ↓

To use a `ForwardInterjection` you must create a subclass and override the
`process` method:

    class MyForwardInterjection(ForwardInterjection):
        def process(self, name: str, module: nn.Module | None, inputs):
            print(name, inputs.shape)

    my_interjection = MyForwardInterjection()

The arguments to `process` are the respective node's name, the module (if
the previous node was an `nn.Module`; `None` otherwise) and its outputs
(presented as the inputs to the interjection). The node name is important
because the same interjection instance might be inserted at multiple points
in the graph. The name allows you to disambiguate which node is providing
the input. The type of the inputs to the method depends on what the
selected node outputs; most typically it will be a single tensor, but it
could potentially be a tuple of tensors, a dictionary or anything else.

#### Wrapped Interjections

Wrapped interjections work by inserting a wrapper around a node in the
computational graph. The wrapper can access both the input to the wrapped
node and its outputs. Taking the previous example, if a wrapped interjection
is placed around `node_2`, the graph is transformed to

               ↓
             node_1
               ↓
    my_interjection(node_2)
               ↓
             node_3
               ↓

where internally `my_interjection` passes the output of `node_1` to `node_2`
and returns the output of the `node_2` call. The interjection my alter the
result of the `node_2` call but cannot modify the input (this could be
achieved by placing an interjection on the previous node however).

We provide two base classes for wrapped interjections: the
`WrappedForwardInterjection` class lets you capture the inputs and outputs,
and the `WrappedForwardBackwardInterjection` additionally lets you capture the
computed gradients of the node in question. The latter is particularly nice
as it lets you hook gradients for arbitrary parts of the graph, and not just
calls to `nn.Module` instances - you can capture gradients for calls to
`nn.functional` functions for example.

`WrappedForwardInterjection` require you implement a `process` method:

    class MyWrappedFwd(WrappedForwardInterjection):
        def process(self, name, module, inputs, outputs):
            print(name)

    my_interjection = MyWrappedFwd()

Using `WrappedForwardBackwardInterjection` involves implementing the
`process_backward` method, and optionally, the `process` method if you also
want to interject the forward pass

    class MyWrappedFwdBwd(WrappedForwardBackwardInterjection):
        def process(self, name, module, inputs, outputs):
            print(name)

        def process_backward(self, name, module, grad_input, grad_output):
            print(name)

    my_interjection = MyWrappedFwdBwd()

As with the forward interjection, the node name is passed to both `process`
methods, together with the inputs and outputs (or input gradients and output
gradients). If the interjection was wrapping an underlying `nn.Module` instance,
a reference to that instance is also passed along (this will be `None` if a
different type of node in the compute graph was wrapped).

### Adding interjections to the graph

#### Inserting by matching a module

Use `interject_by_module_class` when you want to attach the same interjection
to every occurrence of a particular `nn.Module` class.

```python
import torch.nn as nn
from torchwatcher.interjection import interject_by_module_class

watched_model = interject_by_module_class(
    model,
    nn.ReLU,
    my_interjection,
)
```

This traces the model and treats the matched module class as a leaf module, so
the interjection is attached to the module call rather than to operations inside
that module. A potential disadvantage of the above example is that you would only
interject on uses of the `nn.ReLU` class, rather than on `nn.functional` calls to
`relu`; the `interject_by_match` API described below allows you to match both.

#### Inserting by name

Use `interject_by_name` when you know the graph node name you want to target.

```python
from torchwatcher.interjection import interject_by_name

watched_model = interject_by_name(
    model,
    "features.3",
    my_interjection,
)
```

A convenient way to discover names is to trace the model or attach a
`NameAnalyser` while developing. The graph visualiser in the
`torchwatcher.drawing` package can also be used to inspect the graph as an image
with the node names shown; see [Drawing model graphs](#drawing-model-graphs).

#### Inserting using the node selector API

Use `interject_by_match` with a node selector when you want to match both
modules and functional calls, such as `nn.ReLU`, `torch.relu`, and
`torch.nn.functional.relu`.

```python
from torchwatcher.interjection import interject_by_match, node_selector

watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    my_interjection,
)
```

Selectors can target broad families of operations, which is useful when a model
mixes module-based layers with calls to the functional API.

#### Tracing and setting custom leaf modules

When tracing custom model components, pass `tracer_kwargs` to control which
modules should be treated as leaves.

```python
watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    my_interjection,
    tracer_kwargs={"leaf_modules": [MyCustomBlock]},
)
```

This is useful when a block should appear as a single graph node rather than
being traced internally.

#### Native Interjections

Native interjections are interjections that are inserted into a model without using the `torch.fx` tracing API.
This is not as powerful as the tracing API because you can only interject on modules themselves, not arbitrary nodes.
However, this is often more convenient than tracing when you want to interject on a module. A common use case would be
to interject on a "block" within a module (for example a Resnet block, or a convolutional layer when you know the
functional api is not being used). They are inserted by calling the `interject_by_module_class_native` function.
Internally wrappers are created to graft the interjection onto the model; from the user perspective usage of the
interjection is identical to those inserted using the `torch.fx` API.

## Rewriting the graph

In addition to interjections, `torchwatcher` also allows you to rewrite the compute graph. This is useful for things
like changing a model dynamically: for example, you could replace all instances of an `nn.Conv2d` with a different
module (for example a quantised version of the same module). Replacement is done through the use of a user-provided
callback function that is given a node in the graph and returns a new node.

`replace_module`: 1-to-1 replacement of a module in the graph, using the `torch.fx` tracing API to identify modules.

`replace_module_native`: 1-to-1 replacement of all modules in the model that match a given class.

## Drawing model graphs

The `torchwatcher.drawing` module can render traced, interjected, or rewritten
models as Graphviz graphs. This is useful for inspecting node names, checking
where interjections were inserted, and seeing how shapes flow through the traced
model.

For most uses, call `trace` first and then pass the traced model to
`draw_graph_pretty`. If `show_shapes=True`, provide example inputs so
`torchwatcher` can run shape propagation before drawing.

```python
import torch
from torchvision.models import resnet18

from torchwatcher.drawing import draw_graph_pretty
from torchwatcher.interjection import trace

model = resnet18()
traced = trace(model)

graph = draw_graph_pretty(
    traced,
    torch.empty(1, 3, 224, 224),
    show_shapes=True,
    show_node_names=True,
)

with open("resnet18.svg", "wb") as f:
    f.write(graph.create_svg())
```

The same drawing helpers work on models returned by `interject_by_match`,
`interject_by_name`, `replace`, and the other graph-manipulation helpers:

```python
from torchwatcher.drawing import draw_graph_pretty
from torchwatcher.interjection import interject_by_match, node_selector

watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    my_interjection,
)

draw_graph_pretty(
    watched_model,
    torch.empty(1, 3, 224, 224),
    show_node_names=True,
).write_png("watched_model.png")
```

`draw_graph_pretty` highlights wrapped interjections and can show tensor shapes,
node names, and selected module constants. For a lower-level view closer to the
underlying FX graph, use `draw_graph(traced_or_watched_model)`. Rendering to
SVG or PNG requires Graphviz to be available on your system.

## High-level APIs: Analysis and logging

Analysers are higher-level interjections for collecting statistics from selected
points in a model. An analyser is inserted into the graph like any other
interjection, but instead of modifying values it accumulates per-node results
across batches.

```python
from torchwatcher.analysis.rank import RankAnalyser
from torchwatcher.interjection import interject_by_match, node_selector

rank_analyser = RankAnalyser(n=4096, threshold=1e-3)

watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    rank_analyser,
)

for inputs, targets in loader:
    watched_model(inputs)

results = rank_analyser.to_dict()
```

`to_dict()` returns a dictionary keyed by graph node name. The value for each
node is analyser-specific. For example, `RankAnalyser` reports values such as
`features_rank`, `features_dim`, and `normalized_features_rank`.

Analysers can be disabled when they are attached to a model but should not run
on every forward pass:

```python
rank_analyser.enabled = False
```

### Evaluating analysers during training

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
