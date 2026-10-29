# TorchWatcher

`torchwatcher` uses `torch.fx` to trace and manipulate the compute graphs of
PyTorch models. It supports two closely related tasks:

- modifying forward and backward computation at selected graph nodes;
- collecting activations, gradients, parameter measurements, and derived
  analyses without changing the source code of the model.

These tasks are implemented with _interjections_: small components inserted
into or wrapped around selected graph operations. Because selection happens on
the FX graph, TorchWatcher can target module calls as well as operations such as
`torch.nn.functional.relu` that ordinary module hooks cannot reach.

## Start here

[Install TorchWatcher and run a small example](getting-started.md).

## Guides

- [Interjections](guides/interjections.md) explains forward and wrapped
  interjections and how to select graph nodes.
- [Rewriting model graphs](guides/rewriting-graphs.md) introduces module
  replacement.
- [Selecting model parameters](guides/parameter-selection.md) covers selectors
  and optimizer parameter groups.
- [Logging observations](guides/observation-logging.md) describes the
  storage-independent observation format and sinks.
- [Drawing model graphs](guides/drawing-model-graphs.md) shows how to inspect
  traced and modified models.
- [Analysis](guides/analysis.md) covers analysers, relational analysers, and
  representation comparison with CKA.
- [Training integration](guides/training.md) covers scheduled analyser
  evaluation, parameter callbacks, and parameter tracking.

## Source and examples

The [TorchWatcher repository](https://github.com/feature-importance/torchwatcher)
contains runnable scripts and marimo notebooks in its
[examples directory](https://github.com/feature-importance/torchwatcher/tree/main/examples).

