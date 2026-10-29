# TorchWatcher

`torchwatcher` uses `torch.fx` to trace and manipulate the compute graphs of
PyTorch models. It can modify forward or backward computation at selected graph
nodes and collect activations, gradients, parameter measurements, and derived
analyses without requiring changes to the model's source code.

Because TorchWatcher operates on the FX graph, it can target module calls as
well as operations such as `torch.nn.functional.relu` that ordinary module
hooks cannot reach.

It also has facilities for tracking and logging model parameters and activations 
(during both training and/or inference), and for drawing the model's compute graph.

## Installation

TorchWatcher requires Python 3.11 or newer and PyTorch 2.2 or newer. Install the
current version directly from GitHub:

```bash
python -m pip install "torchwatcher @ git+https://github.com/feature-importance/torchwatcher.git"
```

## Quick example

This interjection observes every `nn.ReLU` module while leaving the model's
output unchanged:

```python
import torch
import torch.nn as nn

from torchwatcher.interjection import (
    ForwardInterjection,
    interject_by_module_class,
)


class ShapePrinter(ForwardInterjection):
    def process(self, name, module, inputs):
        print(name, inputs.shape)


model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
watched_model = interject_by_module_class(model, nn.ReLU, ShapePrinter())
watched_model(torch.randn(8, 16))
```

## Documentation

The full documentation is available at
[feature-importance.github.io/torchwatcher](https://feature-importance.github.io/torchwatcher/).

It includes guides to:

- [interjections](docs/guides/interjections.md);
- [graph rewriting](docs/guides/rewriting-graphs.md);
- [parameter selection](docs/guides/parameter-selection.md);
- [observation logging](docs/guides/observation-logging.md);
- [graph drawing](docs/guides/drawing-model-graphs.md);
- [analysis and CKA](docs/guides/analysis.md);
- [training integration and parameter tracking](docs/guides/training.md).

The documentation sources are available directly at
[docs/index.md](docs/index.md).

## Development

```bash
python -m pip install -e ".[test]"
pytest
```

Build and preview the documentation:

```bash
python -m pip install -r requirements-docs.txt
zensical serve
```

The project is licensed under the BSD 3-Clause License following the terms in [LICENSE](LICENSE).
