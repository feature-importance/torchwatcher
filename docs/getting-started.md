# Getting started

## Requirements

TorchWatcher requires Python 3.11 or newer and PyTorch 2.2 or newer.

The project is currently installed directly from GitHub:

```bash
python -m pip install "torchwatcher @ git+https://github.com/feature-importance/torchwatcher.git"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/feature-importance/torchwatcher.git
cd torchwatcher
python -m pip install -e ".[test]"
```

## A first interjection

The following interjection prints the shape produced by every `nn.ReLU` module
while leaving its output unchanged:

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


model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)

watched_model = interject_by_module_class(
    model,
    nn.ReLU,
    ShapePrinter(),
)
watched_model(torch.randn(8, 16))
```

A forward interjection can also return a replacement value. If it returns
nothing, as above, the original output continues through the graph unchanged.

Continue with the [interjections guide](guides/interjections.md) to learn about
wrapped forward and backward interjections, functional operations, node
selectors, and custom tracing.

