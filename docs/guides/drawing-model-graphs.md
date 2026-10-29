# Drawing model graphs

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

