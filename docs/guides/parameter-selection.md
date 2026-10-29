# Selecting model parameters

Parameter selectors provide a reusable way to find model parameters by their
qualified name, local name, owning module, or arbitrary properties. Selectors
compose with `&` (and), `|` (or), `^` (exclusive or), and `~` (not), in
the same way as node selectors.

```python
import torch.nn as nn
from torchwatcher.parameters import selectors

conv_weights = (
    selectors.matches_module_class(nn.Conv2d)
    & selectors.matches_parameter_name("weight")
)

selected = conv_weights.select(model)
for info in selected:
    print(info.full_name, info.parameter.shape)
```

Each result is a `ParameterInfo` containing the parameter and all of its
`ParameterRegistration` records. Convenience properties provide the qualified
parameter name, local name, and owning module from the first, canonical
registration.
Exact-name and shell-style glob selectors are available for qualified
parameter names, module names, and local parameter names. Arbitrary predicates
can be created with `parameter_lambda`. Use `selectors.all` when an API requires
a selector but every parameter should be included.

```python
classifier = selectors.matches_module_name_glob("classifier*")
trainable = selectors.requires_grad

selected = (classifier & trainable).select(model)
selected_parameters = [info.parameter for info in selected]
```

When the metadata is not needed, use `selector.parameters(model)` to iterate
over the `nn.Parameter` objects directly. This can be passed straight to a
single-group optimizer:

```python
import torch

optimizer = torch.optim.Adam(
    selectors.requires_grad.parameters(model),
    lr=1e-3,
)
```

The equivalent function form is `parameters(model, selector)`, imported from
`torchwatcher.parameters`. Omitting the selector returns every parameter.

For multiple optimizer groups, describe each selection with `ParameterGroup`
and resolve them with `build_parameter_groups`:

```python
from torchwatcher.training import ParameterGroup, build_parameter_groups

groups = build_parameter_groups(
    model,
    ParameterGroup(selectors.is_bias, weight_decay=0.0),
)
optimizer = torch.optim.AdamW(
    groups,
    lr=1e-3,
    weight_decay=0.01,
)
```

Parameters not matched by an explicit group are placed in a final group, where
they use the optimizer defaults. Explicit groups must not overlap or be empty.
Pass `include_remaining=False` to intentionally leave unmatched parameters out
of the optimizer.

Shared parameters are returned once. Every owning module and qualified name is
retained in `ParameterInfo.registrations`, while `ParameterInfo.aliases`
provides the qualified names directly. This makes selections safe to use when
constructing optimizer parameter groups.

