# Analysis

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

Forward-only analysers can drive a complete dataset or dataloader pass with
`run` for convenience. This resets and temporarily enables the analyser,
puts the watched model in evaluation mode, moves inputs to the model's device,
and restores the previous states afterward:

```python
results = rank_analyser.run(watched_model, loader)
```

When passing a dataset instead, provide a batch size. Use `prepare_inputs` for
custom batch structures.

```python
results = rank_analyser.run(
    watched_model,
    dataset,
    batch_size=64,
    prepare_inputs=lambda batch, device: batch["image"].to(device),
)
```

You can however just write or use your own forward (or forward-backward) logic
if you desire.

## Relational analysers

A regular `Analyser` processes each watched node independently. A
`RelationalAnalyser` instead combines observations from pairs of watched nodes,
possibly belonging to different models. It must therefore distinguish each
model or activation stream and know which forwards belong to the same input
batch.

Do not pass a relational analyser directly to `interject_by_match`. Create one
source-bound interjection per model with `watch(source)`. Here
`LinearCKAAnalyser` is a concrete example of a relational analyser:

```python
from torchwatcher.analysis import LinearCKAAnalyser

analyser = LinearCKAAnalyser()

watched_student = interject_by_match(
    student,
    student_selector,
    analyser.watch("student"),
)
watched_teacher = interject_by_match(
    teacher,
    teacher_selector,
    analyser.watch("teacher"),
)
```

The source names identify the models even when their graph nodes have identical
names. `run` accepts a mapping using those same source names and runs every
model on each batch inside one relational transaction:

```python
results = analyser.run(
    {
        "student": watched_student,
        "teacher": watched_teacher,
    },
    loader,
)
```

With one watched source, the default comparison relates that source to itself.
With two sources, the default relates the first source to the second. Explicit
comparisons can restrict the comparison or configure more than two sources:

```python
analyser = LinearCKAAnalyser(
    comparisons=[
        ("student", "teacher"),
        ("student", "student"),
    ]
)
```

For each configured source pair, the analyser processes all ordered pairs 
of their selected layers (Cartesian product). Every source must observe a 
stable set of layers across batches, and paired models must receive corresponding 
samples. A source transform can select a tensor from structured layer output:

```python
analyser.watch("student", transform=lambda output: output.hidden_state)
```

For a custom loop, use `batch()` directly rather than `run`:

```python
for inputs, _ in loader:
    with analyser.batch():
        watched_student(inputs)
        watched_teacher(inputs)
```

To implement a new relational analysis, subclass `RelationalAnalyser` and
implement `process_batch_relation(relation, states, working_results)`. The two
`AnalyserState` objects contain the paired node observations, while
`working_results` contains the value returned for that relation on the previous
batch. `prepare_state` can preprocess each observation, and
`validate_batch_states` can validate the complete transaction before any
relation is accumulated.

## Comparing representations with CKA

`LinearCKAAnalyser` is a relational analyser: it combines corresponding
activations rather than processing each watched layer independently. Give each
model or activation stream a source name with `watch`, then pass the watched
models to `run` under those same names.

```python
from torchwatcher.analysis import LinearCKAAnalyser
from torchwatcher.interjection import interject_by_match, node_selector

cka = LinearCKAAnalyser()

watched_a = interject_by_match(
    model_a,
    node_selector.Activations.is_relu,
    cka.watch("model_a"),
)
watched_b = interject_by_match(
    model_b,
    node_selector.Activations.is_relu,
    cka.watch("model_b"),
)

cka.run(
    {"model_a": watched_a, "model_b": watched_b},
    loader,
)

result = cka.result("model_a", "model_b")
print(result.row_names)
print(result.column_names)
print(result.values)
```

`run` sends each input batch to both models inside one relational transaction,
so their samples correspond automatically. Layers can have different feature
dimensions. Activations are flattened after the batch dimension and only
batch-sized Gram matrices are retained temporarily.

To compare layers within one model, register only one source:

```python
cka = LinearCKAAnalyser()
watched_model = interject_by_match(
    model,
    node_selector.Activations.is_relu,
    cka.watch("model"),
)

cka.run(watched_model, loader)

result = cka.result()  # square layer-by-layer matrix
```

For nodes which return tuples or other structures, select the representation
with a source transform:

```python
cka.watch("model", transform=lambda output: output[0])
```

Pass `debiased=True` to use the unbiased HSIC estimator; each batch must then
contain at least four samples.

Runnable marimo notebooks are available for the [same-model and cross-model
workflow](https://github.com/feature-importance/torchwatcher/blob/main/examples/cka.py) and for a [numerical comparison with torch-cka
0.21](https://github.com/feature-importance/torchwatcher/blob/main/examples/cka_torch_cka_comparison.py).

