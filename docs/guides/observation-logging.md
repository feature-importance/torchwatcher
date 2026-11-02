# Logging observations

`torchwatcher.logging` defines a storage-independent observation format for
parameter tracking and analyser results. An observation has a stable
hierarchical path, a value, optional training context, and descriptive
metadata. `ObservationPath` is the public type alias for its canonical
immutable `tuple[str, ...]` representation. `Observation` also accepts a bare
string or another sequence of strings and normalizes it to that form. A bare
string is always one component; dots and slashes are not interpreted as
separators:

```python
from torchwatcher.logging import (
    Observation,
    ObservationContext,
    ObservationPath,
)

path: ObservationPath = (
    "parameters",
    "classifier.weight",
    "gradient",
    "l2",
)

observation = Observation(
    path=path,
    value=gradient_norm,
    context=ObservationContext(
        global_step=100,
        epoch=2,
        batch=20,
        event="backward",
    ),
)
```

Storage implementations conform to the `ObservationSink` protocol, which
accepts batches of observations and provides explicit `flush` and `close`
operations. Observation values remain unrestricted so sinks can support both
small scalar statistics and large tensor representations.

`InMemorySink` retains storage-safe snapshots for small experiments and tests.
Tensor values are detached, copied to CPU, and cloned when written.
`CompositeSink` sends the same observations to multiple destinations:

```python
from torchwatcher.logging import CompositeSink, InMemorySink

memory = InMemorySink()
sink = CompositeSink(memory, another_sink)
sink.write([observation])
```

`flatten_observation` turns nested mapping results into individual observations
by extending their paths. `normalise_observation` and `normalise_value` provide
the recursive tensor snapshotting used by storage implementations.

Any completed analyser result can be converted into observations with
`analyser_observations`:

```python
from torchwatcher.analysis import analyser_observations
from torchwatcher.logging import ObservationContext

result = analyser.run(model, loader)
sink.write(analyser_observations(
    analyser,
    result=result,
    path=("analysers", "feature-statistics"),
    context=ObservationContext(event="evaluation"),
))
```

If `result` is omitted, the adapter calls `analyser.to_dict()`. Nested analyser
results are flattened below the supplied path, and the analyser's qualified
class name is included in observation metadata. When no path is supplied, the
default is `("analysers", type(analyser).__name__)`.

## Dataframes

Observation collections can be converted to long-form tabular records without
an additional dependency, or directly to a pandas dataframe when TorchWatcher
is installed with the `dataframe` extra:

```bash
python -m pip install -e ".[dataframe]"
```

```python
from torchwatcher.logging import (
    observation_records,
    observations_to_dataframe,
)

records = list(observation_records(sink.observations))
frame = observations_to_dataframe(sink.observations)
```

Each row contains `path`, `global_step`, `epoch`, `batch`, `event`, `value` and
`metadata`. Scalar tensors become Python scalars; non-scalar tensors are
detached and copied to CPU. Pass `expand_values=True` to emit one row per tensor
element with its coordinate in a `value_index` column:

```python
expanded = observations_to_dataframe(
    sink.observations,
    expand_values=True,
)
```

The resulting long-form dataframe can be reshaped when observations have a
suitable unique index:

```python
wide = frame.pivot(
    index=["global_step", "epoch", "batch", "event"],
    columns="path",
    values="value",
)
```

