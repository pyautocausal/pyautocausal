# Wrapper functions and notebook export

Graph node names do not have to match function parameter names. For a simple rename, use an explicit predecessor mapping; a wrapper is unnecessary:

```python
from pyautocausal import ExecutableGraph


def mean_value(values: list) -> float:
    return sum(values) / len(values)


graph = ExecutableGraph().create_input_node("observations", list)
graph.create_node("mean", mean_value, predecessors={"values": "observations"})
graph.fit(observations=[1, 2, 3])
assert graph.get("mean").get_result_value() == 2.0
```

The mapping direction is **function parameter → predecessor node name**. See the [pipeline guide](pipeline-guide.md) for multiple inputs and alternative predecessors.

## Exposing a thin wrapper's target

Existing code may use wrappers to rename arguments. `expose_in_notebook` attaches metadata telling the exporter which target to show and call. Its `arg_mapping` direction is **wrapper parameter → target parameter**.

Save this as `wrapper_demo.py` and run it with the [notebook extra](../README.md#installation) installed:

```python
from pathlib import Path
import pandas as pd
from pyautocausal import ExecutableGraph, export_outputs
from pyautocausal.persistence.notebook_decorators import expose_in_notebook


def mean_outcome(df: pd.DataFrame, scale: float = 1.0) -> float:
    """Return a scaled mean outcome."""
    return float(df["y"].mean()) * scale


@expose_in_notebook(
    target_function=mean_outcome,
    arg_mapping={"data": "df", "multiplier": "scale"},
)
def mean_action(data: pd.DataFrame, multiplier: float = 1.0) -> float:
    return mean_outcome(df=data, scale=multiplier)


data = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
output = Path("wrapper-output")
graph = ExecutableGraph().configure_runtime(output_path=output)
graph.create_input_node("data", pd.DataFrame)
graph.create_node("mean", mean_action, predecessors=["data"], display_function=print)
graph.fit(data=data)
assert graph.get("mean").get_result_value() == 2.0
export_outputs(graph, df=data, output_path=output)
```

During normal graph execution, `mean_action` runs unchanged. In the notebook, the exporter resolves `mean_outcome`, includes its source or a supported module-backed definition, and calls it with mapped arguments. It does **not** promise to emit both function bodies or call the original wrapper.

Use this decorator only for transparent argument adapters. Extra filtering, transformations, side effects or different defaults in a wrapper may be lost when the notebook calls the target directly. For such logic, export the undecorated action or make the transformation a separate graph node.

## Making exports portable

Keep exposed targets at module scope with an importable module/name, and make their source inspectable. Nested target functions cannot be resolved through the stored dotted reference. Keep required imports at module scope or inside the function. The exporter analyzes imports, but it is not a general serializer for closures, global application state or unavailable dependencies.

Run the generated notebook in the intended environment to verify replay. Built-in package helpers may remain imports, so install the same PyAutoCausal version and dependencies. The `df=` export route saves a checked Parquet snapshot; retain it beside the notebook. Notebook replay, run-context support, compact output and failure reporting are described in [exporting a reproducible report](pipeline-guide.md#exporting-a-reproducible-notebook-and-html-report).

The package also uses `make_transformable(...).transform(...)` internally to adapt callable signatures. For user-defined graphs, explicit predecessor mappings are usually sufficient and avoid depending on that internal adapter pattern.
