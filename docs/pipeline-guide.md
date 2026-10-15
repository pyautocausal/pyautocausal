# Building pipelines

`ExecutableGraph` is a directed graph of named input, action and decision nodes. `fit()` sets external inputs and executes the selected path. For the provided causal pipelines, start with [getting started](getting-started.md); this guide covers custom graphs and exports.

## Actions and argument binding

Use an explicit mapping from **function parameter** to **predecessor node name** when their names differ:

```python
from pyautocausal import ExecutableGraph


def square(number: int) -> int:
    """Square the supplied integer."""
    return number ** 2


graph = (
    ExecutableGraph()
    .create_input_node("input_number", input_dtype=int)
    .create_node("squared", square, predecessors={"number": "input_number"})
)
graph.fit(input_number=3)
assert graph.get("squared").get_result_value() == 9
```

Create predecessors before their successors. `fit()` requires exactly the names of external input nodes; connected inputs in a merged graph are supplied internally. Input values are checked against `input_dtype`.

A list such as `predecessors=["number"]` forwards output names. There is also a shorthand that forwards a lone unbound value to a callable's single required parameter. Explicit mappings are clearer for composition and take precedence over that shorthand. An unknown mapped parameter raises an error unless the callable accepts `**kwargs`.

Values from `RunContext(metadata={...})`, passed to `configure_runtime(run_context=...)`, fill parameters not supplied by predecessors. Function defaults apply after that. An explicitly supplied `None` is a value; it does not trigger a context/default fallback.

Use named functions with inspectable source for notebook exports. A `node_description` overrides the function's docstring in descriptions; otherwise the docstring is used.

## Decisions and alternative predecessors

Register every direct successor of a decision using `when_true` or `when_false`. A decision evaluates its condition once per run, forwards its input values to the selected path, and marks the other path as skipped (`passed` in diagnostics). Its forwarded value is not the boolean condition result.

```python
from pyautocausal import ExecutableGraph


def is_positive(number: int) -> bool:
    return number > 0


def double(number: int) -> int:
    return 2 * number


def negate(number: int) -> int:
    return -number


def add_one(value: int) -> int:
    return value + 1


branches = ExecutableGraph().create_input_node("number", int)
branches.create_decision_node("positive", is_positive, predecessors=["number"])
branches.create_node("left", double, predecessors=["positive"])
branches.create_node("right", negate, predecessors=["positive"])
branches.when_true("positive", "left").when_false("positive", "right")
branches.create_node(
    "answer", add_one, predecessors={"value": ["left", "right"]}
)
branches.fit(number=4)
assert branches.get("answer").get_result_value() == 9
assert branches.get("right").is_passed()

branches.reset().fit(number=-4)
assert branches.get("answer").get_result_value() == 5
```

The list inside the mapping declares alternative providers for one parameter. Arrange branches so exactly one provides the value. Two completed providers raise an ambiguity error, even if their values are equal; skipped providers supply nothing. Joins wait for completed or skipped predecessors. Readiness is recomputed after each execution, including when a skip makes a join ready. A missing required argument raises an error if no predecessor or run-context value can supply it.

## Composition and repeated runs

`merge_with` mutates the left graph and returns it; the right graph remains independent. Wire a left-hand node to a right-hand **input** node with `>>`. Both graphs must have all nodes in `PENDING` state, so merge before execution or call `reset()` first.

The following continues the `square` and `add_one` definitions above:

```python
left = (
    ExecutableGraph()
    .create_input_node("number", int)
    .create_node("squared", square, predecessors=["number"])
)
right = (
    ExecutableGraph()
    .create_input_node("value", int)
    .create_node("answer", add_one, predecessors=["value"])
)
left.merge_with(right, left.get("squared") >> right.get("value"))
left.fit(number=3)
assert left.get("answer").get_result_value() == 10
```

Name collisions in the copied graph receive suffixes such as `_1`. Branch registrations and argument bindings follow the renamed nodes. Unwired right-hand inputs remain external inputs in the combined graph. Configure the combined graph's output handler and run context on the left graph.

Completed nodes do not automatically rerun on another `fit()`. Call `reset()` before a new analysis: it clears node results, inputs, errors, counts and decisions while retaining graph structure and runtime configuration. It cannot run while a node is running.

`to_yaml(path)` and `ExecutableGraph.from_yaml(path)` round-trip the graph definition, including branch registrations, bindings and descriptions. They do not store fitted results or runtime configuration. Configure output handling and run context on a restored graph before execution. Importable function definitions and their dependencies must remain available.

## Saving values and diagnosing failures

Configure a destination and opt individual nodes into saving:

```python
from pathlib import Path
from pyautocausal import ExecutableGraph, OutputConfig, OutputType

saved = ExecutableGraph().configure_runtime(output_path=Path("graph-output"))
saved.create_input_node("number", int)
saved.create_node(
    "squared", square, predecessors=["number"], save_node=True,
    output_config=OutputConfig(output_type=OutputType.JSON, output_filename="square"),
)
saved.fit(number=3)  # Writes square.json containing 9 and execution_report.json.
```

If `output_config` is omitted, `save_node=True` requires a supported return annotation to infer the output format. JSON supports scalars and lists as well as dictionaries; non-finite saved values are rejected. For plots, return the `matplotlib.figure.Figure` you intend to save and select `OutputType.PNG`; the handler saves that figure. Input nodes are not saved as node outputs.

`graph.execution_report()` includes every node's state, execution count and recorded error, plus available analysis settings, sample/cleaning information and package versions. With a configured output handler it is also written as `execution_report.json`. Skipped nodes have state `passed`; they are expected in conditional graphs.

Execution errors propagate. A cycle, stalled graph or previously failed node raises `IncompleteExecutionError` with relevant diagnostics. Node failures record the underlying error; output-saving failures are marked with phase `saving_output`. Correct the cause and call `reset()` before retrying. `is_execution_finished()` only checks that every node is terminal, so a failed node can still make it return `True`; inspect failures or rely on a successful `fit()` return when determining success.

## Exporting a reproducible notebook and HTML report

Install the [notebook extra](../README.md#installation). Save this example as a Python file and run it so the exporter's source inspection can read `mean_outcome`:

```python
from pathlib import Path
import pandas as pd
from pyautocausal import ExecutableGraph, export_outputs


def mean_outcome(data: pd.DataFrame) -> float:
    """Compute the mean outcome in the input table."""
    return float(data["y"].mean())


data = pd.DataFrame({"id_unit": ["001", "01", "1"], "y": [1.0, 2.0, 3.0]})
output = Path("report-output")
report_graph = ExecutableGraph().configure_runtime(output_path=output)
report_graph.create_input_node("df", pd.DataFrame)
report_graph.create_node(
    "mean", mean_outcome, predecessors={"data": "df"}
)
report_graph.fit(df=data)
export_outputs(report_graph, df=data, output_path=output, title="Mean outcome analysis")
```

Numeric and text return values, pandas tables and figures are displayed automatically in the notebook and HTML. Supported fitted specifications receive result summaries. A node's explicit `display_function` overrides automatic display; for example, `display_function=print` uses plain text. Returned summaries remain visible, while routine printed output appears in expandable execution logs. Warnings and errors remain visible.

The helper writes:

| File | Purpose |
| --- | --- |
| `data.csv` | Inspectable copy of the supplied DataFrame. |
| `data.csv.parquet` | Checked snapshot used for replay, retaining identifiers, dtypes and index. |
| `text/pipeline_visualization.md` | Graph structure and execution states. |
| `pipeline_execution.ipynb` | Notebook of the completed analysis path, including saved execution outputs. |
| `pipeline_execution.html` | Results from executing that notebook. |
| `execution_report.json` | Original execution diagnostics plus export status. |

Pass the same DataFrame used for fitting and keep the Parquet snapshot beside the notebook. `df=` and `data_path=` are mutually exclusive. The `data_path=` convenience route loads CSV and rejects a detected mismatch with the executed DataFrame. Use `df=` or `NotebookExporter` with an explicit typed loader when CSV inference changes values or dtypes.

Exports replay completed nodes and record failed/skipped nodes as diagnostics. They do not rerun decision selection: changed inputs may require fitting the original graph again. Runtime settings actually consumed by completed nodes are replayed for supported Python containers, NumPy scalars/arrays and date/time values. Unsupported consumed objects produce an explicit export error; unused settings are omitted from executable cells.

`NotebookExporter.export_notebook()` writes without executing. `export_and_run_to_html()` also executes, using a temporary Jupyter kernel for the current Python interpreter unless `kernel_name` is supplied. The execution stage saves cell outputs back to the `.ipynb` only after all cells succeed, so results remain visible when the notebook is opened later. A shared `data_path`/loader cannot replay multiple external inputs; export without it and fill each input cell separately.

Notebook exports use `compact=True` by default. Setup and implementation code remains executable, starts collapsed in supporting notebook viewers, and is hidden in HTML. Analysis calls and results stay visible. Pass `compact=False` to `export_outputs`, `NotebookExporter.export_notebook` or `export_and_run_to_html` to show all code, or `title="My analysis"` to name the notebook and HTML report. A `node_description` supplies the step heading and expandable method documentation; otherwise the callable's docstring is used.

Reports include navigation, bounded table previews and embedded figures. Table formatting does not change the underlying result. The original execution record and workflow appear in expandable appendices; execution diagnostics also remain in notebook metadata. These describe the original graph run, while displayed results come from the notebook replay. Package helpers and dependencies may still be imported: exported notebooks are not package-free standalone programs. See [wrapper functions](wrapper_functions.md) for argument adapters.

A report failure is separate from an estimation failure. `export_outputs` records `export.state` (`running`, `failed` or `completed`) and `export.stage` (`data`, `visualization`, `notebook` or `html`). Errors propagate; a completed graph alone does not mean the report finished exporting.
