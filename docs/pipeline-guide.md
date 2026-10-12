# Building pipelines

`ExecutableGraph` executes named nodes and passes predecessor results to callables. Configure output handling separately from graph structure.

```python
from pyautocausal import ExecutableGraph

def square(number: int) -> int:
    return number ** 2

graph = (ExecutableGraph()
         .create_input_node('number', input_dtype=int)
         .create_node('square', square, predecessors=['number']))
graph.fit(number=3)
assert graph.get('square').get_result_value() == 9
```

Use named functions and return annotations for inspectable, portable notebook exports. `save_node=True` requires a supported return annotation or an explicit `OutputConfig`. Scalar/list JSON outputs and selected matplotlib figures are supported.

## Conditional paths

Every direct successor of a decision node must be classified with `when_true` or `when_false`. Conditions are evaluated once per execution. Skipped paths do not supply argument values. Joins can execute when their required active values are available.

Use explicit argument mappings for alternative predecessors:

```python
graph.create_node('summary', summarize, predecessors={'result': ['left_result', 'right_result']})
```

Exactly one active alternative must provide `result`; ambiguous providers are rejected. Defaults apply only where the callable and graph explicitly allow them. Avoid relying on node insertion order.

Explicit predecessor mappings take precedence over implicit single-argument forwarding. A mapped optional argument cannot also replace a required value supplied by the run context. Misspelled mapped argument names are rejected when binding the node's inputs.

## Composition and lifecycle

`left.merge_with(right, left.get('result') >> right.get('input'))` connects a result to an input node. The merge preserves decision memberships and maps renamed nodes. Merging must not alter the right-hand graph’s behavior.

A graph instance retains its execution state. Call `reset()` deliberately before fitting it again. Incomplete graphs and failed nodes must be inspected rather than treated as valid completed analyses. Execution diagnostics include skipped and failed nodes and available provenance.

## Exports

The notebook exporter records execution diagnostics and relevant run-context values. Exported estimates must match the executed graph. Failed-node information is diagnostic: it must not suggest an unfinished run succeeded. Notebook execution uses a temporary kernel pointing to the current interpreter by default.

Replay preserves supported Python containers, NumPy arrays/scalars, and date/time values. Unsupported objects actually consumed from the run context cause an explicit export error; unused context values are omitted. A single `data_path` loader is supported only for a graph with one external input. Graphs with multiple external inputs must load their inputs separately in the notebook; passing one shared data path is rejected before the notebook is written.

`export_outputs` writes `execution_report.json` with a separate `export` status and stage. Data, visualization, notebook, or HTML failures are recorded and raised to the caller. A completed estimation graph does not imply a successful report export; check this status when inspecting an output directory after a failed export.

When called with `df=`, the helper saves an inspectable CSV and a checked Parquet snapshot (`data.csv.parquet` by default). Notebook replay uses the snapshot to retain leading-zero identifiers, categorical columns, timestamps and the index. Keep that snapshot alongside the notebook. With `data_path=`, the CSV loader must reproduce the executed DataFrame; a detected mismatch raises an error with instructions to use `df=` or an explicit typed loader.

Install `pyautocausal[notebooks]` for execution and HTML output. Source-level graph and notebook examples are tested independently of the installed-wheel checks described in [contributing](../CONTRIBUTING.md).

Notebook exports use `compact=True` by default. Setup and implementation cells retain their executable code, start collapsed in supporting notebook viewers, and are hidden in HTML. Pass `compact=False` to `export_outputs`, `NotebookExporter.export_notebook`, or `export_and_run_to_html` to show all code. Analysis calls and results remain visible. Explicit `node_description` values are preserved; otherwise the callable’s docstring supplies a heading and expandable method documentation.

Use `title="My analysis"` with those export functions to name the notebook and HTML report. Exports include navigation, rich table previews and embedded figures; a node's explicit `display_function` still overrides automatic result display. Table previews format values without changing the underlying result. Routine printed output appears in expandable execution logs; warnings and errors remain visible. The original execution record and workflow appear in expandable appendices and remain in notebook metadata. They describe the original graph run, while displayed results come from the notebook replay.

Executing a notebook to HTML also saves its cell outputs in the `.ipynb`, so results are visible when it is opened later. Execution must succeed before the saved notebook is replaced. Plain `export_notebook` still only generates code; use `export_and_run_to_html` or `export_outputs` to execute it.
