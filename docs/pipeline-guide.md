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

## Composition and lifecycle

`left.merge_with(right, left.get('result') >> right.get('input'))` connects a result to an input node. The merge preserves decision memberships and maps renamed nodes. Merging must not alter the right-hand graph’s behavior.

A graph instance retains its execution state. Call `reset()` deliberately before fitting it again. Incomplete graphs and failed nodes must be inspected rather than treated as valid completed analyses. Execution diagnostics include skipped and failed nodes and available provenance.

## Exports

The notebook exporter records execution diagnostics and relevant run-context values. Exported estimates must match the executed graph. Failed-node information is diagnostic: it must not suggest an unfinished run succeeded. Notebook execution uses a temporary kernel pointing to the current interpreter by default.

Install `pyautocausal[notebooks]` for execution and HTML output. Source-level graph and notebook examples are tested independently of the installed-wheel checks described in [contributing](../CONTRIBUTING.md).

For a concise notebook view, pass `compact=True` to `NotebookExporter.export_notebook` or `export_and_run_to_html`. Implementation cells retain their code but carry standard Jupyter hidden-source metadata; execution, provenance and diagnostics remain visible. Explicit `node_description` values are preserved; otherwise the callable’s docstring supplies a default description.
