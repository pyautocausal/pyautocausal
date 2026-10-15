# Getting started

Use Python 3.10–3.12 and install the corrected 0.1.2 GitHub wheel with the commands in the [README](../README.md#installation). Choose the **`notebooks` extra** to complete this guide through HTML export. The base wheel supports estimation and notebook generation; it does not execute notebooks. PyPI publication is separate; older PyPI versions lack these correctness fixes.

## Verify a simple result

The [README regression example](../README.md#a-regression-with-an-explicit-specification) is exercised by a documentation test. It should estimate 10 regardless of row order. Build explicit specifications when you need control over the outcome, treatment or controls. `control_cols=[]` requests no covariates; `None` requests inference from eligible columns.

## Fit panel data

```python
from pathlib import Path
from pyautocausal import create_panel_graph
from pyautocausal.datasets import minimum_wage

output = Path("output/minimum_wage")
data = minimum_wage.copy()
graph = create_panel_graph(
    output_dir=output,
    missing_strategy="drop_rows",
    max_missing_fraction=0.1,
)
graph.fit(df=data)
```

Use a fresh output directory per analysis. This example uses the prepared `minimum_wage` dataset and its numeric `lpop` covariate. Use treatment status at each observation, not an ever-treated indicator. The raw `mpdta` export needs additional preparation; see [bundled datasets](../pyautocausal/datasets/README.md).

The defaults shown allow at most 10% missing values per column before dropping incomplete rows. `missing_strategy="reject"` refuses missing values. Cleaning records sample changes and revalidates the remaining design. See [data requirements](data-requirements.md) for the full policy and [panel routing](did_flow.md) for the selected analysis paths.

Inspect the run before exporting:

```python
report = graph.execution_report()
print({name: node["state"] for name, node in report["nodes"].items()})
cleaned = graph.get("panel_cleaned_data").get_result_value()
print(cleaned.attrs["cleaning_metadata"])
```

`completed` means a node ran successfully. `passed` means a branch was skipped, not that an estimator was fitted. `failed`, `pending` or `running` states require investigation; a partially failed graph is not a completed analysis. See [methods and limitations](causal-methods.md) before interpreting fitted coefficients or uncertainty.

## Export the executed analysis

```python
from pyautocausal import export_outputs
export_outputs(graph, df=data, output_path=output, compact=True)
```

`export_outputs` generates a notebook and executes it to create HTML, so this call requires the notebook extra. Successful execution also saves cell outputs in the `.ipynb`. The default `compact=True` hides implementation inputs in HTML while retaining notebook code, results and diagnostics. Supply the same input frame used for fitting; `df` and `data_path` are mutually exclusive. A CSV supplied through `data_path` is rejected if inference of its types or values changes the fitted input.

Notebook execution uses the current Python interpreter by default, without replacing user-level Jupyter kernel configuration. A named kernel can be selected through the lower-level notebook runner; see the [pipeline guide](pipeline-guide.md). Install the relevant extras in that kernel's environment.

## Inspect results and replay

The output directory contains:

| Artifact | Purpose |
| --- | --- |
| `pipeline_execution.html` | Executed report to open in a browser. |
| `pipeline_execution.ipynb` | Replay code and saved outputs from executing the completed analysis path. |
| `data.csv.parquet` | Typed input snapshot used by the notebook; preserves IDs, timestamps and column types. |
| `data.csv` | Additional copy for inspection, not the notebook's replay input. |
| `execution_report.json` | Node states, analysis provenance, recorded transformations and separate export status. |
| `text/` and `plots/` | Method-specific saved summaries/figures and `text/pipeline_visualization.md`. |

Check the saved export status separately from graph fitting:

```python
import json

saved = json.loads((output / "execution_report.json").read_text())
assert saved["export"]["state"] == "completed"
print(saved["export"]["artifacts"])
```

Export failures raise an exception and record their failing stage in this JSON file. Fix the reported problem and rerun the export; the existence of an HTML or notebook file alone does not establish success. Missing uncertainty or an experimental-method notice must remain visible in downstream reports.

Keep the output directory together when moving or sharing it. Run the notebook with its own directory as the working directory and with PyAutoCausal and the relevant dependencies installed. Its code replays the completed path; to analyze changed data, fit a new graph so branch selection and validation run again. Preserve package versions and estimator-specific seeds/settings from the report rather than assuming a notebook is independent of its environment.

### Generate a notebook without executing HTML

With the base package, the lower-level exporter can generate a notebook from the fitted `graph` above:

```python
from pyautocausal.persistence.notebook_export import NotebookExporter

output.mkdir(parents=True, exist_ok=True)
data.to_parquet(output / "input.parquet", index=True)
NotebookExporter(graph).export_notebook(
    str(output / "analysis.ipynb"),
    data_path="input.parquet",
    loading_function="pd.read_parquet",
    compact=True,
)
```

This creates `analysis.ipynb` and relies on the typed input you saved; it does not create an HTML report or mark HTML export complete. Notebook execution still requires the notebook extra. Prefer `export_outputs` for the checked input snapshot and complete report bundle.

## Common setup problems

- **Missing notebook or DoubleML dependency:** install the corresponding extra using the [GitHub wheel command](../README.md#installation) in the Python environment that executes the code.
- **An already-executed graph rejects another fit:** create a new graph for a new analysis, or use the explicit [reset lifecycle](pipeline-guide.md).
- **Cleaning rejects the panel:** inspect the error, unit/time keys, treatment pattern and missing-data policy. Increasing a threshold does not repair a design that loses its comparison group or required observations.
- **Notebook cannot find its input:** move the typed snapshot with the notebook and execute from that output directory.

## Reproduce the bundled analyses

From a [configured source checkout](../CONTRIBUTING.md), use the Poetry environment:

```bash
poetry run python examples/scripts/minimum_wage.py --output /tmp/minimum-wage-results
poetry run python examples/scripts/california_prop99.py --output /tmp/california-results
```

These scripts use packaged data, accept an output directory, and do not replace missing values with arbitrary numeric sentinels. They demonstrate usage; observational estimates still require a justified research design. See the [example guide](../examples/README.md) for report contents and checked-in examples.
