# Getting started

Install Python 3.10–3.12 and `pip install 'pyautocausal[notebooks]'` if you need executable notebooks or HTML; use the base package for estimation and notebook generation.

## Verify a simple result

The [README regression example](../README.md#a-regression-with-an-explicit-specification) is exercised by a documentation test. It should estimate 10 regardless of row order. Build explicit specifications when you need control over the outcome, treatment or controls. `control_cols=[]` requests no covariates; `None` requests inference from eligible columns.

## Fit panel data

```python
from pathlib import Path
from pyautocausal import create_panel_graph
from pyautocausal.datasets import minimum_wage

output = Path("output/minimum_wage")
graph = create_panel_graph(output)
graph.fit(df=minimum_wage.copy())
```

Use treatment status at each observation, not an ever-treated indicator. Preserve the original unit/time keys and inspect the cleaning record before interpreting the selected method. See [data requirements](data-requirements.md) and [methods](causal-methods.md).

## Export the executed analysis

```python
from pyautocausal import export_outputs
export_outputs(graph, df=minimum_wage, output_path=output)
```

Notebook execution uses the current Python interpreter by default; it does not replace your user-level Jupyter kernel configuration. A named kernel may be supplied to the notebook runner if intentionally using a different environment. Install the notebook extra in the executing environment.

Inspect node results and execution diagnostics. A run that failed partway is not a completed analysis; exported failure information is for diagnosis. Missing uncertainty or an experimental-method notice must remain visible in downstream reports.

## Reproduce the bundled analyses

From a source checkout:

```bash
python examples/scripts/minimum_wage.py --output /tmp/minimum-wage-results
python examples/scripts/california_prop99.py --output /tmp/california-results
```

These scripts use packaged data, accept an output directory, and do not replace missing values with arbitrary numeric sentinels. They demonstrate usage; observational estimates still require a justified research design.
