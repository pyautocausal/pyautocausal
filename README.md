# PyAutoCausal

Python pipelines for fitting and reporting causal-inference models. PyAutoCausal provides data checks, graph execution, panel estimators and notebook exports. Choosing a statistical model does not establish its identifying assumptions: users must justify treatment assignment, covariate choice, parallel trends and the relevant comparison group.

Python **3.10–3.12** is supported. The package remains alpha software. The [0.1.2 correctness release](https://github.com/pyautocausal/pyautocausal/releases/tag/v0.1.2) repairs estimation, graph execution and saved results. Read the [correction notes](CHANGELOG.md) and rerun affected analyses from 0.1.0 or 0.1.1.

Start with the [getting-started guide](docs/getting-started.md), or use the [documentation index](docs/README.md) to find data contracts, method limitations and graph APIs.

## Installation

Install the corrected **0.1.2 wheel from the GitHub release** in an environment using a supported Python version. PyPI publication is separate; the older PyPI versions do not contain these fixes.

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
```

Choose the installation that matches your workflow:

```bash
python -m pip install 'pyautocausal @ https://github.com/pyautocausal/pyautocausal/releases/download/v0.1.2/pyautocausal-0.1.2-py3-none-any.whl'
# Execute exported notebooks and produce HTML:
python -m pip install 'pyautocausal[notebooks] @ https://github.com/pyautocausal/pyautocausal/releases/download/v0.1.2/pyautocausal-0.1.2-py3-none-any.whl'
# Optional partially linear DoubleML model:
python -m pip install 'pyautocausal[doubleml] @ https://github.com/pyautocausal/pyautocausal/releases/download/v0.1.2/pyautocausal-0.1.2-py3-none-any.whl'
# Both optional features:
python -m pip install 'pyautocausal[all] @ https://github.com/pyautocausal/pyautocausal/releases/download/v0.1.2/pyautocausal-0.1.2-py3-none-any.whl'
```

[Release notes and checksums](https://github.com/pyautocausal/pyautocausal/releases/tag/v0.1.2) identify the corrected artifacts and source commit. The downloadable source archive is also installable with pip.

The base package supports estimation, saving results and generating notebook files. Executing notebooks and calling `export_outputs` require `notebooks` or `all`. The `doubleml` extra is only needed for the separate DoubleML wrapper. For a source checkout and tests, see [Contributing](CONTRIBUTING.md).

## A regression with an explicit specification

```python
import pandas as pd
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.estimators import fit_ols

frame = pd.DataFrame({"treat": [0, 0, 1, 1], "y": [1.0, 2.0, 11.0, 12.0]})
spec = create_cross_sectional_specification(frame, control_cols=[])
result = fit_ols(spec).model
assert abs(result.params["treat"] - 10.0) < 1e-8
print(result.summary())
```

This is a deliberately simple regression example; interpreting its coefficient causally requires substantive assumptions. Treatment coding must be explicit: `0` means control and `1` means treated. Row order does not change that meaning. For other binary labels, specify `treatment_value` when constructing a cross-sectional specification.

## A panel pipeline

Install the **`notebooks` or `all` extra** before running this example, which fits the model and executes an HTML report. Use a fresh output directory for each analysis.

```python
from pathlib import Path
from pyautocausal import create_panel_graph, export_outputs
from pyautocausal.datasets import minimum_wage

output = Path("output/minimum_wage")
graph = create_panel_graph(output_dir=output)
graph.fit(df=minimum_wage.copy())
export_outputs(graph, df=minimum_wage, output_path=output)
```

The bundled minimum-wage data has columns `id_unit`, `t`, `treat`, `y` and `lpop`. Treatment is zero before adoption and one afterward. The factory builds a graph whose route depends on treatment timing and comparison-unit availability. [Data requirements](docs/data-requirements.md) explain supported designs and cleaning policies.

Open `output/minimum_wage/pipeline_execution.html` for the report. Keep `pipeline_execution.ipynb`, `data.csv.parquet` and `execution_report.json` together: they provide the replayable analysis, typed input snapshot and diagnostics. See [inspecting results and replay](docs/getting-started.md#inspect-results-and-replay).

## What is supported

- Cross-sectional OLS and explicit panel regression specifications.
- Panel pipelines with simultaneous or staggered treatment adoption, including synthetic DiD and Callaway–Sant’Anna paths.
- Graph composition, conditional execution, persisted results and notebook generation.
- Notebook execution/HTML and the separate DoubleML wrapper through installation extras.
- Experimental uplift APIs. S/T/X-learner uncertainty is not reported as a validated confidence interval; see the [method reference](docs/causal-methods.md).

The pipeline cannot prove unconfoundedness or parallel trends. Its checks and balance diagnostics help identify invalid inputs; they do not certify a research design. Unsupported designs and unavailable inference should be handled explicitly rather than interpreted as successful analyses.

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Data requirements and cleaning](docs/data-requirements.md)
- [Methods, assumptions and limitations](docs/causal-methods.md)
- [Building pipelines](docs/pipeline-guide.md)
- [Function argument mapping](docs/wrapper_functions.md)
- [Examples](examples/README.md)
- [Bundled datasets](pyautocausal/datasets/README.md)
- [Contributing and verification](CONTRIBUTING.md)
- [Issue triage](docs/issue-triage.md)
- [Release notes](CHANGELOG.md)

Interactive CLI nodes, notebooks that run without PyAutoCausal installed, and uniform coefficient-table/plot styling remain [open enhancements](docs/issue-triage.md).

## License and citation

[MIT](LICENSE). Please cite the underlying estimation methods used in your analysis as well as this software. Record the package version, treatment encoding, covariates and data transformations with reported results.
