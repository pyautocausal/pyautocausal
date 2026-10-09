# PyAutoCausal

Python pipelines for fitting and reporting causal-inference models. PyAutoCausal provides data checks, graph execution, panel estimators and notebook exports. Choosing a statistical model does not establish its identifying assumptions: users must justify treatment assignment, covariate choice, parallel trends and the relevant comparison group.

Python **3.10–3.12** is supported. The package remains alpha software. See [corrections in 0.1.2](CHANGELOG.md) before reusing results produced by earlier versions.

## Installation

These instructions target version 0.1.2 or newer. Until the correctness release is published on PyPI, use the development installation below.

```bash
pip install 'pyautocausal>=0.1.2'
# Execute exported notebooks and produce HTML:
pip install 'pyautocausal[notebooks]>=0.1.2'
# Optional partially linear DoubleML model:
pip install 'pyautocausal[doubleml]>=0.1.2'
# Both optional features:
pip install 'pyautocausal[all]>=0.1.2'
```

For the development version:

```bash
git clone https://github.com/pyautocausal/pyautocausal.git
cd pyautocausal
poetry install --all-extras
poetry run pytest tests/ -m 'not scientific'
```

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

## What is supported

- Cross-sectional OLS and explicit panel regression specifications.
- Panel pipelines with simultaneous or staggered treatment adoption, including synthetic DiD and Callaway–Sant’Anna paths.
- Graph composition, conditional execution, persisted results and notebook generation.
- Notebook execution/HTML and the separate DoubleML wrapper through installation extras.
- Experimental uplift APIs. S/T/X-learner uncertainty is not reported as a validated confidence interval; see the [method reference](docs/causal-methods.md).

The pipeline cannot prove unconfoundedness or parallel trends. Its checks and balance diagnostics help identify invalid inputs; they do not certify a research design. Unsupported designs and unavailable inference should be handled explicitly rather than interpreted as successful analyses.

## Documentation

- [Getting started](docs/getting-started.md)
- [Data requirements and cleaning](docs/data-requirements.md)
- [Methods, assumptions and limitations](docs/causal-methods.md)
- [Building pipelines](docs/pipeline-guide.md)
- [Function argument mapping](docs/wrapper_functions.md)
- [Examples](examples/README.md)
- [Contributing and verification](CONTRIBUTING.md)
- [Issue triage](docs/issue-triage.md)
- [Release notes](CHANGELOG.md)

## License and citation

[MIT](LICENSE). Please cite the underlying estimation methods used in your analysis as well as this software. Record the package version, treatment encoding, covariates and data transformations with reported results.
