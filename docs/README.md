# PyAutoCausal documentation

These guides describe the 0.1.2 APIs and supported workflows. Start with the [installation instructions](../README.md#installation); the corrected wheel is available from the [GitHub release](https://github.com/pyautocausal/pyautocausal/releases/tag/v0.1.2). Affected results from 0.1.0 or 0.1.1 should be rerun after reading the [correction notes](../CHANGELOG.md).

## Fit and inspect an analysis

- [Getting started](getting-started.md): fit a panel, inspect cleaning and node states, export a report, and replay its notebook.
- [Data requirements](data-requirements.md): column contracts, treatment coding, string IDs, covariates, missing-data policies and sample validation.
- [Methods and limitations](causal-methods.md): estimator support, assumptions, uncertainty, reproducibility and experimental APIs.
- [Panel routing](did_flow.md): how the panel factory selects branches and what its decisions do not establish.
- [Examples](../examples/README.md): run the minimum-wage and California analyses into fresh output directories.
- [Bundled datasets](../pyautocausal/datasets/README.md): public data exports, prepared versus raw panels, and sources.

## Build and extend pipelines

- [Pipeline guide](pipeline-guide.md): construct nodes and branches, bind arguments, merge graphs, inspect failures and reset execution.
- [Function argument mapping](wrapper_functions.md): adapters and argument maps, including notebook-export restrictions.
- [Architecture](design-document.md): implemented modules, execution lifecycle and extension boundaries.
- [Data validation](../pyautocausal/data_validation/README.md): validators, configuration and custom checks.

## Maintain and release the project

- [Contributing](../CONTRIBUTING.md): locked development setup, targeted tests, clean-wheel verification and release steps.
- [Issue triage](issue-triage.md): completed correctness criteria and the remaining enhancements.
- [Changelog](../CHANGELOG.md): corrections that can change estimates and saved output.

The base installation can estimate models and generate notebooks. The `notebooks` extra executes notebooks and produces HTML; `doubleml` enables the separate DoubleML wrapper; `all` includes both. Generated notebooks still require PyAutoCausal and the relevant dependencies in their executing environment.

Model selection and data validation do not establish causal identification. Review assumptions, the retained sample and reported inference status before interpreting an estimate.
