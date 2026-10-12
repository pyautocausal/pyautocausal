# Examples

These scripts fit packaged data and export executed notebook/HTML reports. They require the **`notebooks` extra**, or a source checkout configured with `poetry install --all-extras`. Follow [Contributing](../CONTRIBUTING.md) for the checkout environment or the [GitHub wheel installation guide](../README.md#installation) for the released package.

From the repository root, use fresh output directories:

```bash
poetry run python examples/scripts/minimum_wage.py --output /tmp/minimum-wage-results
poetry run python examples/scripts/california_prop99.py --output /tmp/california-results
```

If using an activated environment with the released wheel instead of Poetry, replace `poetry run python` with `python`. The example scripts live in the source repository; the datasets are included in the installed package.

## What the examples fit

- [Minimum wage](scripts/minimum_wage.py) uses the prepared panel of 500 units over 2003–2007, with `lpop` as a covariate. Staggered adoption and available never-treated units select the corresponding Callaway–Sant’Anna path alongside the event-study regression.
- [California Proposition 99](scripts/california_prop99.py) uses cigarette sales as the outcome, state as the unit and observed treatment status. It renames `state`, `year`, `treated` and `cigsale` to the standard panel columns. The example selects only these design/outcome columns rather than imputing missing covariates with a sentinel. Specify a research-appropriate adjustment model separately if required.

See [bundled datasets](../pyautocausal/datasets/README.md) for preparation details and [methods](../docs/causal-methods.md) for assumptions. Neither example demonstrates causal identification merely by running successfully.

## Inspect and replay

Open `pipeline_execution.html` in the requested output directory. Inspect the cleaning history, model assumptions and inference status before using an estimate. Previously generated reports from versions 0.1.0 and 0.1.1 should be rerun after reading the [correctness release notes](../CHANGELOG.md).

Keep the generated `data.csv.parquet` file with each notebook. It preserves the input's types and identifiers during replay; `data.csv` is an additional copy for inspection. `execution_report.json` records estimation states, inference availability and the separate report-export status.

Run the notebook from its containing directory, with PyAutoCausal and the relevant extras installed. It replays the completed route; rerun the example script to validate and select a route for changed input. See [the output artifact guide](../docs/getting-started.md#inspect-results-and-replay).

The repository includes generated [minimum-wage outputs](outputs/minimum_wage/) and [California outputs](outputs/california_prop99/) for inspection. Generate a separate directory when checking a change so you can compare results without replacing the checked-in reports.
