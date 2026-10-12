# Examples

Run from a source checkout with the package installed. Use `pyautocausal[notebooks]` for HTML export.

```bash
python examples/scripts/minimum_wage.py --output /tmp/minimum-wage-results
python examples/scripts/california_prop99.py --output /tmp/california-results
```

Minimum wage uses the packaged, prepared panel. California Proposition 99 uses cigarette sales as the outcome, state as the unit and observed treatment status. The example deliberately selects only required design/outcome columns rather than imputing missing covariates with a sentinel. Specify a research-appropriate adjustment model separately if required.

Examples write to the requested output directory. Previously generated reports from versions 0.1.0 and 0.1.1 should be rerun after reading the correctness release notes.

Keep the generated `data.csv.parquet` file with each notebook. It preserves the input's types and identifiers during replay; `data.csv` is an additional copy for inspection. `execution_report.json` records estimation states, inference availability and the separate report-export status.

New exports save executed notebook outputs, embed plots and table previews, and collapse setup code and technical appendices. The example scripts supply analysis titles. Pass `compact=False` to `export_outputs` to show all implementation code, or `title="My analysis"` to set a custom title. Rerun the scripts to regenerate older reports with this layout.
