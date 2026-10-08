# Data requirements and cleaning

Panel factories expect `id_unit`, `t`, `treat` and `y`. Cross-sectional factories require `treat` and `y`. Use a copy of the input when developing an analysis, and retain the original data and the transformation record.

- `id_unit` identifies the same observational unit across periods; numeric and string IDs are supported. IDs are labels, not regression covariates.
- `t` identifies an ordered period. Numeric periods and parseable timestamps are supported. Distinct intraday timestamps remain distinct. Time standardization creates chronological indices 1…T and records the mapping to original values.
- `treat` is observed treatment status, coded 0/1. For adoption designs it is zero before adoption and one afterward, including for units eventually treated. An ever-treated flag is not a valid substitute.
- `y` is a finite numeric outcome. Outcome transformations and their causal interpretation are the caller’s responsibility.
- Additional columns may be inferred as controls. Use an explicit specification when control selection matters; an empty control list means no controls. Avoid post-treatment controls unless the design specifically justifies them.

## Missing data

Factory defaults are `missing_strategy='drop_rows'` and `max_missing_fraction=0.1`. The threshold is the maximum missing fraction allowed in each checked column before cleaning. Make the choice explicitly for analyses where dropping observations changes the target population.

```python
from pathlib import Path
from pyautocausal import create_panel_graph

graph = create_panel_graph(
    Path('output/panel'),
    missing_strategy='reject',
    max_missing_fraction=0.1,
)
```

`reject` refuses missing values. `drop_rows` permits dropping within the configured threshold; set the threshold deliberately between 0 and 1 if a different policy is justified. Increasing it is permission to remove more observations, not a claim that the missingness is ignorable. Factory pipelines do not fill missing outcomes or treatments with fabricated numbers.

Cleaning metadata lives in `cleaned_frame.attrs['cleaning_metadata']` and is included in graph provenance where available. It records before/after row and unit counts, dropped rows, policy settings and transformations. Time transformations include original values and standardized periods.

## Design validation

Panel unit/time keys must be unique. Missing required keys, invalid treatment coding, non-finite outcomes, and loss of a required treatment/comparison group are errors. Cleaning failures propagate rather than allowing the original malformed data through silently. Repeated cross-sectional rows are not automatically discarded merely because their observed values coincide.

Synthetic DiD additionally requires a complete unit-by-period grid, donors, pretreatment information and supported simultaneous adoption. Other methods have different requirements; a globally valid DataFrame is not necessarily valid for every estimator. Inspect the chosen route and [method limitations](causal-methods.md).

## Custom cleaning

`AutoCleaner` exposes individual checks and transformation metadata. Manually chosen imputation is separate from the factory’s drop/reject policy. Revalidate causal design after any transformation that changes units, periods, outcome or treatment. Keep the cleaning record with the analysis; it is part of the definition of the estimation sample.
