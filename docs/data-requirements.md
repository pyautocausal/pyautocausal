# Data requirements and cleaning

Panel factories expect one row per unit and period, with columns `id_unit`, `t`, `treat` and `y`. Cross-sectional factories require `treat` and `y`. Retain the original data and the transformation record; cleaning returns a copy.

- `id_unit` identifies the same observational unit across periods; numeric and string IDs are supported. IDs are labels, not regression covariates.
- `t` identifies an ordered period. Numeric periods and parseable timestamps are supported. Distinct intraday timestamps remain distinct. Time standardization creates chronological indices 1…T and records the mapping to original values.
- `treat` is observed treatment status, coded 0/1. For adoption designs it is zero before adoption and one afterward, including for units eventually treated. An ever-treated flag is not a valid substitute.
- `y` is a finite numeric outcome. Outcome transformations and their causal interpretation are the caller’s responsibility.
- Additional columns may be inferred as controls. Use an explicit specification when control selection matters; an empty control list means no controls. Avoid post-treatment controls unless the design specifically justifies them.

Rename source columns before calling a factory. Direct specification constructors instead accept names such as `outcome_col`, `treatment_cols`, `unit_col` and `time_col`. A specification accepts exactly one treatment column. Preserve label types while loading files: for example, `pd.read_csv(path, dtype={'id_unit': 'string'})` keeps IDs such as `001`, `01` and `1` distinct.

For panel data, encode exposure at each observation, not eventual membership in the treated group. This small frame illustrates a treated unit adopting in the third period and a never-treated donor:

```python
import pandas as pd

periods = pd.to_datetime([
    '2026-01-01 09:00', '2026-01-01 12:00', '2026-01-01 15:00'
])
panel = pd.DataFrame({
    'id_unit': pd.Series(['001'] * 3 + ['01'] * 3, dtype='string'),
    't': list(periods) * 2,
    'treat': [0, 0, 1, 0, 0, 0],
    'y': [2., 3., 14., 1., 2., 3.],
})
assert not panel.duplicated(['id_unit', 't']).any()
```

The example demonstrates input structure, not a sample size adequate for inference. A chronological index counts observed periods; it does not measure elapsed calendar time. If periods are irregularly spaced, define the appropriate observation interval before interpreting event-time effects.

## Control selection

Default causal cleaning preserves numeric control dtypes, including low-cardinality and binary variables. It does not convert them to categorical dtype based on their number of distinct values. With `control_cols=None`, specifications infer numeric controls after excluding outcome, treatment, structural identifiers and periods. The conventional design-metadata names `post`, `relative_time` and `treatment_time` are also excluded from automatic inference. An explicitly supplied control list takes precedence; `[]` requests no controls.

Cross-sectional inference excludes `id_unit` and `t`, even when they are numeric. For other column names, pass `unit_col` and `time_col` to `create_cross_sectional_specification`. Relabeling an identifier from numbers to strings should not change the regression's adjustment variables.

Factory functions do not expose a `control_cols` argument: pass only the design columns and intended numeric controls in their input, or use the specification/estimator API to select controls explicitly. For example:

```python
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.estimators import fit_ols

sample = pd.DataFrame({
    'id_unit': ['001', '002', '003', '004'],
    'treat': [1, 0, 1, 0],
    'y': [12., 2., 14., 4.],
    'baseline': [2., 2., 4., 4.],
    'description': ['a', 'b', 'c', 'd'],
})
spec = create_cross_sectional_specification(
    sample, control_cols=['baseline'], missing_strategy='reject'
)
result = fit_ols(spec)
assert abs(result.model.params['treat'] - 10) < 1e-10
```

Explicit selection leaves `description` out of the model. Use the fitted specification's `control_cols` and `formula` to check what was selected. The optional DoubleML wrapper uses a separate `covariates` argument; its default selection differs, as described in [methods](causal-methods.md#optional-doubleml-partially-linear-model).

Observed categorical, object or string columns are not silently discarded as potential controls. Automatic inference raises an error asking for deliberate encoding or selection. Encode needed categorical controls into numeric columns before using factory pipelines; choose the omitted category and avoid a redundant full set of indicators with an intercept. Alternatively, build an explicit specification with the intended `control_cols`. Some formula estimators support explicitly selected categorical controls, but the C&S, synthetic and machine-learning methods have their own numeric-input requirements. Remove unused descriptive columns from factory inputs or select controls explicitly through the specification API.

Specifications preserve input columns whose names coincide with generated analysis variables. They allocate a distinct internal name and record the generated event-time mapping, so a selected covariate called `event_post0` does not become an event indicator or a plotted treatment-effect coefficient.

## Missing data

Factory defaults are `missing_strategy='drop_rows'` and `max_missing_fraction=0.1`. The factory checks each input column against the original input row count before dropping any incomplete rows. Missing outcomes cannot shrink the denominator used to evaluate missing covariates later in the pipeline. The threshold is per column, so dropping the union of incomplete rows can remove more than that fraction of the sample. Make the choice explicitly when dropping observations changes the target population.

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

The two entry points have different missingness thresholds:

| Entry point | Columns checked | Default threshold | Audit location |
| --- | --- | --- | --- |
| Graph factory | Every supplied input column | 10% missing in each column, using the original row count | Cleaned frame's `attrs['cleaning_metadata']` |
| Direct specification | Outcome, treatment, required design columns and selected controls | 100% of rows missing any required value; an empty remaining sample still fails | `spec.data.attrs['preparation']` |

Both default to `drop_rows`. Set `missing_strategy='reject'` or an explicit threshold for direct specifications as well. Unused descriptive columns can cause a factory to drop observations; select the intended analysis columns before fitting.

Cleaning metadata lives in `cleaned_frame.attrs['cleaning_metadata']` and is included in graph provenance where available. It records before/after row and unit counts, dropped rows, policy settings and transformations. Time transformations include original values and standardized periods.

After fitting a panel graph, retrieve the cleaned frame with `graph.get('panel_cleaned_data').get_result_value()`; for cross-sectional graphs, use `cross_sectional_cleaned_data`. The metadata is a list of cleaning stages. Sum `rows_dropped` across those stages for the total, and inspect each stage's `units_dropped` and `post_validation_passed`. Direct specifications use the separate preparation audit shown in the table.

## Design validation

The factory validates the sample again after cleaning. Panel unit/time keys must be unique, both treatment values must remain, and outcomes must be finite. Each treated unit needs an observed untreated period before adoption; treatment reversals are unsupported. At least one adoption period must retain untreated comparison observations. Unbalanced panels can pass this general check, but individual estimators impose additional requirements. Cleaning failures propagate rather than allowing the original malformed data through silently. Repeated cross-sectional rows are not automatically discarded merely because their observed values coincide.

Dropping missing required keys is possible under `drop_rows`; any required missing values that remain after cleaning are errors. Use `reject` when dropping those observations is unacceptable. A drop that removes all pretreatment observations for a treated unit, or all comparison support, fails revalidation even when it satisfies the missingness threshold.

Synthetic DiD additionally requires a complete unit-by-period grid, donors, pretreatment information and supported simultaneous adoption. Other methods have different requirements; a globally valid DataFrame is not necessarily valid for every estimator. Inspect the chosen route and [method limitations](causal-methods.md).

## Custom cleaning

`AutoCleaner` exposes individual checks and transformation metadata. With the example panel above:

```python
from pyautocausal.data_cleaner_interface.autocleaner import AutoCleaner

cleaner = (
    AutoCleaner(unit_column='id_unit')
    .check_required_columns(['id_unit', 't', 'treat', 'y'])
    .check_for_missing_data(strategy='reject')
    .standardize_time_periods(treatment_column='treat', time_column='t')
    .check_causal_design(unit_column='id_unit', time_column='t')
)
cleaned = cleaner.clean(panel)
assert cleaned['t'].tolist() == [1, 2, 3, 1, 2, 3]
assert cleaned['id_unit'].equals(panel['id_unit'])
assert cleaner.metadata['post_validation_passed']
time_record = next(
    item for item in cleaner.metadata['full_metadata']['transformations']
    if item['operation'] == 'standardize_time_periods'
)
print(time_record['details']['value_mapping'])
```

Manually chosen imputation is separate from the factory's drop/reject policy. Add `check_causal_design` to a custom cleaner to validate its final sample; merely running schema checks does not request this revalidation. A failed post-cleaning design check raises `DataValidationError` with `validation_result`, `cleaning_metadata` and `cleaning_history`. See the [validation module guide](../pyautocausal/data_validation/README.md) for diagnostic checks without cleaning.
