# Data validation

`DataValidator` runs configured checks on a pandas DataFrame and returns an `AggregatedValidationResult`. It does not clean or mutate the data. Use [AutoCleaner and the factory policies](../../docs/data-requirements.md) when an analysis needs both validation and recorded transformations.

## Validate a DataFrame

Supply check classes or instances. Configuration keys must match the checks' names, such as `required_columns` and `binary_treatment`.

```python
import pandas as pd
from pyautocausal.data_validation import DataValidator, DataValidatorConfig
from pyautocausal.data_validation.checks import (
    RequiredColumnsCheck, RequiredColumnsConfig,
    BinaryTreatmentCheck, BinaryTreatmentConfig,
    CausalDesignCheck, CausalDesignConfig,
)

validator = DataValidator(
    checks=[RequiredColumnsCheck, BinaryTreatmentCheck, CausalDesignCheck],
    config=DataValidatorConfig(check_configs={
        'required_columns': RequiredColumnsConfig(
            required_columns=['treat', 'y']
        ),
        'binary_treatment': BinaryTreatmentConfig(treatment_column='treat'),
        'causal_design': CausalDesignConfig(
            treatment_column='treat', outcome_column='y'
        ),
    }),
)
df = pd.DataFrame({'treat': [0, 0, 1, 1], 'y': [1., 2., 3., 4.]})
result = validator.validate(df)
assert result.passed
print(result.to_string(verbose=True))
```

`BinaryTreatmentCheck` checks allowed values; it does not by itself require both treatment arms. `CausalDesignCheck` additionally requires both arms and finite numeric outcomes. For panel checks, configure `CausalDesignConfig(unit_column='id_unit', time_column='t')` as well. That adds unique unit/time keys, absorbing treatment, observed pretreatment data and comparison-support checks. String unit IDs remain labels. A valid general panel can still fail an estimator's stricter requirements, such as a complete grid for synthetic DiD.

## Inspect failures

By default, validation stops at the first check with ERROR-level issues and raises `DataValidationError`. The exception contains the results collected up to that point. Place required-column checks before checks that use those columns.

```python
from pyautocausal.data_validation.validator_base import DataValidationError

invalid = df.assign(treat=2)
try:
    validator.validate(invalid)
except DataValidationError as error:
    print(error.validation_result.get_failed_checks())

# Diagnostic mode runs all enabled checks without raising for validation issues.
report = validator.validate(invalid, dry_run=True)
assert not report.passed
assert report.summary['total_errors'] > 0
print(report.to_string(verbose=True))
```

`dry_run=True` does not catch arbitrary exceptions inside custom checks; each check must handle invalid input it might receive. An aggregated result exposes `passed`, `individual_results`, `summary`, `get_all_issues()` and `get_failed_checks()`. Each individual result includes its name, issues, metadata and any cleaning hints. Hints describe possible operations; validation alone does not apply them.

The aggregation configuration supports:

| Setting | Behavior |
| --- | --- |
| `aggregation_strategy='all'` | Every executed check must pass; default |
| `aggregation_strategy='any'` | At least one executed check must pass |
| `fail_on_warning=True` | Any WARNING-level issue also makes the aggregate fail |
| `check_configs` | Maps each check name to its configuration |
| `enabled_checks` | Selects names from the supplied checks; `None` runs all supplied enabled checks |

An ERROR still raises in normal mode even with aggregation strategy `any`. A failed result without ERROR-level issues can be returned without an exception; consumers must inspect `passed`. Individual checks also have `enabled` and `severity_on_fail` settings. Changing severity does not necessarily make a check pass: that decision belongs to its implementation. An empty set of checks under `all` passes vacuously, so ensure the intended checks are enabled.

## Use validation in a graph

A validator is callable and can serve as a graph action. Make downstream analysis depend on the validation node so it cannot run first. This example continues from the validator and DataFrame above:

```python
from pyautocausal.orchestration.graph import ExecutableGraph


def summarize(df, validate_data):
    if not validate_data.passed:
        raise ValueError('Resolve validation issues before analysis')
    return float(df['y'].mean())


graph = ExecutableGraph()
graph.create_input_node('df', input_dtype=pd.DataFrame)
graph.create_node('validate_data', action_function=validator, predecessors=['df'])
graph.create_node(
    'analyze', action_function=summarize, predecessors=['df', 'validate_data']
)
graph.fit(df=df)
assert graph.get('analyze').get_result_value() == 2.5
```

ERROR-level failures stop this graph during validation. For a custom reporting branch, wrap `validator.validate(df, dry_run=True)` and route on the returned `passed` flag. A callable validator itself uses normal, raising validation. See the [pipeline guide](../../docs/pipeline-guide.md) for graph branching and argument binding.

## Built-in checks

Checks and their corresponding `...Config` classes are exported from `pyautocausal.data_validation.checks`.

| Purpose | Checks |
| --- | --- |
| Schema and sample size | `NonEmptyDataCheck`, `RequiredColumnsCheck`, `ColumnTypesCheck`, `NoDuplicateColumnsCheck` |
| Duplicate records | `DuplicateRowsCheck` |
| Missingness | `MissingDataCheck`, `CompleteCasesCheck` |
| Treatment and outcome | `BinaryTreatmentCheck`, `TreatmentVariationCheck`, `OutcomeVariableCheck` |
| Panel structure and timing | `PanelStructureCheck`, `TimeColumnCheck`, `TreatmentPersistenceCheck`, `TreatmentTimingPatternsCheck` |
| Analysis sample after cleaning | `CausalDesignCheck` |
| Method-specific data requirements | `CausalMethodRequirementsCheck` |
| Optional transformation hints | `TimePeriodStandardizationCheck`, `InferCategoricalColumnsCheck` |

General checks are not proof of causal identification. In particular, repeated cross-sectional rows can represent distinct observations, and numeric low-cardinality columns can be valid controls. The factories do not automatically delete those rows or convert those controls into categorical variables.

## Create a custom check

Subclass `DataValidationConfig` for settings and `DataValidationCheck` for logic. Implement `name`, `get_default_config` and `validate`. This complete example reports negative values in a cost column without changing the input:

```python
from dataclasses import dataclass
import pandas as pd
from pyautocausal.data_validation import DataValidator
from pyautocausal.data_validation.base import (
    DataValidationCheck, DataValidationConfig, DataValidationResult,
    ValidationIssue,
)


@dataclass
class NonnegativeConfig(DataValidationConfig):
    column: str = 'cost'


class NonnegativeCheck(DataValidationCheck[NonnegativeConfig]):
    @property
    def name(self) -> str:
        return 'nonnegative'

    @classmethod
    def get_default_config(cls) -> NonnegativeConfig:
        return NonnegativeConfig()

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        column = self.config.column
        issues = []
        if column not in df:
            message = f'Missing required column: {column}'
        elif not pd.api.types.is_numeric_dtype(df[column]):
            message = f'{column} must be numeric'
        elif df[column].isna().any() or (df[column] < 0).any():
            message = f'{column} must contain nonnegative, nonmissing values'
        else:
            message = None
        if message:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=message,
                affected_columns=[column],
            ))
        return self._create_result(passed=not issues, issues=issues)


custom = DataValidator(checks=[NonnegativeCheck])
assert custom.validate(pd.DataFrame({'cost': [0., 2.5]})).passed
```

Revalidate the final sample after any transformation affecting units, periods, treatment or outcome. `AutoCleaner.check_causal_design(...)` schedules that final check and attaches the cleaning audit to post-cleaning failures. A standalone validator checks exactly the DataFrame passed to it.
