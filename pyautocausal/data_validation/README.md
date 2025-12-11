# Data Validation Module

The data validation module provides a composable system for validating pandas DataFrames before they are used in causal inference pipelines. This helps catch data quality issues early and ensures that your analysis is based on clean, properly formatted data.

## Key Features

- **Composable Checks**: Build complex validation logic by combining simple, reusable checks
- **Configurable**: Each check can be configured with custom parameters
- **Severity Levels**: Issues can be classified as errors, warnings, or info
- **Integration with PyAutoCausal**: Use validation as nodes in your analysis graphs
- **Extensible**: Easy to add custom validation checks

## Architecture

### Base Classes

- `DataValidationCheck`: Abstract base class for all validation checks
- `DataValidationConfig`: Base configuration class that can be extended
- `DataValidationResult`: Result object containing validation outcome and issues
- `DataValidatorNode`: Aggregates multiple checks into a single validation step

### Built-in Checks

#### Basic Checks
- `NonEmptyDataCheck`: Ensures DataFrame has minimum rows/columns
- `RequiredColumnsCheck`: Verifies required columns are present
- `ColumnTypesCheck`: Validates column data types
- `NoDuplicateColumnsCheck`: Ensures column names are unique

#### Missing Data Checks
- `MissingDataCheck`: Checks for missing values within thresholds
- `CompleteCasesCheck`: Ensures sufficient complete cases

#### Causal Inference Checks
- `BinaryTreatmentCheck`: Validates treatment variable is binary (0/1)
- `TreatmentVariationCheck`: Ensures sufficient treatment/control units
- `PanelStructureCheck`: Validates panel data structure
- `TimeColumnCheck`: Checks time column properties

## Usage Examples

### Basic Usage

```python
from pyautocausal.data_validation import DataValidatorNode, DataValidatorConfig
from pyautocausal.data_validation.checks import RequiredColumnsCheck, BinaryTreatmentCheck

# Create a validator with specific checks
validator = DataValidatorNode(
    checks=[RequiredColumnsCheck, BinaryTreatmentCheck],
    config=DataValidatorConfig(
        check_configs={
            "required_columns": RequiredColumnsConfig(
                required_columns=["treatment", "outcome", "unit_id"]
            ),
            "binary_treatment": BinaryTreatmentConfig(
                treatment_column="treatment"
            )
        }
    )
)

# Validate a DataFrame
result = validator.validate(df)
print(result.to_string(verbose=True))
```

### Integration with PyAutoCausal Graphs

```python
from pyautocausal.orchestration.graph import ExecutableGraph

graph = ExecutableGraph()
graph.configure_runtime(output_path="./outputs")

# Add input node
graph.create_input_node("df", input_dtype=pd.DataFrame)

# Add validation node
graph.create_node(
    "validate_data",
    action_function=validator,
    predecessors=["df"]
)

# Add decision node based on validation
graph.create_decision_node(
    "validation_passed",
    condition=lambda validate_data: validate_data.passed,
    predecessors=["validate_data"]
)

# Continue with analysis if validation passes
graph.create_node(
    "analyze",
    action_function=my_analysis_function,
    predecessors=["df"]
)

# Handle validation failures
graph.create_node(
    "report_errors",
    action_function=create_error_report,
    predecessors=["validate_data"]
)

# Wire the decision
graph.when_true("validation_passed", "analyze")
graph.when_false("validation_passed", "report_errors")
```

## Creating Custom Checks

To create a custom validation check:

1. Extend `DataValidationConfig` for your configuration
2. Extend `DataValidationCheck` for your check logic
3. Implement required methods

```python
from dataclasses import dataclass
from pyautocausal.data_validation.base import (
    DataValidationCheck, 
    DataValidationConfig,
    ValidationIssue,
    ValidationSeverity
)

@dataclass
class MyCustomConfig(DataValidationConfig):
    """Configuration for my custom check."""
    my_parameter: float = 0.5

class MyCustomCheck(DataValidationCheck[MyCustomConfig]):
    """My custom validation check."""
    
    @property
    def name(self) -> str:
        return "my_custom_check"
    
    @classmethod
    def get_default_config(cls) -> MyCustomConfig:
        return MyCustomConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Your validation logic here
        if some_condition:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message="Description of the issue",
                affected_columns=["col1", "col2"],
                details={"additional": "information"}
            ))
        
        passed = len(issues) == 0
        return self._create_result(passed, issues)
```

## Configuration

The `DataValidatorConfig` class controls how validation results are aggregated:

- `fail_on_warning`: Whether to fail if any WARNING-level issues are found (default: False)
- `aggregation_strategy`: How to combine check results
  - `"all"`: All checks must pass (default)
  - `"any"`: At least one check must pass
- `check_configs`: Dictionary of configurations for individual checks
- `enabled_checks`: List of check names to run (if None, run all)

## Best Practices

1. **Start Simple**: Begin with basic checks and add more as needed
2. **Use Appropriate Severity**: 
   - ERROR for critical issues that prevent analysis
   - WARNING for issues that might affect results
   - INFO for suggestions
3. **Configure Thoughtfully**: Set thresholds based on your specific use case
4. **Test Your Validation**: Ensure validation catches the issues you expect
5. **Document Custom Checks**: Clearly explain what each custom check validates 