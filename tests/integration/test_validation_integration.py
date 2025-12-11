import pandas as pd
import pytest
import numpy as np

from pyautocausal.data_cleaning.cleaner import DataCleaner
from pyautocausal.data_cleaning.operations.categorical_operations import ConvertToCategoricalOperation, EncodeMissingAsCategoryOperation
from pyautocausal.data_cleaning.operations.missing_data_operations import DropMissingRowsOperation, FillMissingWithValueOperation
from pyautocausal.data_cleaning.operations.duplicate_operations import DropDuplicateRowsOperation
from pyautocausal.data_cleaning.operations.schema_operations import UpdateColumnTypesOperation
from pyautocausal.data_cleaning.planner import DataCleaningPlanner
from pyautocausal.data_validation.checks.basic_checks import (
    ColumnTypesCheck, RequiredColumnsCheck, RequiredColumnsConfig, ColumnTypesConfig,
    NonEmptyDataCheck, NonEmptyDataConfig
)
from pyautocausal.data_validation.checks.categorical_checks import InferCategoricalColumnsCheck, InferCategoricalColumnsConfig
from pyautocausal.data_validation.checks.missing_data_checks import MissingDataCheck, MissingDataConfig
from pyautocausal.data_validation.checks.causal_checks import BinaryTreatmentCheck, BinaryTreatmentConfig
from pyautocausal.data_validation.validator_base import DataValidator, DataValidatorConfig, AggregatedValidationResult
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.data_cleaning.base import CleaningPlan, CleaningMetadata
from pyautocausal.data_cleaning.hints import DropDuplicateRowsHint, DropMissingRowsHint





def test_validation_to_cleaning_integration(lalonde_data, output_dir):
    """An integration test for the validation -> planning -> cleaning pipeline."""

    # Introduce some issues for the pipeline to fix
    test_data = lalonde_data.copy()
    
    # Create a 'race' column for the test
    test_data['race'] = 'other'
    test_data.loc[test_data['black'] == 1, 'race'] = 'black'
    test_data.loc[test_data['hispan'] == 1, 'race'] = 'hispan'

    # The 'race' column in lalonde is 'black', 'hispan', 'white'
    # The ColumnTypesCheck will fail because it expects a specific dtype, but we can fix this by converting it
    test_data["race"] = test_data["race"].astype("object")
    test_data.loc[0, "re74"] = None  # Introduce missing data
    test_data.loc[1, "re75"] = None
    
    # 1. Setup validation
    validation_checks = [
        RequiredColumnsCheck(config=RequiredColumnsConfig(required_columns=["treat", "age", "educ", "race", "married", "nodegree", "re74", "re75", "re78"])),
        ColumnTypesCheck(config=ColumnTypesConfig(expected_types={"treat": int, "nodegree": int, "married": int})),
        MissingDataCheck(config=MissingDataConfig(check_columns=["re74", "re75"])),
        InferCategoricalColumnsCheck(config=InferCategoricalColumnsConfig(ignore_columns=["treat", "married", "nodegree"]))
    ]
    validator = DataValidator(checks=validation_checks)


    # function to create a cleaning plan based on validation results
    def create_cleaning_plan(validator: AggregatedValidationResult) -> CleaningPlan:
    # Basic cleaning - applies to both cross-sectional and panel data
    # 2. Setup cleaning
        cleaning_operations = [
            UpdateColumnTypesOperation(),
            ConvertToCategoricalOperation(),
            DropMissingRowsOperation(),
        ]

        return DataCleaningPlanner(validator, operations=cleaning_operations).create_plan()


    def execute_cleaning_plan(df: pd.DataFrame, plan: CleaningPlan) -> pd.DataFrame:
        """Execute basic cleaning operations."""
        return plan(df)

    # 3. Create and run graph
    graph = (
        ExecutableGraph()
        .configure_runtime(output_path=output_dir)
        .create_input_node("df", input_dtype=pd.DataFrame)
        .create_node("validator", validator.validate, predecessors=["df"])
        .create_node("plan", create_cleaning_plan, predecessors=["validator"])
        .create_node("cleaner", execute_cleaning_plan, predecessors=["df", "plan"])
    )


    result = graph.fit(df=test_data)
    
    cleaned_df = result.get("cleaner").output.get_only_item()
    plan_metadata = result.get("plan").output.get_only_item().get_metadata()

    # 4. Assertions
    # Check that missing rows were dropped
    assert len(cleaned_df) < len(test_data)
    assert cleaned_df["re74"].isnull().sum() == 0
    assert cleaned_df["re75"].isnull().sum() == 0

    # Check that column types were corrected
    assert isinstance(cleaned_df["race"].dtype, pd.CategoricalDtype)

    # Check metadata
    assert len(plan_metadata.transformations) > 0
    op_names = [op.operation_name for op in plan_metadata.transformations]
    assert "convert_to_categorical" in op_names
    assert "drop_missing_rows" in op_names
    assert plan_metadata.total_rows_dropped > 0
    assert any("race" in t.columns_modified for t in plan_metadata.transformations)


def test_comprehensive_validation_cleaning_direct_dry_run():
    """
    Comprehensive integration test that directly tests validation -> planning -> cleaning 
    pipeline without using a graph. Tests multiple validation scenarios and cleaning operations.
    """
    
    # Create a comprehensive test dataset with various issues
    np.random.seed(42)  # For reproducible results
    n_rows = 1000
    
    # Create base dataset with intentional issues
    test_data = pd.DataFrame({
        # Treatment variable (should be binary 0/1)
        'treatment': np.random.choice([0, 1], n_rows),
        
        # Outcome variable
        'outcome': np.random.normal(10, 5, n_rows),
        
        # Categorical variables that should be converted to category type
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows),
        
        # Numeric variables
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.exponential(50000, n_rows),
        
        # Variables with missing values
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired', None], n_rows, p=[0.6, 0.2, 0.15, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', None], n_rows, p=[0.4, 0.4, 0.15, 0.05]),
        
        # Numeric variable with missing values
        'years_education': np.random.randint(8, 20, n_rows).astype(float),
        
        # ID variable (should remain as-is)
        'id': range(n_rows)
    })
    
    # Introduce specific issues to be fixed by cleaning
    
    # 1. Add missing values to numeric variables
    missing_indices = np.random.choice(n_rows, size=int(0.03 * n_rows), replace=False)
    test_data.loc[missing_indices, 'years_education'] = np.nan
    
    # 2. Add duplicate rows
    duplicates = test_data.sample(n=20, random_state=42)
    test_data = pd.concat([test_data, duplicates], ignore_index=True)
    
    # 3. Add some invalid treatment values that should be caught
    invalid_treatment_indices = np.random.choice(len(test_data), size=5, replace=False)
    test_data.loc[invalid_treatment_indices, 'treatment'] = 2  # Invalid treatment value
    
    # 4. Make sure categorical columns are stored as object dtype (not category)
    test_data['region'] = test_data['region'].astype('object')
    test_data['education_level'] = test_data['education_level'].astype('object')
    test_data['employment_status'] = test_data['employment_status'].astype('object')
    test_data['marital_status'] = test_data['marital_status'].astype('object')
    
    print(f"Original dataset: {test_data.shape}")
    print(f"Missing values: {test_data.isnull().sum().sum()}")
    print(f"Duplicates: {test_data.duplicated().sum()}")
    print(f"Data types:\n{test_data.dtypes}")
    
    # STEP 1: COMPREHENSIVE VALIDATION
    
    # Setup validation checks
    validation_config = DataValidatorConfig(
        fail_on_warning=False,
        aggregation_strategy="all",
        check_configs={
            "required_columns": RequiredColumnsConfig(
                required_columns=["treatment", "outcome", "region", "education_level", "age", "income", "id"]
            ),
            "column_types": ColumnTypesConfig(
                expected_types={
                    "treatment": "int64",
                    "outcome": "float64", 
                    "age": "int64",
                    "income": "float64",
                    "id": "int64"
                },
            ),
            "infer_categorical_columns": InferCategoricalColumnsConfig(
                categorical_threshold=10,
                ignore_columns=['treatment']
            ),
            "missing_data": MissingDataConfig(
                max_missing_fraction=0.1,  # Allow up to 10% missing per column
                check_columns=["treatment", "outcome", "years_education", "employment_status", "marital_status"]
            ),
            "binary_treatment": BinaryTreatmentConfig(
                treatment_column="treatment",
                valid_values={0, 1},
                allow_missing=False
            ),
            "non_empty_data": NonEmptyDataConfig(
                min_rows=100,
                min_columns=5
            )
        }
    )
    
    validation_checks = [
        RequiredColumnsCheck(),
        ColumnTypesCheck(),
        MissingDataCheck(),
        BinaryTreatmentCheck(),
        NonEmptyDataCheck(),
        InferCategoricalColumnsCheck()
    ]
    
    validator = DataValidator(checks=validation_checks, config=validation_config)
    
    # Run validation
    print("\n" + "="*50)
    print("RUNNING VALIDATION")
    print("="*50)
    
    validation_result = validator.validate(test_data, dry_run=True)
    
    print(f"Validation passed: {validation_result.passed}")
    print(f"Total issues found: {len(validation_result.get_all_issues())}")
    print(f"Failed checks: {validation_result.get_failed_checks()}")
    
    # Print validation summary
    print("\nValidation Summary:")
    print(validation_result.to_string(verbose=True))
    
    # STEP 2: CLEANING PLAN CREATION
    
    # Setup all available cleaning operations
    cleaning_operations = [
        UpdateColumnTypesOperation(),
        ConvertToCategoricalOperation(),
        EncodeMissingAsCategoryOperation(),
        DropMissingRowsOperation(),
        FillMissingWithValueOperation(),
        DropDuplicateRowsOperation()
    ]
    
    # Create cleaning plan from validation results
    print("\n" + "="*50)
    print("CREATING CLEANING PLAN")
    print("="*50)
    
    planner = DataCleaningPlanner(validation_result, operations=cleaning_operations)
    cleaning_plan = planner.create_plan()
    
    # Since validation checks generate all necessary hints now, 
    # we can remove the manual hint creation.
    
    # Add hint to drop duplicates
    cleaning_plan.add_operation(
        DropDuplicateRowsOperation(), 
        DropDuplicateRowsHint(subset=None, keep="first")
    )
    
    # Add hint to handle missing data in specific columns
    cleaning_plan.add_operation(
        DropMissingRowsOperation(),
        DropMissingRowsHint(target_columns=["treatment", "outcome"], how="any")
    )
    
    # Sort operations by priority after adding manual hints
    cleaning_plan.sort_operations()
    
    # Additionally, we need to handle invalid treatment values manually
    # since there's no built-in cleaning operation for this yet
    print("\nHandling invalid treatment values manually...")
    before_invalid_fix = len(test_data)
    test_data = test_data[test_data['treatment'].isin([0, 1])]
    after_invalid_fix = len(test_data)
    print(f"Removed {before_invalid_fix - after_invalid_fix} rows with invalid treatment values")
    
    print("Cleaning Plan:")
    print(cleaning_plan.describe())
    
    # STEP 3: EXECUTE CLEANING
    
    print("\n" + "="*50)
    print("EXECUTING CLEANING")
    print("="*50)
    
    # Execute the cleaning plan
    cleaned_data = cleaning_plan(test_data)
    cleaning_metadata = cleaning_plan.get_metadata()
    
    print(f"Cleaned dataset: {cleaned_data.shape}")
    print(f"Operations executed: {cleaning_metadata.total_operations}")
    print(f"Rows dropped: {cleaning_metadata.total_rows_dropped}")
    print(f"Columns modified: {len(set().union(*[t.columns_modified for t in cleaning_metadata.transformations]))}")
    
    # Print cleaning metadata
    print("\nCleaning Transformations:")
    for i, transform in enumerate(cleaning_metadata.transformations, 1):
        print(f"{i}. {transform.operation_name}")
        if transform.columns_modified:
            print(f"   Modified columns: {transform.columns_modified}")
        if transform.rows_dropped > 0:
            print(f"   Rows dropped: {transform.rows_dropped}")
        print(f"   Details: {transform.details}")
    
    # STEP 4: VALIDATE CLEANING RESULTS
    
    print("\n" + "="*50)
    print("VALIDATING CLEANING RESULTS")
    print("="*50)
    
    # Re-run validation on cleaned data to ensure issues were fixed
    final_validation_result = validator.validate(cleaned_data)
    
    print(f"Final validation passed: {final_validation_result.passed}")
    print(f"Remaining issues: {len(final_validation_result.get_all_issues())}")
    
    # STEP 5: COMPREHENSIVE ASSERTIONS
    
    # Dataset size assertions
    assert len(cleaned_data) <= len(test_data), "Cleaned dataset should not be larger than original"
    assert len(cleaned_data) > 0, "Cleaned dataset should not be empty"
    
    # Missing data assertions
    assert cleaned_data['treatment'].isnull().sum() == 0, "Treatment column should have no missing values"
    assert cleaned_data['outcome'].isnull().sum() == 0, "Outcome column should have no missing values"
    
    # Treatment validity assertions
    treatment_values = set(cleaned_data['treatment'].unique())
    assert treatment_values.issubset({0, 1}), f"Treatment should only contain 0 and 1, found: {treatment_values}"
    
    # Categorical conversion assertions
    assert isinstance(cleaned_data['region'].dtype, pd.CategoricalDtype), "Region should be categorical"
    assert isinstance(cleaned_data['education_level'].dtype, pd.CategoricalDtype), "Education level should be categorical"
    assert isinstance(cleaned_data['employment_status'].dtype, pd.CategoricalDtype), "Employment status should be categorical"
    assert isinstance(cleaned_data['marital_status'].dtype, pd.CategoricalDtype), "Marital status should be categorical"
    
    # Duplicate removal assertions
    assert cleaned_data.duplicated().sum() == 0, "Should have no duplicate rows"
    
    # Required columns assertions
    required_columns = ['treatment', 'outcome', 'region', 'education_level', 'age', 'income', 'id']
    missing_columns = set(required_columns) - set(cleaned_data.columns)
    assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"
    
    # Cleaning metadata assertions
    assert cleaning_metadata.total_operations > 0, "Should have executed at least one cleaning operation"
    assert cleaning_metadata.start_shape == test_data.shape, "Should record original shape"
    assert cleaning_metadata.end_shape == cleaned_data.shape, "Should record final shape"
    assert cleaning_metadata.duration_seconds is not None, "Should record execution time"
    
    # Check that specific operations were applied
    operation_names = [t.operation_name for t in cleaning_metadata.transformations]
    assert "convert_to_categorical" in operation_names, "Should have inferred categorical columns"
    assert "drop_duplicate_rows" in operation_names, "Should have dropped duplicate rows"
    
    # Data quality improvements
    original_missing_pct = (test_data.isnull().sum().sum() / test_data.size) * 100
    cleaned_missing_pct = (cleaned_data.isnull().sum().sum() / cleaned_data.size) * 100
    print(f"\nData quality improvements:")
    print(f"Missing data: {original_missing_pct:.2f}% -> {cleaned_missing_pct:.2f}%")
    print(f"Duplicate rows: {test_data.duplicated().sum()} -> {cleaned_data.duplicated().sum()}")
    print(f"Invalid treatment values: {(test_data['treatment'] == 2).sum()} -> {(cleaned_data['treatment'] == 2).sum()}")
    
    # Final validation should have fewer issues
    original_issues = len(validation_result.get_all_issues())
    final_issues = len(final_validation_result.get_all_issues())
    assert final_issues < original_issues, f"Cleaning should reduce issues: {original_issues} -> {final_issues}"
    
    print("\n" + "="*50)
    print("INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("="*50)
    print(f"✓ Validated {len(validation_checks)} types of data quality issues")
    print(f"✓ Applied {cleaning_metadata.total_operations} cleaning operations")
    print(f"✓ Reduced data issues from {original_issues} to {final_issues}")
    print(f"✓ Cleaned dataset shape: {cleaned_data.shape}")
    print(f"✓ All assertions passed")

