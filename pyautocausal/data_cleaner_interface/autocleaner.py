import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# Import validation components
from pyautocausal.data_validation.validator_base import DataValidator
from pyautocausal.data_validation.checks.basic_checks import RequiredColumnsCheck, RequiredColumnsConfig, ColumnTypesCheck, ColumnTypesConfig, DuplicateRowsCheck, DuplicateRowsConfig
from pyautocausal.data_validation.checks.missing_data_checks import MissingDataCheck, MissingDataConfig
from pyautocausal.data_validation.checks.categorical_checks import InferCategoricalColumnsCheck, InferCategoricalColumnsConfig
from pyautocausal.data_validation.checks.causal_checks import BinaryTreatmentCheck, BinaryTreatmentConfig, TimePeriodStandardizationCheck, TimePeriodStandardizationConfig

# Import cleaning components
from pyautocausal.data_cleaning.operations.missing_data_operations import DropMissingRowsOperation, FillMissingWithValueOperation
from pyautocausal.data_cleaning.operations.categorical_operations import ConvertToCategoricalOperation, EncodeMissingAsCategoryOperation
from pyautocausal.data_cleaning.operations.duplicate_operations import DropDuplicateRowsOperation
from pyautocausal.data_cleaning.operations.schema_operations import UpdateColumnTypesOperation
from pyautocausal.data_cleaning.operations.time_operations import StandardizeTimePeriodsOperation
from pyautocausal.data_cleaning.planner import DataCleaningPlanner
from pyautocausal.data_cleaning.base import CleaningMetadata

class AutoCleaner:
    """A high-level facade for performing common data validation and cleaning tasks."""
    def __init__(self):
        self._checks = []
        self._operations = []
        self._configs = {}

    def check_required_columns(self, required_columns: List[str], **kwargs):
        """Ensures that the given columns are present in the DataFrame."""
        self._checks.append(RequiredColumnsCheck())
        self._configs["required_columns"] = RequiredColumnsConfig(required_columns=required_columns, **kwargs)
        return self

    def check_column_types(self, expected_types: Dict[str, Any], **kwargs):
        """Checks if columns have the specified data types and adds an operation to fix them."""
        self._checks.append(ColumnTypesCheck())
        self._operations.append(UpdateColumnTypesOperation())
        self._configs["column_types"] = ColumnTypesConfig(expected_types=expected_types, **kwargs)
        return self

    def check_for_missing_data(
        self,
        strategy: str = "drop_rows",
        check_columns: Optional[List[str]] = None,
        fill_value=None,
        **kwargs
    ):
        """
        Adds a check for missing data and a corresponding cleaning operation.

        Args:
            strategy (str): 'drop_rows' or 'fill'.
            check_columns (list, optional): Columns to check for missing data. Defaults to all.
            fill_value: Value to use when strategy is 'fill'.
        """
        self._checks.append(MissingDataCheck())
        self._configs["missing_data"] = MissingDataConfig(check_columns=check_columns, **kwargs)

        if strategy == "drop_rows":
            self._operations.append(DropMissingRowsOperation())
        elif strategy == "fill":
            # Store fill configuration for later use in hint creation
            self._fill_strategy = {"fill_value": fill_value, "check_columns": check_columns}
            self._operations.append(FillMissingWithValueOperation())
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        return self

    def infer_and_convert_categoricals(self, **kwargs):
        """Infers which columns are categorical and adds an operation to convert them."""
        self._checks.append(InferCategoricalColumnsCheck())
        self._operations.append(ConvertToCategoricalOperation())
        self._configs["infer_categorical_columns"] = InferCategoricalColumnsConfig(**kwargs)
        return self

    def check_binary_treatment(self, treatment_column: str, **kwargs):
        """Adds a check to ensure the treatment column is binary."""
        self._checks.append(BinaryTreatmentCheck())
        self._configs["binary_treatment"] = BinaryTreatmentConfig(treatment_column=treatment_column, **kwargs)
        # Note: No default cleaning operation for this. Assumes user handles it or that it's a validation stop.
        return self
    
    def standardize_time_periods(self, treatment_column: str = "treatment", time_column: str = "time", **kwargs):
        """
        Adds a check to standardize time periods relative to first treatment period.
        
        This check:
        1. Finds the minimum time period where treatment==1 occurs
        2. Creates a standardized mapping where that period becomes index 0
        3. Earlier periods get negative indices (-1, -2, etc.)
        4. Later periods get positive indices (1, 2, etc.)
        5. Automatically applies the standardization as a cleaning operation
        
        Args:
            treatment_column (str): Name of the treatment column. Defaults to "treatment".
            time_column (str): Name of the time column. Defaults to "time".
            **kwargs: Additional configuration options.
            
        Returns:
            AutoCleaner: Self for method chaining.
            
        Raises:
            ValidationError: If no treatment data exists (no rows where treatment==1).
            ValidationError: If time column contains mixed data types.
        """
        self._checks.append(TimePeriodStandardizationCheck())
        self._operations.append(StandardizeTimePeriodsOperation())
        self._configs["time_period_standardization"] = TimePeriodStandardizationConfig(
            treatment_column=treatment_column, 
            time_column=time_column, 
            **kwargs
        )
        return self

    def drop_duplicates(self, **kwargs):
        """Adds a check for duplicate rows and an operation to drop them."""
        self._checks.append(DuplicateRowsCheck())
        self._operations.append(DropDuplicateRowsOperation())
        self._configs["duplicate_rows"] = DuplicateRowsConfig(**kwargs)
        return self

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full validation, planning, and cleaning pipeline with automatic metadata logging.

        Args:
            df (pd.DataFrame): The input DataFrame to clean.

        Returns:
            The cleaned DataFrame. Metadata is automatically logged using Python's logging system for framework capture.
            
        Raises:
            DataValidationError: If any validation checks fail with ERROR-level issues. 
                                 The exception contains detailed validation results for debugging.
        """
        logger = logging.getLogger(f"{__name__}.AutoCleaner")
        
        # 1. Validate
        validator = DataValidator(checks=self._checks, config=self._configs)
        validation_result = validator.validate(df)

        # 2. Plan
        planner = DataCleaningPlanner(validation_result, operations=self._operations)
        cleaning_plan = planner.create_plan()

        # 3. Handle fill missing operations that need manual hints
        if hasattr(self, '_fill_strategy'):
            from pyautocausal.data_cleaning.hints import FillMissingWithValueHint
            fill_operation = next((op for op in self._operations if isinstance(op, FillMissingWithValueOperation)), None)
            if fill_operation:
                hint = FillMissingWithValueHint(
                    target_columns=self._fill_strategy.get("check_columns") or list(df.columns),
                    fill_value=self._fill_strategy.get("fill_value"),
                    strategy="constant"
                )
                cleaning_plan.add_operation(fill_operation, hint)

        # Sort operations by priority
        cleaning_plan.sort_operations()

        # 4. Clean
        cleaned_df = cleaning_plan(df.copy())
        metadata = cleaning_plan.get_metadata()
        
        # 5. Log structured metadata for framework capture
        rows_before = len(df)
        rows_after = len(cleaned_df)
        rows_dropped = rows_before - rows_after
        
        # Create summary of operations performed
        operation_names = [op.operation_name for op in metadata.transformations]
        
        # Log the cleaning summary with extra data for framework capture
        logger.info(
            f"Data cleaning completed: {rows_dropped} rows dropped, {len(operation_names)} operations performed",
            extra={
                'cleaning_metadata': {
                    'rows_before': rows_before,
                    'rows_after': rows_after,
                    'rows_dropped': rows_dropped,
                    'operations_performed': operation_names,
                    'total_transformations': len(metadata.transformations),
                    'full_metadata': metadata.to_dict() if hasattr(metadata, 'to_dict') else str(metadata)
                }
            }
        )
        
        # Log individual operations for detailed tracking
        for i, transformation in enumerate(metadata.transformations):
            logger.debug(
                f"Operation {i+1}: {transformation.operation_name}",
                extra={
                    'operation_details': {
                        'operation_name': transformation.operation_name,
                        'order': i + 1,
                        'total_operations': len(metadata.transformations)
                    }
                }
            )

        return cleaned_df 