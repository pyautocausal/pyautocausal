"""Basic data validation checks for pandas DataFrames."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable
import pandas as pd
import numpy as np
import datetime

from ..base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)
from pyautocausal.data_cleaning.hints import UpdateColumnTypesHint


@dataclass
class NonEmptyDataConfig(DataValidationConfig):
    """Configuration for NonEmptyDataCheck."""
    min_rows: int = 1
    min_columns: int = 1


class NonEmptyDataCheck(DataValidationCheck[NonEmptyDataConfig]):
    """Check that the DataFrame is not empty."""
    
    @property
    def name(self) -> str:
        return "non_empty_data"
    
    @classmethod
    def get_default_config(cls) -> NonEmptyDataConfig:
        return NonEmptyDataConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check rows
        if len(df) < self.config.min_rows:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"DataFrame has {len(df)} rows, but minimum required is {self.config.min_rows}",
                details={"actual_rows": len(df), "required_rows": self.config.min_rows}
            ))
        
        # Check columns
        if len(df.columns) < self.config.min_columns:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"DataFrame has {len(df.columns)} columns, but minimum required is {self.config.min_columns}",
                details={"actual_columns": len(df.columns), "required_columns": self.config.min_columns}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "n_rows": len(df),
            "n_columns": len(df.columns)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class RequiredColumnsConfig(DataValidationConfig):
    """Configuration for RequiredColumnsCheck."""
    required_columns: List[str] = None  # Will be set in __post_init__
    case_sensitive: bool = True
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = []


class RequiredColumnsCheck(DataValidationCheck[RequiredColumnsConfig]):
    """Check that required columns are present in the DataFrame."""
    
    @property
    def name(self) -> str:
        return "required_columns"
    
    @classmethod
    def get_default_config(cls) -> RequiredColumnsConfig:
        return RequiredColumnsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        if not self.config.required_columns:
            # No required columns specified, pass by default
            return self._create_result(True, [], {"message": "No required columns specified"})
        
        # Get DataFrame columns
        df_columns = set(df.columns)
        if not self.config.case_sensitive:
            df_columns = {col.lower() for col in df_columns}
        
        # Check each required column
        missing_columns = []
        for required_col in self.config.required_columns:
            check_col = required_col if self.config.case_sensitive else required_col.lower()
            if check_col not in df_columns:
                missing_columns.append(required_col)
        
        if missing_columns:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Missing required columns: {', '.join(missing_columns)}",
                details={"missing_columns": missing_columns}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "required_columns": self.config.required_columns,
            "missing_columns": missing_columns,
            "found_columns": list(df.columns)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class ColumnTypesConfig(DataValidationConfig):
    """Configuration for ColumnTypesCheck."""
    expected_types: Dict[str, type] = None  # Column name -> expected type
    type_checkers: Optional[Dict[type, Callable[[pd.Series], bool]]] = None  # supplementary map of type to checker
    
    def __post_init__(self):
        if self.expected_types is None:
            self.expected_types = {}
        if self.type_checkers is None:
            self.type_checkers = {}


class ColumnTypesCheck(DataValidationCheck[ColumnTypesConfig]):
    """Check that columns have expected data types."""
    
    @property
    def name(self) -> str:
        return "column_types"
    
    @classmethod
    def get_default_config(cls) -> ColumnTypesConfig:
        return ColumnTypesConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        actual_types = {}
        cleaning_hints = []
        columns_to_convert = {}

        from pandas.api import types as ptypes
        default_type_checkers: Dict[type, Callable[[pd.Series], bool]] = {
            str: ptypes.is_string_dtype,
            "object": ptypes.is_object_dtype,
            int: ptypes.is_integer_dtype,
            float: ptypes.is_float_dtype,
            bool: ptypes.is_bool_dtype,
            datetime.datetime: ptypes.is_datetime64_any_dtype,
            pd.CategoricalDtype: ptypes.is_categorical_dtype,
        }
        type_checkers = {**default_type_checkers, **self.config.type_checkers}
        
        # Check specified column types
        for col, expected_type in self.config.expected_types.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' not found for type checking",
                    affected_columns=[col]
                ))
                continue
            
            actual_dtype = df[col].dtype
            actual_types[col] = str(actual_dtype)
            
            checker = type_checkers.get(expected_type)
            if checker is None:
                # Fallback to simple type checking if no checker is registered
                if not issubclass(actual_dtype.type, np.dtype(expected_type).type):
                    type_matches = False
                else:
                    type_matches = True
            else:
                type_matches = checker(df[col])
 
            if not type_matches:
                # If types don't match, try a safe conversion
                try:
                    df[col].astype(expected_type)
                    # If conversion is possible, create an INFO issue and a hint
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{col}' has type {actual_dtype} but is convertible to {expected_type}",
                        affected_columns=[col],
                        details={"actual_type": str(actual_dtype), "expected_type": str(expected_type)}
                    ))
                    columns_to_convert[col] = expected_type

                except (ValueError, TypeError, pd.errors.IntCastingNaNError):
                    # If conversion fails, it's a more severe issue
                    issues.append(ValidationIssue(
                        severity=self.config.severity_on_fail,
                        message=f"Column '{col}' has type {actual_dtype}, expected {expected_type}, and cannot be converted.",
                        affected_columns=[col],
                        details={"actual_type": str(actual_dtype), "expected_type": str(expected_type)}
                    ))
        
        if columns_to_convert:
            cleaning_hints.append(UpdateColumnTypesHint(type_mapping=columns_to_convert))

        passed = not any(issue.severity >= self.config.severity_on_fail for issue in issues)
        metadata = {
            "actual_types": actual_types,
        }
        
        return self._create_result(passed, issues, metadata, cleaning_hints)


@dataclass
class NoDuplicateColumnsConfig(DataValidationConfig):
    """Configuration for NoDuplicateColumnsCheck."""
    case_sensitive: bool = True


class NoDuplicateColumnsCheck(DataValidationCheck[NoDuplicateColumnsConfig]):
    """Check that there are no duplicate column names."""
    
    @property
    def name(self) -> str:
        return "no_duplicate_columns"
    
    @classmethod
    def get_default_config(cls) -> NoDuplicateColumnsConfig:
        return NoDuplicateColumnsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check for duplicate column names
        columns = df.columns.tolist()
        if not self.config.case_sensitive:
            columns = [col.lower() for col in columns]
        
        seen = set()
        duplicates = set()
        for col in columns:
            if col in seen:
                duplicates.add(col)
            seen.add(col)
        
        if duplicates:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Duplicate column names found: {', '.join(duplicates)}",
                details={"duplicate_columns": list(duplicates)}
            ))
        
        passed = len(issues) == 0
        metadata = {
            "n_columns": len(df.columns),
            "n_unique_columns": len(set(columns)),
            "duplicates": list(duplicates)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class DuplicateRowsConfig(DataValidationConfig):
    """Configuration for DuplicateRowsCheck."""
    subset: Optional[List[str]] = None  # Columns to check for duplicates (None = all columns)
    keep: str = "first"  # Which duplicate to keep ("first", "last", False)
    max_duplicate_fraction: float = 0.05  # Maximum allowed fraction of duplicate rows


class DuplicateRowsCheck(DataValidationCheck[DuplicateRowsConfig]):
    """Check for duplicate rows in the DataFrame."""
    
    @property
    def name(self) -> str:
        return "duplicate_rows"
    
    @classmethod
    def get_default_config(cls) -> DuplicateRowsConfig:
        return DuplicateRowsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        from pyautocausal.data_cleaning.hints import DropDuplicateRowsHint
        
        issues = []
        cleaning_hints = []
        
        # Check for duplicates
        if self.config.subset is not None:
            # Validate that subset columns exist
            missing_cols = set(self.config.subset) - set(df.columns)
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate check subset columns not found: {', '.join(missing_cols)}",
                    details={"missing_columns": list(missing_cols)}
                ))
                return self._create_result(False, issues)
            
            duplicates = df.duplicated(subset=self.config.subset)
        else:
            duplicates = df.duplicated()
        
        duplicate_count = duplicates.sum()
        duplicate_fraction = duplicate_count / len(df) if len(df) > 0 else 0
        
        # Create metadata
        metadata = {
            "total_rows": len(df),
            "duplicate_rows": int(duplicate_count),
            "duplicate_fraction": float(duplicate_fraction),
            "subset_columns": self.config.subset or "all",
            "keep_strategy": self.config.keep
        }
        
        # Check if duplicates exceed threshold
        if duplicate_count > 0:
            if duplicate_fraction > self.config.max_duplicate_fraction:
                issues.append(ValidationIssue(
                    severity=self.config.severity_on_fail,
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_fraction:.1%}), exceeding threshold of {self.config.max_duplicate_fraction:.1%}",
                    details={
                        "duplicate_count": int(duplicate_count),
                        "duplicate_fraction": float(duplicate_fraction),
                        "threshold": self.config.max_duplicate_fraction,
                        "sample_duplicate_indices": duplicates[duplicates].index[:5].tolist()
                    }
                ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_fraction:.1%})",
                    details={
                        "duplicate_count": int(duplicate_count),
                        "duplicate_fraction": float(duplicate_fraction)
                    }
                ))
            
            # Always generate cleaning hint when duplicates are found
            cleaning_hints.append(
                DropDuplicateRowsHint(
                    subset=self.config.subset,
                    keep=self.config.keep
                )
            )
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        return self._create_result(passed, issues, metadata, cleaning_hints)