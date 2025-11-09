"""Missing data validation checks for pandas DataFrames."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)
from pyautocausal.data_cleaning.hints import DropMissingRowsHint


@dataclass
class MissingDataConfig(DataValidationConfig):
    """Configuration for MissingDataCheck."""
    max_missing_fraction: float = 0.1  # Maximum fraction of missing data allowed per column
    check_columns: Optional[List[str]] = None  # If None, check all columns
    ignore_columns: Optional[List[str]] = None  # Columns to ignore


class MissingDataCheck(DataValidationCheck[MissingDataConfig]):
    """Check for missing data in the DataFrame."""
    
    @property
    def name(self) -> str:
        return "missing_data"
    
    @classmethod
    def get_default_config(cls) -> MissingDataConfig:
        return MissingDataConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        missing_stats = {}
        
        # Determine which columns to check
        # validate that check_columns and ignore_columns are valid
        if self.config.check_columns is not None and not all(col in df.columns for col in self.config.check_columns):
            raise ValueError("check_columns contains columns not present in the DataFrame")
        if self.config.ignore_columns is not None and not all(col in df.columns for col in self.config.ignore_columns):
            raise ValueError("ignore_columns contains columns not present in the DataFrame")
        if self.config.check_columns is not None and self.config.ignore_columns is not None:
            raise ValueError("Cannot specify both check_columns and ignore_columns")

        columns_to_check = list(df.columns)
        if self.config.check_columns is not None:
            columns_to_check = [col for col in self.config.check_columns if col in df.columns]
        if self.config.ignore_columns is not None:
            columns_to_check = [col for col in columns_to_check if col not in self.config.ignore_columns]
        
        # Check each column for missing data
        for col in columns_to_check:
            missing_count = df[col].isna().sum()
            missing_fraction = missing_count / len(df)
            missing_stats[col] = {
                "missing_count": int(missing_count),
                "missing_fraction": float(missing_fraction)
            }
            
            cleaning_hints = []
            if missing_fraction > 0:
                if missing_fraction > self.config.max_missing_fraction:
                    issues.append(ValidationIssue(
                        severity=self.config.severity_on_fail,
                        message=f"Column '{col}' has {missing_fraction:.1%} missing values, exceeding threshold of {self.config.max_missing_fraction:.1%}",
                        affected_columns=[col],
                        details={
                            "missing_count": int(missing_count),
                            "missing_fraction": float(missing_fraction),
                            "threshold": self.config.max_missing_fraction
                        }
                    ))
                else:
                    issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Column '{col}' has {missing_count} missing values ({missing_fraction:.1%})",
                    affected_columns=[col],
                    details={
                        "missing_count": int(missing_count),
                        "missing_fraction": float(missing_fraction)
                    }
                ))
        cleaning_hints.append(
            DropMissingRowsHint(
                target_columns=list(missing_stats.keys()),
                how="any"
            )
        )
                
        # Determine if the check passed
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "missing_stats": missing_stats,
            "total_missing": sum(stats["missing_count"] for stats in missing_stats.values()),
            "columns_checked": len(columns_to_check)
        }
        
        return self._create_result(passed, issues, metadata, cleaning_hints=cleaning_hints)


@dataclass
class CompleteCasesConfig(DataValidationConfig):
    """Configuration for CompleteCasesCheck."""
    min_complete_fraction: float = 0.8  # Minimum fraction of complete cases required
    check_columns: Optional[List[str]] = None  # If None, check all columns


class CompleteCasesCheck(DataValidationCheck[CompleteCasesConfig]):
    """Check that there are sufficient complete cases (rows without any missing values)."""
    
    @property
    def name(self) -> str:
        return "complete_cases"
    
    @classmethod
    def get_default_config(cls) -> CompleteCasesConfig:
        return CompleteCasesConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Determine which columns to check
        columns_to_check = list(df.columns)
        if self.config.check_columns is not None:
            columns_to_check = [col for col in self.config.check_columns if col in df.columns]
        
        # Count complete cases
        df_subset = df[columns_to_check]
        complete_cases = (~df_subset.isna().any(axis=1)).sum()
        complete_fraction = complete_cases / len(df)
        
        if complete_fraction < self.config.min_complete_fraction:
            # Find rows with missing data for details
            incomplete_rows = df_subset.isna().any(axis=1)
            sample_incomplete = incomplete_rows[incomplete_rows].index[:5].tolist()
            
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Only {complete_fraction:.1%} of rows are complete, below threshold of {self.config.min_complete_fraction:.1%}",
                affected_rows=sample_incomplete,
                details={
                    "complete_cases": int(complete_cases),
                    "total_cases": len(df),
                    "complete_fraction": float(complete_fraction),
                    "threshold": self.config.min_complete_fraction,
                    "sample_incomplete_rows": sample_incomplete
                }
            ))
        
        passed = len(issues) == 0
        metadata = {
            "complete_cases": int(complete_cases),
            "incomplete_cases": len(df) - int(complete_cases),
            "complete_fraction": float(complete_fraction),
            "columns_checked": columns_to_check
        }
        
        return self._create_result(passed, issues, metadata) 