"""Categorical data validation checks for pandas DataFrames."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from ..base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)
from pyautocausal.data_cleaning.hints import InferCategoricalHint


@dataclass
class InferCategoricalColumnsConfig(DataValidationConfig):
    """Configuration for InferCategoricalColumnsCheck."""
    categorical_threshold: int = 10  # Max unique values to consider as categorical
    ignore_columns: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.categorical_threshold, int):
            raise TypeError(f"categorical_threshold must be an integer, got {type(self.categorical_threshold).__name__}")
        
        if self.categorical_threshold < 0:
            raise ValueError(f"categorical_threshold must be non-negative, got {self.categorical_threshold}")


class InferCategoricalColumnsCheck(DataValidationCheck[InferCategoricalColumnsConfig]):
    """Check for columns that could be inferred as categorical."""

    @property
    def name(self) -> str:
        return "infer_categorical_columns"

    @classmethod
    def get_default_config(cls) -> InferCategoricalColumnsConfig:
        return InferCategoricalColumnsConfig()

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        inferred_categorical = []
        
        columns_to_check = df.columns
        if self.config.ignore_columns:
            columns_to_check = [col for col in columns_to_check if col not in self.config.ignore_columns]

        for col in columns_to_check:
            # We only want to infer for object or numeric types that aren't already categorical
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                continue

            if df[col].nunique() <= self.config.categorical_threshold:
                inferred_categorical.append(col)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Column '{col}' appears to be categorical ({df[col].nunique()} unique values)",
                    affected_columns=[col],
                    details={"unique_values": df[col].nunique()}
                ))

        passed = True  # This check only produces INFO issues
        metadata = {
            "inferred_categorical_columns": inferred_categorical,
            "categorical_threshold": self.config.categorical_threshold
        }

        cleaning_hints = []
        if inferred_categorical:
            cleaning_hints.append(InferCategoricalHint(
                target_columns=inferred_categorical,
                threshold=self.config.categorical_threshold,
                unique_counts={col: df[col].nunique() for col in inferred_categorical}
            ))

        return self._create_result(passed, issues, metadata, cleaning_hints)
