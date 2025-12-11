"""Cleaning operations for categorical data."""

from typing import Tuple
import pandas as pd
from datetime import datetime

from ..base import CleaningOperation, TransformationRecord
from ..hints import CleaningHint, EncodeMissingAsCategoryHint, InferCategoricalHint


class ConvertToCategoricalOperation(CleaningOperation):
    """Convert columns to pandas categorical dtype."""
    
    @property
    def name(self) -> str:
        return "convert_to_categorical"
    
    @property
    def priority(self) -> int:
        return 80  # Runs after schema enforcement
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, InferCategoricalHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Convert specified columns to categorical dtype based on inference."""
        assert self.can_apply(hint)
        df_cleaned = df.copy()
        modified_columns = []
        
        for col in hint.target_columns:
            if col not in df_cleaned.columns:
                raise ValueError(f"Error converting to categorical {col}: Column not found in dataframe")
            elif not isinstance(df_cleaned[col].dtype, pd.CategoricalDtype):
                df_cleaned[col] = df_cleaned[col].astype('category')
                modified_columns.append(col)
        
        details = { "columns_converted": len(modified_columns) }
        if isinstance(hint, InferCategoricalHint):
            details["threshold"] = hint.threshold
            details["unique_counts"] = hint.unique_counts
            
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            columns_modified=modified_columns,
            details=details
        )
        
        return df_cleaned, record


class EncodeMissingAsCategoryOperation(CleaningOperation):
    """Encode missing values in categorical columns as a separate category."""
    
    @property
    def name(self) -> str:
        return "encode_missing_as_category"
    
    @property
    def priority(self) -> int:
        return 85  # High priority - do this before dropping missing values
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, EncodeMissingAsCategoryHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Encode missing values as a category in categorical columns."""
        assert isinstance(hint, EncodeMissingAsCategoryHint)
        df_cleaned = df.copy()
        modified_columns = []
        missing_category = hint.missing_category
        
        for col in hint.target_columns:
            if col in df_cleaned.columns:
                if isinstance(df_cleaned[col].dtype, pd.CategoricalDtype):
                    # Add missing category if not present
                    if missing_category not in df_cleaned[col].cat.categories:
                        df_cleaned[col] = df_cleaned[col].cat.add_categories([missing_category])
                    # Fill NaN with missing category
                    df_cleaned[col] = df_cleaned[col].fillna(missing_category)
                    modified_columns.append(col)
                else:
                    # Convert to categorical first if needed
                    df_cleaned[col] = df_cleaned[col].fillna(missing_category).astype('category')
                    modified_columns.append(col)
        
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            columns_modified=modified_columns,
            details={
                "missing_category": missing_category,
                "columns_processed": len(modified_columns)
            }
        )
        
        return df_cleaned, record 