"""Cleaning operations for missing data."""

from typing import Tuple
import pandas as pd
from datetime import datetime

from ..base import CleaningOperation, TransformationRecord
from ..hints import CleaningHint, DropMissingRowsHint, FillMissingWithValueHint


class DropMissingRowsOperation(CleaningOperation):
    """Drop rows with missing values."""
    
    @property
    def name(self) -> str:
        return "drop_missing_rows"
    
    @property
    def priority(self) -> int:
        return 20  # Low priority - do this after other operations
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, DropMissingRowsHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Drop rows with missing values in specified columns."""
        assert isinstance(hint, DropMissingRowsHint)
        initial_rows = len(df)
        
        if hint.target_columns:
            # Drop rows with missing values in specific columns
            df_cleaned = df.dropna(subset=hint.target_columns, how=hint.how)
        else:
            # Drop all rows with any missing values
            df_cleaned = df.dropna()
        
        rows_dropped = initial_rows - len(df_cleaned)
        
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            rows_dropped=rows_dropped,
            details={
                "initial_rows": initial_rows,
                "final_rows": len(df_cleaned),
                "columns_checked": hint.target_columns or "all"
            }
        )
        
        return df_cleaned, record


class FillMissingWithValueOperation(CleaningOperation):
    """Fill missing values with a specified value or strategy."""
    
    @property
    def name(self) -> str:
        return "fill_missing_with_value"
    
    @property
    def priority(self) -> int:
        return 50  # Medium priority
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, FillMissingWithValueHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Fill missing values with specified value or strategy."""
        assert isinstance(hint, FillMissingWithValueHint)
        df_cleaned = df.copy()
        modified_columns = []
        
        fill_value = hint.fill_value
        strategy = hint.strategy
        
        for col in hint.target_columns:
            if col in df_cleaned.columns and df_cleaned[col].isna().any():
                if strategy == "constant":
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                elif strategy == "mode":
                    mode_value = df_cleaned[col].mode()
                    if len(mode_value) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                elif strategy == "forward_fill":
                    df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                elif strategy == "backward_fill":
                    df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
                
                modified_columns.append(col)
        
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            columns_modified=modified_columns,
            details={
                "strategy": strategy,
                "fill_value": fill_value if strategy == "constant" else None,
                "columns_filled": len(modified_columns)
            }
        )
        
        return df_cleaned, record 