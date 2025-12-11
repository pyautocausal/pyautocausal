"""Cleaning operations for schema adjustments."""

from typing import Tuple
import pandas as pd
from datetime import datetime

from ..base import CleaningOperation, TransformationRecord
from ..hints import CleaningHint, UpdateColumnTypesHint


class UpdateColumnTypesOperation(CleaningOperation):
    """Update column dtypes based on a type mapping."""
    
    @property
    def name(self) -> str:
        return "update_column_types"
    
    @property
    def priority(self) -> int:
        return 95  # Highest priority to enforce schema
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, UpdateColumnTypesHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Convert specified columns to their target dtypes."""
        assert isinstance(hint, UpdateColumnTypesHint)
        df_cleaned = df.copy()
        modified_columns = []
        
        for col, dtype in hint.type_mapping.items():
            if col not in df_cleaned.columns:
                raise ValueError(f"Error updating dtype for {col}: Column not found in dataframe")
            
            # Only apply if the type is different
            if df_cleaned[col].dtype != dtype:
                try:
                    df_cleaned[col] = df_cleaned[col].astype(dtype)
                    modified_columns.append(col)
                except Exception as e:
                    # Could log a warning here if conversion fails, but for now we raise
                    raise TypeError(f"Could not convert column '{col}' to type {dtype}: {e}")

        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            columns_modified=modified_columns,
            details={
                "columns_converted": len(modified_columns),
                "type_mapping": {k: str(v) for k, v in hint.type_mapping.items()}
            }
        )
        
        return df_cleaned, record 