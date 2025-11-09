"""Cleaning operations for time period standardization."""

from typing import Tuple
import pandas as pd
from datetime import datetime

from ..base import CleaningOperation, TransformationRecord
from ..hints import CleaningHint, StandardizeTimePeriodHint


class StandardizeTimePeriodsOperation(CleaningOperation):
    """Standardize time periods using a pre-computed mapping."""
    
    @property
    def name(self) -> str:
        return "standardize_time_periods"
    
    @property
    def priority(self) -> int:
        return 75  # High priority - standardize before most other operations
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, StandardizeTimePeriodHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Apply time period standardization using the pre-computed mapping."""
        assert isinstance(hint, StandardizeTimePeriodHint)
        df_cleaned = df.copy()
        
        if hint.time_column not in df_cleaned.columns:
            raise ValueError(f"Time column '{hint.time_column}' not found in dataframe")
        
        # Apply the mapping directly using original values as keys
        time_column = df_cleaned[hint.time_column]
        
        # Count how many values will be changed by checking which values are in the mapping
        values_in_mapping = time_column.isin(hint.value_mapping.keys())
        values_to_change = values_in_mapping.sum()
        
        if values_to_change == 0:
            raise ValueError(f"No values in time column '{hint.time_column}' match the standardization mapping. "
                           f"Column dtype: {time_column.dtype}, "
                           f"Sample column values: {time_column.unique()[:3].tolist()}, "
                           f"Sample mapping keys: {list(hint.value_mapping.keys())[:3]}")
        
        # Apply the mapping directly and ensure integer result
        mapped_values = time_column.map(hint.value_mapping)
        df_cleaned[hint.time_column] = mapped_values.fillna(df_cleaned[hint.time_column])
        
        # Ensure the result is integer type
        df_cleaned[hint.time_column] = df_cleaned[hint.time_column].astype('int64')
        
        # Create transformation record
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            columns_modified=[hint.time_column],
            details={
                "time_column": hint.time_column,
                "values_standardized": int(values_to_change),
                "total_unique_periods": len(hint.value_mapping),
                "standardized_range": [min(hint.value_mapping.values()), max(hint.value_mapping.values())],
                "pre_treatment_periods": hint.metadata.get("pre_treatment_periods", 0),
                "post_treatment_periods": hint.metadata.get("post_treatment_periods", 0),
                "sample_mapping": dict(list(hint.value_mapping.items())[:5])  # First 5 for logging
            }
        )
        
        return df_cleaned, record