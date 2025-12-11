"""Cleaning operations for duplicate data."""

from typing import Tuple
import pandas as pd
from datetime import datetime

from ..base import CleaningOperation, TransformationRecord
from ..hints import CleaningHint, DropDuplicateRowsHint


class DropDuplicateRowsOperation(CleaningOperation):
    """Drop duplicate rows from the DataFrame."""
    
    @property
    def name(self) -> str:
        return "drop_duplicate_rows"
    
    @property
    def priority(self) -> int:
        return 70  # Relatively high priority
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, DropDuplicateRowsHint)
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint) -> Tuple[pd.DataFrame, TransformationRecord]:
        """Drop duplicate rows based on specified columns."""
        assert isinstance(hint, DropDuplicateRowsHint)
        initial_rows = len(df)
        
        # Get parameters
        subset = hint.subset
        keep = hint.keep
        
        # Drop duplicates
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        
        rows_dropped = initial_rows - len(df_cleaned)
        
        record = TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            rows_dropped=rows_dropped,
            details={
                "initial_rows": initial_rows,
                "final_rows": len(df_cleaned),
                "subset": subset or "all columns",
                "keep": keep,
                "duplicates_found": rows_dropped
            }
        )
        
        return df_cleaned, record 