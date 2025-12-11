"""Built-in cleaning operations."""

from typing import List
from ..base import CleaningOperation
from .categorical_operations import ConvertToCategoricalOperation, EncodeMissingAsCategoryOperation
from .missing_data_operations import DropMissingRowsOperation, FillMissingWithValueOperation
from .duplicate_operations import DropDuplicateRowsOperation
from .schema_operations import UpdateColumnTypesOperation
from .time_operations import StandardizeTimePeriodsOperation


def get_all_operations() -> List[CleaningOperation]:
    """Return a list of all available cleaning operations."""
    return [
        UpdateColumnTypesOperation(),
        ConvertToCategoricalOperation(),
        EncodeMissingAsCategoryOperation(),
        DropMissingRowsOperation(),
        FillMissingWithValueOperation(),
        DropDuplicateRowsOperation(),
        StandardizeTimePeriodsOperation()
    ]

__all__ = [
    "get_all_operations",
    "CleaningOperation",
    "UpdateColumnTypesOperation",
    "ConvertToCategoricalOperation",
    "EncodeMissingAsCategoryOperation",
    "DropMissingRowsOperation",
    'FillMissingWithValueOperation',
    'DropDuplicateRowsOperation',
    'StandardizeTimePeriodsOperation',
] 