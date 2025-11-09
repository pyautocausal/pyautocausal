"""Type-safe cleaning hint classes.

This module defines specific hint classes for each type of cleaning operation,
ensuring type safety and explicit contracts between validation and cleaning.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class CleaningHint(ABC):
    """Abstract base class for all cleaning hints.
    
    Each concrete hint class represents a specific type of cleaning operation
    and carries the exact data needed for that operation.
    """
    @property
    def priority(self) -> int:
        """Priority for ordering operations (higher = executed first)."""
        raise NotImplementedError("Subclasses must define priority")


@dataclass
class UpdateColumnTypesHint(CleaningHint):
    """Hint to convert column dtypes for specified columns."""
    type_mapping: Dict[str, Any]

    @property
    def priority(self) -> int:
        return 95  # Highest priority to enforce schema


@dataclass
class InferCategoricalHint(CleaningHint):
    """Hint to convert columns to categorical dtype based on inference."""
    target_columns: List[str]
    threshold: int
    unique_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return 80  # Lower priority than explicit conversion


@dataclass
class EncodeMissingAsCategoryHint(CleaningHint):
    """Hint to encode missing values as a category in categorical columns."""
    target_columns: List[str]
    missing_category: str = "MISSING"
    
    @property
    def priority(self) -> int:
        return 85  # High priority - do this before dropping missing values


@dataclass
class DropMissingRowsHint(CleaningHint):
    """Hint to drop rows with missing values."""
    target_columns: Optional[List[str]] = None  # None means check all columns
    how: str = "any"  # 'any' or 'all'
    
    @property
    def priority(self) -> int:
        return 20  # Low priority - do this after other operations


@dataclass
class FillMissingWithValueHint(CleaningHint):
    """Hint to fill missing values with a specified value or strategy."""
    target_columns: List[str]
    strategy: str = "constant"  # 'constant', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
    fill_value: Any = 0  # Used when strategy='constant'
    
    @property
    def priority(self) -> int:
        return 50  # Medium priority


@dataclass
class DropDuplicateRowsHint(CleaningHint):
    """Hint to drop duplicate rows."""
    subset: Optional[List[str]] = None  # None means consider all columns
    keep: str = "first"  # 'first', 'last', or False
    
    @property
    def priority(self) -> int:
        return 70  # Relatively high priority


@dataclass
class StandardizeTimePeriodHint(CleaningHint):
    """Hint to standardize time periods relative to first treatment period."""
    time_column: str
    value_mapping: Dict[str, int]  # original_value -> standardized_index (str keys for JSON serialization)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def priority(self) -> int:
        return 75  # High priority - standardize before most other operations


# For backward compatibility during migration
def create_legacy_hint(operation_type: str, **kwargs) -> Optional[CleaningHint]:
    """Create a hint from legacy string-based operation type.
    
    This function helps during migration from string-based hints to type-safe hints.
    """
    mapping = {
        "update_column_types": UpdateColumnTypesHint,
        "encode_missing_as_category": EncodeMissingAsCategoryHint,
        "drop_missing_rows": DropMissingRowsHint,
        "fill_missing_with_value": FillMissingWithValueHint,
        "drop_duplicate_rows": DropDuplicateRowsHint,
        "standardize_time_periods": StandardizeTimePeriodHint,
    }
    
    hint_class = mapping.get(operation_type)
    if hint_class is None:
        return None
    
    # Extract relevant kwargs for each hint type
    if operation_type == "update_column_types":
        return hint_class(
            type_mapping=kwargs.get("type_mapping", {})
        )
    elif operation_type == "encode_missing_as_category":
        return hint_class(
            target_columns=kwargs.get("target_columns", []),
            missing_category=kwargs.get("missing_category", "MISSING")
        )
    elif operation_type == "drop_missing_rows":
        return hint_class(
            target_columns=kwargs.get("target_columns"),
            how=kwargs.get("how", "any")
        )
    elif operation_type == "fill_missing_with_value":
        return hint_class(
            target_columns=kwargs.get("target_columns", []),
            strategy=kwargs.get("strategy", "constant"),
            fill_value=kwargs.get("fill_value", 0)
        )
    elif operation_type == "drop_duplicate_rows":
        return hint_class(
            subset=kwargs.get("subset"),
            keep=kwargs.get("keep", "first")
        )
    elif operation_type == "standardize_time_periods":
        return hint_class(
            time_column=kwargs.get("time_column", "time"),
            value_mapping=kwargs.get("value_mapping", {}),
            treatment_start_period=kwargs.get("treatment_start_period", ""),
            metadata=kwargs.get("metadata", {})
        )
    
    return None 