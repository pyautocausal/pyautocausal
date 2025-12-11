"""Data cleaning module for PyAutoCausal.

This module provides automated data cleaning based on validation results.
"""

from .base import (
    CleaningOperation,
    CleaningPlan,
    CleaningMetadata,
    TransformationRecord
)
from .planner import DataCleaningPlanner
from .cleaner import DataCleaner
from .hints import (
    CleaningHint,
    UpdateColumnTypesHint,
    InferCategoricalHint,
    EncodeMissingAsCategoryHint,
    DropMissingRowsHint,
    FillMissingWithValueHint,
    DropDuplicateRowsHint,
)
from .operations import *

__all__ = [
    # Base classes
    'CleaningOperation',
    'CleaningPlan',
    'CleaningMetadata',
    'TransformationRecord',
    # Main components
    'DataCleaningPlanner',
    'DataCleaner',
    # Hint classes
    'CleaningHint',
    'UpdateColumnTypesHint',
    'InferCategoricalHint',
    'EncodeMissingAsCategoryHint',
    'DropMissingRowsHint',
    'FillMissingWithValueHint',
    'DropDuplicateRowsHint',
] 