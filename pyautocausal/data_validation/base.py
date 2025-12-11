"""Base classes and interfaces for data validation.

This module defines the core abstractions for the data validation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import pandas as pd


class ValidationSeverity(IntEnum):
    """Severity levels for validation issues."""
    INFO = 1
    WARNING = 2
    ERROR = 3


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the data."""
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None


# CleaningHint has been moved to data_cleaning.hints for type safety
# Import it from there when needed


@dataclass
class DataValidationResult:
    """Result of a data validation check.
    
    Attributes:
        check_name: Name of the validation check that produced this result
        passed: Whether the validation check passed
        issues: List of issues found during validation
        metadata: Additional metadata about the validation (e.g., statistics)
        cleaning_hints: Hints about potential cleaning operations
    """
    check_name: str
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleaning_hints: List[Any] = field(default_factory=list)  # List of CleaningHint subclasses
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]


@dataclass
class DataValidationConfig:
    """Base configuration class for data validation checks.
    
    Each specific validation check can extend this with its own parameters.
    """
    enabled: bool = True
    severity_on_fail: ValidationSeverity = ValidationSeverity.ERROR


T = TypeVar('T', bound=DataValidationConfig)


class DataValidationCheck(ABC, Generic[T]):
    """Abstract base class for data validation checks.
    
    Each validation check should:
    1. Define its configuration type (extending DataValidationConfig)
    2. Implement the validate method
    3. Have a unique name
    """
    
    def __init__(self, config: Optional[T] = None):
        """Initialize the validation check with optional configuration.
        
        Args:
            config: Configuration for this validation check. If None, uses default config.
        """
        self.config = config or self.get_default_config()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this validation check."""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> T:
        """Get the default configuration for this validation check."""
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        """Perform the validation check on the given DataFrame.
        
        Args:
            df: The DataFrame to validate
            
        Returns:
            DataValidationResult containing the validation outcome
        """
        pass
    
    def _create_result(self, passed: bool, issues: Optional[List[ValidationIssue]] = None, 
                      metadata: Optional[Dict[str, Any]] = None,
                      cleaning_hints: Optional[List[Any]] = None) -> DataValidationResult:
        """Helper method to create a validation result."""
        return DataValidationResult(
            check_name=self.name,
            passed=passed,
            issues=issues or [],
            metadata=metadata or {},
            cleaning_hints=cleaning_hints or []
        ) 