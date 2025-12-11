"""Tests for basic data validation checks."""

import pytest
import pandas as pd
import numpy as np

from pyautocausal.data_validation.base import ValidationSeverity
from pyautocausal.data_validation.checks.basic_checks import (
    NonEmptyDataCheck,
    NonEmptyDataConfig,
    RequiredColumnsCheck,
    RequiredColumnsConfig,
    ColumnTypesCheck,
    ColumnTypesConfig,
    NoDuplicateColumnsCheck,
)


class TestNonEmptyDataCheck:
    """Test the NonEmptyDataCheck validator."""
    
    def test_valid_dataframe(self):
        """Test that a valid DataFrame passes."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        check = NonEmptyDataCheck()
        result = check.validate(df)
        
        assert result.passed
        assert len(result.issues) == 0
        assert result.metadata["n_rows"] == 3
        assert result.metadata["n_columns"] == 2
    
    def test_empty_dataframe(self):
        """Test that an empty DataFrame fails."""
        df = pd.DataFrame()
        check = NonEmptyDataCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 2  # Both rows and columns fail
    
    def test_custom_thresholds(self):
        """Test custom row/column thresholds."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        config = NonEmptyDataConfig(min_rows=5, min_columns=3)
        check = NonEmptyDataCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 2
        assert "5" in result.issues[0].message  # min_rows
        assert "3" in result.issues[1].message  # min_columns


class TestRequiredColumnsCheck:
    """Test the RequiredColumnsCheck validator."""
    
    def test_all_columns_present(self):
        """Test that all required columns are present."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        config = RequiredColumnsConfig(required_columns=["a", "b"])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        
        assert result.passed
        assert len(result.issues) == 0
    
    def test_missing_columns(self):
        """Test detection of missing columns."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        config = RequiredColumnsConfig(required_columns=["a", "b", "c", "d"])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "c" in result.issues[0].message
        assert "d" in result.issues[0].message
        assert result.metadata["missing_columns"] == ["c", "d"]
    
    def test_case_sensitivity(self):
        """Test case sensitivity in column names."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        
        # Case sensitive (default)
        config = RequiredColumnsConfig(required_columns=["a", "b"])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        assert not result.passed
        
        # Case insensitive
        config = RequiredColumnsConfig(required_columns=["a", "b"], case_sensitive=False)
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        assert result.passed


class TestColumnTypesCheck:
    """Test the ColumnTypesCheck validator."""
    
    def test_correct_types(self):
        """Test that correct types pass validation."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        
        config = ColumnTypesConfig(expected_types={
            "int_col": int,
            "float_col": float,
            "str_col": str,
            "bool_col": bool
        })
        check = ColumnTypesCheck(config)
        result = check.validate(df)
        
        assert result.passed
        assert len([i for i in result.issues if i.severity == ValidationSeverity.ERROR]) == 0
    
    def test_incorrect_types_are_convertible(self):
        """Test detection of incorrect but convertible types."""
        df = pd.DataFrame({
            "should_be_int": [1.0, 2.0, 3.0],  # float instead of int
            "should_be_str": [1, 2, 3]  # int instead of str
        })
        
        config = ColumnTypesConfig(expected_types={
            "should_be_int": int,
            "should_be_str": str
        })
        check = ColumnTypesCheck(config)
        result = check.validate(df)
        
        # The check should pass because the types are convertible
        assert result.passed
        # It should generate INFO issues and cleaning hints
        assert len(result.issues) == 2
        assert all(i.severity == ValidationSeverity.INFO for i in result.issues)
        assert len(result.cleaning_hints) == 1
        assert "should_be_int" in result.cleaning_hints[0].type_mapping
        assert "should_be_str" in result.cleaning_hints[0].type_mapping


class TestNoDuplicateColumnsCheck:
    """Test the NoDuplicateColumnsCheck validator."""
    
    def test_no_duplicates(self):
        """Test that unique column names pass."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        check = NoDuplicateColumnsCheck()
        result = check.validate(df)
        
        assert result.passed
        assert len(result.issues) == 0
    
    def test_duplicate_columns(self):
        """Test detection of duplicate column names."""
        # Create DataFrame with duplicate column names
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        check = NoDuplicateColumnsCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert "a" in result.issues[0].message
        assert result.metadata["duplicates"] == ["a"] 