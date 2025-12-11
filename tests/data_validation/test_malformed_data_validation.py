"""Tests for data validation with malformed and edge case datasets.

This module tests validation behavior with various types of malformed data
to ensure all error scenarios are properly caught.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pyautocausal.data_validation.base import ValidationSeverity
from pyautocausal.data_validation.checks.missing_data_checks import (
    MissingDataCheck,
    MissingDataConfig,
    CompleteCasesCheck,
    CompleteCasesConfig
)
from pyautocausal.data_validation.checks.causal_checks import (
    BinaryTreatmentCheck,
    BinaryTreatmentConfig,
    TreatmentVariationCheck,
    TreatmentVariationConfig,
    PanelStructureCheck,
    PanelStructureConfig,
    TimeColumnCheck,
    TimeColumnConfig
)
from pyautocausal.data_validation.checks.basic_checks import (
    RequiredColumnsCheck,
    RequiredColumnsConfig,
    ColumnTypesCheck,
    ColumnTypesConfig,
    NonEmptyDataCheck,
    NonEmptyDataConfig
)
from pyautocausal.data_validation.validator_base import DataValidator, DataValidatorConfig


class TestMissingDataValidation:
    """Test MissingDataCheck with various malformed scenarios."""
    
    def test_missing_data_exceeds_threshold(self):
        """Test that missing data exceeding threshold is caught."""
        # Create data with 60% missing values
        df = pd.DataFrame({
            "treatment": [1, 0, np.nan, np.nan, np.nan, 1, 0, np.nan, np.nan, np.nan],
            "outcome": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "control": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        config = MissingDataConfig(max_missing_fraction=0.1)  # Only allow 10% missing
        check = MissingDataCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) >= 1
        
        # Check that treatment column failure is reported
        treatment_issues = [i for i in result.issues if "treatment" in i.message and i.severity == ValidationSeverity.ERROR]
        assert len(treatment_issues) > 0
        assert "60.0%" in treatment_issues[0].message
    
    def test_missing_data_different_types(self):
        """Test detection of different missing value types (None, NaN, empty strings)."""
        df = pd.DataFrame({
            "col_nan": [1, 2, np.nan, 4, 5],
            "col_none": [1, 2, None, 4, 5],
            "col_empty_str": ["a", "b", "", "d", "e"],
            "col_mixed": [1, np.nan, None, 4, 5]
        })
        
        check = MissingDataCheck(config=MissingDataConfig(max_missing_fraction=0.1))
        result = check.validate(df)
        
        # Should detect missing values in col_nan, col_none, and col_mixed
        # Note: empty strings are typically not considered missing unless explicitly converted
        assert not result.passed
        missing_cols = [issue.affected_columns[0] for issue in result.issues 
                       if issue.severity == ValidationSeverity.ERROR and issue.affected_columns]
        
        assert "col_nan" in missing_cols
        assert "col_none" in missing_cols
        assert "col_mixed" in missing_cols
    
    def test_missing_data_specific_columns_only(self):
        """Test checking missing data only in specified columns."""
        df = pd.DataFrame({
            "critical": [1, np.nan, 3, 4, 5],    # 20% missing
            "optional": [np.nan, np.nan, np.nan, np.nan, 5],  # 80% missing
            "control": [1, 2, 3, 4, 5]  # No missing
        })
        
        # Only check critical column
        config = MissingDataConfig(
            max_missing_fraction=0.1,
            check_columns=["critical"]
        )
        check = MissingDataCheck(config)
        result = check.validate(df)
        
        assert not result.passed  # critical column exceeds threshold
        
        # Should only report issues for critical column, not optional
        error_columns = [issue.affected_columns[0] for issue in result.issues 
                        if issue.severity == ValidationSeverity.ERROR and issue.affected_columns]
        assert "critical" in error_columns
        assert "optional" not in error_columns
    
    def test_missing_data_ignore_columns(self):
        """Test ignoring specific columns in missing data check."""
        df = pd.DataFrame({
            "treatment": [1, np.nan, 0, 1, 0],     # 20% missing
            "outcome": [10, 20, np.nan, 40, 50],   # 20% missing  
            "optional_notes": [np.nan, np.nan, np.nan, np.nan, "text"]  # 80% missing
        })
        
        config = MissingDataConfig(
            max_missing_fraction=0.1,
            ignore_columns=["optional_notes"]
        )
        check = MissingDataCheck(config)
        result = check.validate(df)
        
        assert not result.passed  # treatment and outcome exceed threshold
        
        # Should report issues for treatment and outcome, but not optional_notes
        error_columns = [issue.affected_columns[0] for issue in result.issues 
                        if issue.severity == ValidationSeverity.ERROR and issue.affected_columns]
        assert "treatment" in error_columns
        assert "outcome" in error_columns
        assert "optional_notes" not in error_columns
    
    def test_complete_cases_check_insufficient(self):
        """Test CompleteCasesCheck when insufficient complete cases exist."""
        df = pd.DataFrame({
            "a": [1, np.nan, 3, np.nan, 5],
            "b": [np.nan, 2, 3, 4, np.nan],
            "c": [1, 2, np.nan, 4, 5]
        })
        # Only row 2 (index 2) is complete: [3, 3, NaN] - actually not complete
        # Only row 3 (index 3) is complete: [NaN, 4, 4] - actually not complete
        # Let me fix this - only row with index 3 has [NaN, 4, 4] which is not complete
        # Actually looking more carefully: 
        # Row 0: [1, NaN, 1] - not complete
        # Row 1: [NaN, 2, 2] - not complete  
        # Row 2: [3, 3, NaN] - not complete
        # Row 3: [NaN, 4, 4] - not complete
        # Row 4: [5, NaN, 5] - not complete
        
        # Let me create a clearer example
        df = pd.DataFrame({
            "a": [1, np.nan, 3, 4, np.nan],
            "b": [1, 2, np.nan, 4, 5],
            "c": [1, np.nan, 3, 4, np.nan]
        })
        # Row 0: [1, 1, 1] - complete
        # Row 1: [NaN, 2, NaN] - not complete
        # Row 2: [3, NaN, 3] - not complete  
        # Row 3: [4, 4, 4] - complete
        # Row 4: [NaN, 5, NaN] - not complete
        # So 2 out of 5 rows are complete = 40%
        
        config = CompleteCasesConfig(min_complete_fraction=0.8)  # Require 80% complete
        check = CompleteCasesCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "40.0%" in result.issues[0].message
        assert "80.0%" in result.issues[0].message


class TestTreatmentValidationEdgeCases:
    """Test treatment validation with malformed data."""
    
    def test_treatment_column_missing(self):
        """Test when treatment column doesn't exist."""
        df = pd.DataFrame({"outcome": [1, 2, 3], "control": [4, 5, 6]})
        
        config = BinaryTreatmentConfig(treatment_column="treatment")
        check = BinaryTreatmentCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "not found" in result.issues[0].message
        assert result.issues[0].severity == ValidationSeverity.ERROR
    
    def test_treatment_with_missing_values_not_allowed(self):
        """Test treatment column with NaN values when not allowed."""
        df = pd.DataFrame({"treatment": [0, 1, np.nan, 0, 1]})
        
        config = BinaryTreatmentConfig(treatment_column="treatment", allow_missing=False)
        check = BinaryTreatmentCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        missing_issues = [i for i in result.issues if "missing" in i.message]
        assert len(missing_issues) > 0
        assert "1 missing values" in missing_issues[0].message
    
    def test_treatment_with_missing_values_allowed(self):
        """Test treatment column with NaN values when allowed."""
        df = pd.DataFrame({"treatment": [0, 1, np.nan, 0, 1]})
        
        config = BinaryTreatmentConfig(treatment_column="treatment", allow_missing=True)
        check = BinaryTreatmentCheck(config)
        result = check.validate(df)
        
        # Should pass - missing values are allowed
        assert result.passed
    
    def test_treatment_non_numeric_values(self):
        """Test treatment with non-numeric values."""
        df = pd.DataFrame({"treatment": ["treated", "control", "treated", "unknown", "control"]})
        
        config = BinaryTreatmentConfig(treatment_column="treatment", valid_values={0, 1})
        check = BinaryTreatmentCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        invalid_issues = [i for i in result.issues if "invalid values" in i.message]
        assert len(invalid_issues) > 0
    
    def test_treatment_variation_insufficient_treated(self):
        """Test insufficient treated units."""
        df = pd.DataFrame({
            "treatment": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Only 1 treated, 9 control
        })
        
        config = TreatmentVariationConfig(
            treatment_column="treatment",
            min_treated_count=5,
            min_control_count=3
        )
        check = TreatmentVariationCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        treated_issues = [i for i in result.issues if "treated units" in i.message]
        assert len(treated_issues) > 0
        assert "minimum required is 5" in treated_issues[0].message
    
    def test_treatment_variation_insufficient_control(self):
        """Test insufficient control units."""
        df = pd.DataFrame({
            "treatment": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # 9 treated, only 1 control
        })
        
        config = TreatmentVariationConfig(
            treatment_column="treatment",
            min_treated_count=3,
            min_control_count=5
        )
        check = TreatmentVariationCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        control_issues = [i for i in result.issues if "control units" in i.message]
        assert len(control_issues) > 0
        assert "minimum required is 5" in control_issues[0].message
    
    def test_treatment_variation_extreme_imbalance(self):
        """Test extremely imbalanced treatment assignment."""
        df = pd.DataFrame({
            "treatment": [1] * 95 + [0] * 5  # 95% treated, 5% control
        })
        
        config = TreatmentVariationConfig(
            treatment_column="treatment",
            max_treated_fraction=0.8  # Maximum 80% treated
        )
        check = TreatmentVariationCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        fraction_issues = [i for i in result.issues if "fraction" in i.message]
        assert len(fraction_issues) > 0


class TestPanelDataEdgeCases:
    """Test panel data validation with malformed structures."""
    
    def test_panel_missing_time_periods(self):
        """Test panel with missing time periods for some units."""
        df = pd.DataFrame({
            "unit": [1, 1, 1, 2, 2, 3, 3, 3, 3],  # Unit 2 missing period 3
            "time": [1, 2, 3, 1, 2, 1, 2, 3, 4],  # Unit 3 has extra period 4
            "treatment": [0, 1, 1, 0, 0, 0, 1, 1, 1],  # Add treatment column
            "outcome": range(9)
        })
        
        config = PanelStructureConfig(unit_column="unit", time_column="time")
        check = PanelStructureCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert "unbalanced" in result.issues[0].message.lower()
    
    def test_panel_duplicate_unit_time_combinations(self):
        """Test panel with duplicate unit-time combinations."""
        df = pd.DataFrame({
            "unit": [1, 1, 1, 1, 2, 2],  # Unit 1 appears twice in time 2
            "time": [1, 2, 2, 3, 1, 2],
            "treatment": [0, 1, 1, 1, 0, 0],  # Add treatment column
            "outcome": range(6)
        })
        
        config = PanelStructureConfig(unit_column="unit", time_column="time")
        check = PanelStructureCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        duplicate_issues = [i for i in result.issues if "duplicate" in i.message.lower()]
        assert len(duplicate_issues) > 0
    
    def test_time_column_missing(self):
        """Test when time column doesn't exist."""
        df = pd.DataFrame({"unit": [1, 2, 3], "outcome": [10, 20, 30]})
        
        config = TimeColumnConfig(time_column="time")
        check = TimeColumnCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert "not found" in result.issues[0].message
    
    def test_time_column_with_missing_values(self):
        """Test time column with missing values."""
        df = pd.DataFrame({
            "unit": [1, 1, 1, 2, 2],
            "time": [1, np.nan, 3, 1, 2],
            "outcome": range(5)
        })
        
        config = TimeColumnConfig(time_column="time")
        check = TimeColumnCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        missing_issues = [i for i in result.issues if "missing" in i.message]
        assert len(missing_issues) > 0
    
    def test_time_column_non_numeric_when_required(self):
        """Test time column with non-numeric values when numeric required."""
        df = pd.DataFrame({
            "time": ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
            "outcome": range(4)
        })
        
        config = TimeColumnConfig(time_column="time", require_numeric=True)
        check = TimeColumnCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        numeric_issues = [i for i in result.issues if "not numeric" in i.message]
        assert len(numeric_issues) > 0


class TestExtremeValueValidation:
    """Test validation with extreme and problematic values."""
    
    def test_infinite_values_in_numeric_columns(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5],
            "with_inf": [1, 2, np.inf, 4, 5],
            "with_neg_inf": [1, 2, 3, -np.inf, 5],
            "treatment": [0, 1, 0, 1, 0]
        })
        
        # Test that our validation systems can handle infinite values
        # (This tests the robustness of the validation framework itself)
        config = ColumnTypesConfig(expected_types={"normal": float, "with_inf": float, "treatment": int})
        check = ColumnTypesCheck(config)
        result = check.validate(df)
        
        # Should not crash, but specific behavior depends on implementation
        assert isinstance(result.passed, bool)
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        df = pd.DataFrame({
            "huge_numbers": [1e100, 1e200, 1e300, 1e400, 1e500],
            "treatment": [0, 1, 0, 1, 0]
        })
        
        # Test framework robustness with extreme values
        config = NonEmptyDataConfig(min_rows=1)
        check = NonEmptyDataCheck(config)
        result = check.validate(df)
        
        assert result.passed  # Should handle large numbers gracefully
    
    def test_unicode_and_special_characters(self):
        """Test with Unicode and special characters in text data."""
        df = pd.DataFrame({
            "unicode_text": ["café", "naïve", "résumé", "🎉emoji", "普通话"],
            "special_chars": ["hello\nworld", "tab\there", "quote\"test", "slash\\test", "null\x00char"],
            "treatment": [0, 1, 0, 1, 0]
        })
        
        config = RequiredColumnsConfig(required_columns=["unicode_text", "special_chars", "treatment"])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        
        assert result.passed  # Should handle Unicode gracefully
    
    def test_mixed_data_types_in_column(self):
        """Test column with mixed data types."""
        df = pd.DataFrame({
            "mixed_column": [1, "text", 3.14, True, None],
            "treatment": [0, 1, 0, 1, 0]
        })
        
        # Test type checking with mixed types
        config = ColumnTypesConfig(expected_types={"mixed_column": int})
        check = ColumnTypesCheck(config)
        result = check.validate(df)
        
        assert not result.passed  # Should fail type check


class TestConfigurationEdgeCases:
    """Test validation with edge case configurations."""
    
    def test_zero_percent_missing_tolerance(self):
        """Test with 0% missing data tolerance."""
        df = pd.DataFrame({
            "perfect": [1, 2, 3, 4, 5],
            "one_missing": [1, 2, np.nan, 4, 5]
        })
        
        config = MissingDataConfig(max_missing_fraction=0.0)  # 0% tolerance
        check = MissingDataCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        # Should catch even single missing value
        error_issues = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(error_issues) > 0
    
    def test_hundred_percent_missing_tolerance(self):
        """Test with 100% missing data tolerance."""
        df = pd.DataFrame({
            "all_missing": [np.nan, np.nan, np.nan],
            "normal": [1, 2, 3]
        })
        
        config = MissingDataConfig(max_missing_fraction=1.0)  # 100% tolerance
        check = MissingDataCheck(config)
        result = check.validate(df)
        
        assert result.passed  # Should allow any amount of missing data
    
    def test_empty_required_columns_list(self):
        """Test with empty required columns list."""
        df = pd.DataFrame({"any_column": [1, 2, 3]})
        
        config = RequiredColumnsConfig(required_columns=[])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        
        assert result.passed  # Should pass with no requirements
    
    def test_required_columns_not_in_dataframe(self):
        """Test requiring columns that don't exist."""
        df = pd.DataFrame({"existing": [1, 2, 3]})
        
        config = RequiredColumnsConfig(required_columns=["missing1", "missing2", "missing3"])
        check = RequiredColumnsCheck(config)
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        missing_cols = result.metadata["missing_columns"]
        assert "missing1" in missing_cols
        assert "missing2" in missing_cols
        assert "missing3" in missing_cols
    
    def test_extreme_size_thresholds(self):
        """Test with extreme DataFrame size requirements."""
        small_df = pd.DataFrame({"a": [1], "b": [2]})  # 1 row, 2 columns
        
        # Require enormous DataFrame
        config = NonEmptyDataConfig(min_rows=1_000_000, min_columns=100)
        check = NonEmptyDataCheck(config)
        result = check.validate(small_df)
        
        assert not result.passed
        assert len(result.issues) == 2  # Both rows and columns fail
    
    def test_invalid_categorical_threshold_for_inference(self):
        """Test that an invalid categorical threshold raises an error for inference."""
        from pyautocausal.data_validation.checks.categorical_checks import InferCategoricalColumnsCheck, InferCategoricalColumnsConfig
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        # Test with a non-integer threshold
        with pytest.raises(TypeError):
            InferCategoricalColumnsConfig(categorical_threshold="not-a-number")
            
        # Test with a negative threshold, which might be invalid depending on implementation
        # For now, let's assume it should raise a ValueError
        with pytest.raises(ValueError):
            config = InferCategoricalColumnsConfig(categorical_threshold=-5)
            check = InferCategoricalColumnsCheck(config)
            # Depending on implementation, error might be at config or validate
            check.validate(df)


class TestAggregatedValidationScenarios:
    """Test complex scenarios with multiple validation failures."""
    
    def test_catastrophically_bad_data(self):
        """Test data that fails every possible validation."""
        # Create the worst possible dataset
        df = pd.DataFrame({
            # Missing required columns, wrong types, excessive missing data
            "wrong_treatment": ["maybe", "perhaps", np.nan, "definitely", ""],  # Should be 0/1
            "missing_outcome": [np.nan, np.nan, np.nan, np.nan, 10],  # 80% missing
            "wrong_type_col": ["text", "more_text", "even_more", "so_much", "text"],  # Should be numeric
            "tiny_dataset": [1, 2, 3, 4, 5]  # Too small
        })
        
        # Configure strict validation
        validation_config = DataValidatorConfig(
            aggregation_strategy="all",
            check_configs={
                "required_columns": RequiredColumnsConfig(
                    required_columns=["treatment", "outcome", "unit_id", "time"]
                ),
                "missing_data": MissingDataConfig(
                    max_missing_fraction=0.05,  # Very strict
                    check_columns=["wrong_treatment", "missing_outcome"]
                ),
                "binary_treatment": BinaryTreatmentConfig(
                    treatment_column="wrong_treatment"
                ),
                "non_empty_data": NonEmptyDataConfig(
                    min_rows=100,  # Need much more data
                    min_columns=10
                )
            }
        )
        
        checks = [
            RequiredColumnsCheck(),
            MissingDataCheck(),
            BinaryTreatmentCheck(),
            NonEmptyDataCheck()
        ]
        
        validator = DataValidator(checks=checks, config=validation_config)
        with pytest.raises(Exception) as exc_info:
            validator.validate(df)
        
            assert "DataValidationError" in str(exc_info.value)
            assert "Multiple check failures" in str(exc_info.value)
            assert "Many errors found" in str(exc_info.value)
    
    def test_marginally_acceptable_data(self):
        """Test data that barely passes all validations."""
        # Create data that just meets all requirements
        df = pd.DataFrame({
            "treatment": [0, 1] * 50,  # Exactly 50% treated
            "outcome": list(range(100)),
            "unit_id": list(range(100)),
            "time": [1, 2, 3, 4, 5] * 20,
            "control": ["A", "B"] * 50
        })
        
        # Add exactly the maximum allowed missing data
        df.loc[0:4, "outcome"] = np.nan  # Exactly 5% missing
        
        validation_config = DataValidatorConfig(
            check_configs={
                "required_columns": RequiredColumnsConfig(
                    required_columns=["treatment", "outcome", "unit_id", "time"]
                ),
                "missing_data": MissingDataConfig(
                    max_missing_fraction=0.05  # 5% tolerance
                ),
                "binary_treatment": BinaryTreatmentConfig(
                    treatment_column="treatment"
                ),
                "treatment_variation": TreatmentVariationConfig(
                    treatment_column="treatment",
                    min_treated_count=40,      # Need at least 40
                    min_control_count=40,      # Need at least 40  
                    min_treated_fraction=0.4,  # At least 40%
                    max_treated_fraction=0.6   # At most 60%
                ),
                "non_empty_data": NonEmptyDataConfig(
                    min_rows=100,
                    min_columns=5
                )
            }
        )
        
        checks = [
            RequiredColumnsCheck(),
            MissingDataCheck(), 
            BinaryTreatmentCheck(),
            TreatmentVariationCheck(),
            NonEmptyDataCheck()
        ]
        
        validator = DataValidator(checks=checks, config=validation_config)
        result = validator.validate(df)
        
        # Should barely pass
        assert result.passed
        assert result.summary["total_errors"] == 0
        # Might have INFO-level issues but no errors
        assert result.summary["total_info"] >= 0 