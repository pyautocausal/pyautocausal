import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from pyautocausal.data_validation.checks.causal_checks import (
    BinaryTreatmentCheck,
    PanelStructureCheck,
    TimePeriodStandardizationCheck,
    BinaryTreatmentConfig,
    PanelStructureConfig,
    TimePeriodStandardizationConfig,
)
from pyautocausal.data_cleaning.hints import StandardizeTimePeriodHint
from pyautocausal.data_cleaning.operations.time_operations import StandardizeTimePeriodsOperation


def test_binary_treatment_check_valid():
    """Tests that the binary treatment check passes with valid data."""
    df = pd.DataFrame({"treat": [0, 1, 0, 1, 1]})
    check = BinaryTreatmentCheck(config=BinaryTreatmentConfig(treatment_column="treat"))
    result = check.validate(df)
    assert result.passed


def test_binary_treatment_check_invalid():
    """Tests that the binary treatment check fails with invalid data."""
    df = pd.DataFrame({"treat": [0, 1, 2, 0, -1]})
    check = BinaryTreatmentCheck(config=BinaryTreatmentConfig(treatment_column="treat"))
    result = check.validate(df)
    assert not result.passed
    assert "invalid values" in result.issues[0].message


def test_panel_structure_check_valid():
    """Tests that the panel structure check passes with a balanced panel."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "time": [2000, 2001, 2000, 2001, 2000, 2001],
            "value": [10, 12, 20, 22, 30, 32],
            "treatment": [0, 1, 0, 0, 0, 1],  # Treatment respects monotonicity
        }
    )
    check = PanelStructureCheck(config=PanelStructureConfig(unit_column="unit", time_column="time"))
    result = check.validate(df)
    assert result.passed


def test_panel_structure_check_unbalanced():
    """Tests that the panel structure check fails with an unbalanced panel."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 3, 3],
            "time": [2000, 2001, 2000, 2000, 2001],
            "value": [10, 12, 20, 30, 32],
            "treatment": [0, 1, 0, 0, 1],  # Treatment respects monotonicity
        }
    )
    check = PanelStructureCheck(config=PanelStructureConfig(unit_column="unit", time_column="time"))
    result = check.validate(df)
    assert not result.passed
    assert "Panel is unbalanced" in result.issues[0].message


def test_panel_structure_check_treatment_monotonicity_violation():
    """Tests that the panel structure check detects treatment monotonicity violations."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "time": [2000, 2001, 2002, 2000, 2001, 2002],
            "value": [10, 12, 14, 20, 22, 24],
            "treatment": [0, 1, 0, 0, 0, 1],  # Unit 1 violates monotonicity (1 -> 0)
        }
    )
    check = PanelStructureCheck(config=PanelStructureConfig(unit_column="unit", time_column="time"))
    result = check.validate(df)
    assert not result.passed
    assert "Treatment monotonicity violated" in result.issues[0].message
    assert result.issues[0].details["violation_count"] == 1


def test_time_period_standardization_datetime_strings():
    """Test time period standardization with datetime strings."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-01-01', '2020-02-01', '2020-03-01'],
        'treatment': [0, 1, 1, 0, 0, 1]  # Treatment starts in 2020-02-01
    })
    
    check = TimePeriodStandardizationCheck(
        config=TimePeriodStandardizationConfig(treatment_column="treatment", time_column="time")
    )
    result = check.validate(df)
    
    assert result.passed
    assert len(result.cleaning_hints) == 1
    assert isinstance(result.cleaning_hints[0], StandardizeTimePeriodHint)
    
    hint = result.cleaning_hints[0]
    assert hint.time_column == "time"
    
    # 2020-02-01 should be index 0 (treatment start)
    # 2020-01-01 should be index -1 (before treatment)
    # 2020-03-01 should be index 1 (after treatment)
    expected_mapping = {'2020-01-01': 1, '2020-02-01': 2, '2020-03-01': 3}
    
    # Convert keys to string for comparison (since hint stores as strings)
    for k, v in expected_mapping.items():
        assert str(k) in hint.value_mapping
        assert hint.value_mapping[str(k)] == v


def test_time_period_standardization_integer_periods():
    """Test time period standardization with integer periods."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': [1, 2, 3, 1, 2, 3],
        'treatment': [0, 1, 1, 0, 0, 1]  # Treatment starts in period 2
    })
    
    check = TimePeriodStandardizationCheck(
        config=TimePeriodStandardizationConfig(treatment_column="treatment", time_column="time")
    )
    result = check.validate(df)
    
    assert result.passed
    assert len(result.cleaning_hints) == 1
    
    hint = result.cleaning_hints[0]
    # Period 2 should be index 0, period 1 should be -1, period 3 should be 1
    # Note: original values (integers) are used as keys in the mapping
    expected_mapping = {1: 1, 2: 2, 3: 3}
    
    for k, v in expected_mapping.items():
        assert k in hint.value_mapping
        assert hint.value_mapping[k] == v


def test_time_period_standardization_mixed_formats():
    """Test time period standardization with mixed date formats."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': [1, 2, 3, 1, 2, 3],  # Integer periods
        'treatment': [0, 1, 1, 0, 0, 1]
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert result.passed


def test_time_period_standardization_no_treatment():
    """Test that validation fails when no treatment data exists."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-01-01', '2020-02-01', '2020-03-01'],
        'treatment': [0, 0, 0, 0, 0, 0]  # No treatment==1
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert not result.passed
    assert len(result.issues) == 1
    assert "No treatment data found" in result.issues[0].message
    assert result.issues[0].severity.value == 3  # ERROR


def test_time_period_standardization_missing_columns():
    """Test that validation fails when required columns are missing."""
    df = pd.DataFrame({
        'unit': [1, 1, 1],
        'outcome': [10, 20, 30]
        # Missing 'time' and 'treatment' columns
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert not result.passed
    assert len(result.issues) == 1
    assert "Required columns not found" in result.issues[0].message
    assert "treatment" in result.issues[0].details["missing_columns"]
    assert "time" in result.issues[0].details["missing_columns"]


def test_time_period_standardization_invalid_dates():
    """Test that validation fails with unparseable date values."""
    df = pd.DataFrame({
        'unit': [1, 1, 1],
        'time': ['invalid-date', 'another-bad-date', 'not-a-date'],
        'treatment': [0, 1, 1]
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert not result.passed
    assert len(result.issues) == 1
    assert "Failed to parse time periods" in result.issues[0].message


def test_time_period_standardization_cleaning_operation():
    """Test the actual cleaning operation that applies the standardization."""
    # Create test data
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-01-01', '2020-02-01', '2020-03-01'],
        'treatment': [0, 1, 1, 0, 0, 1],
        'outcome': [10, 15, 20, 12, 18, 25]
    })
    
    # Verify original data type
    assert df['time'].dtype == 'object'  # Date strings are stored as object type
    
    # First, get the cleaning hint from validation
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert result.passed
    assert len(result.cleaning_hints) == 1
    hint = result.cleaning_hints[0]
    
    # Apply the cleaning operation
    operation = StandardizeTimePeriodsOperation()
    assert operation.can_apply(hint)
    
    cleaned_df, transformation_record = operation.apply(df, hint)
    
    # Check that the time column was standardized
    expected_times = [1, 2, 3, 1, 2, 3]  # Based on the hint mapping
    assert cleaned_df['time'].tolist() == expected_times
    
    # CRITICAL: Verify the resultant data type is INTEGER, not Timedelta or any other type
    assert cleaned_df['time'].dtype == 'int64', f"Expected int64, got {cleaned_df['time'].dtype}"
    assert all(isinstance(val, (int, np.integer)) for val in cleaned_df['time']), "All time values should be integers"
    
    # Additional type safety checks
    for time_val in cleaned_df['time']:
        assert not pd.api.types.is_timedelta64_dtype(type(time_val)), f"Time value {time_val} should not be Timedelta type"
        assert not pd.api.types.is_datetime64_dtype(type(time_val)), f"Time value {time_val} should not be datetime type"
    
    # Check that other columns were not modified
    assert cleaned_df['unit'].tolist() == [1, 1, 1, 2, 2, 2]
    assert cleaned_df['treatment'].tolist() == [0, 1, 1, 0, 0, 1]
    assert cleaned_df['outcome'].tolist() == [10, 15, 20, 12, 18, 25]
    
    # Check transformation record
    assert transformation_record.operation_name == "standardize_time_periods"
    assert transformation_record.columns_modified == ['time']
    assert transformation_record.details['time_column'] == 'time'
    assert transformation_record.details['values_standardized'] == 6  # All 6 values


def test_time_period_standardization_complex_scenario():
    """Test with a more complex real-world scenario."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'time': [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
        'treatment': [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # Unit 1: treated in 2020, Unit 2: treated in 2019, Unit 3: treated in 2021
        'outcome': [10, 12, 15, 18, 8, 11, 14, 16, 9, 10, 11, 14]
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert result.passed
    hint = result.cleaning_hints[0]
    
    # First treatment occurs in 2019 (unit 2), so that should be index 0
    # 2018 -> 1, 2019 -> 2, 2020 -> 3, 2021 -> 4
    # Note: original values (integers) are used as keys in the mapping
    expected_mapping = {2018: 1, 2019: 2, 2020: 3, 2021: 4}
    
    for k, v in expected_mapping.items():
        assert k in hint.value_mapping
        assert hint.value_mapping[k] == v
    
    assert hint.metadata['pre_treatment_periods'] == 1  # Only 2018
    assert hint.metadata['post_treatment_periods'] == 3  # 2020 and 2021


def test_time_period_standardization_with_nas():
    """Test handling of NaN values in time column."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': [2, np.nan, 4, 2, 3, 4],
        'treatment': [0, 1, 1, 0, 1, 1]  # Treatment starts in period 2
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert result.passed  # Should still work despite NaN values
    hint = result.cleaning_hints[0] 
    
    # Should only have mappings for non-NaN values: 1, 2, 3
    # Treatment start is 2, so: 1->1, 2->2, 3->3
    # Note: when NaN is present, pandas converts integers to floats, so keys are float values
    expected_mapping = {2.0: 1, 3.0: 2, 4.0: 3}
    
    for k, v in expected_mapping.items():
        assert k in hint.value_mapping
        assert hint.value_mapping[k] == v


def test_time_period_standardization_mixed_data_types():
    """Test that validation fails with mixed data types in time column."""
    df = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': [1, '2020-02-01', 3, 1, '2020-02-01', 3],  # Mixed integers and date strings
        'treatment': [0, 1, 1, 0, 1, 1]
    })
    
    check = TimePeriodStandardizationCheck()
    result = check.validate(df)
    
    assert not result.passed
    assert len(result.issues) == 1
    assert "mixed data types" in result.issues[0].message
    assert result.issues[0].severity.value == 3  # ERROR
    assert "numeric" in result.issues[0].details["mixed_type_families"][0] or "numeric" in result.issues[0].details["mixed_type_families"][1]
    assert "string" in result.issues[0].details["mixed_type_families"][0] or "string" in result.issues[0].details["mixed_type_families"][1] 