"""Tests for the AutoCleaner high-level facade."""

import pandas as pd
import pytest
import numpy as np
import logging
from io import StringIO

from pyautocausal.data_cleaner_interface.autocleaner import AutoCleaner

@pytest.fixture
def sample_data_for_cleaning():
    """Creates a sample DataFrame with various issues to be cleaned."""
    np.random.seed(42)
    n_rows = 100
    data = pd.DataFrame({
        'treatment': np.random.choice([0, 1, 2], n_rows, p=[0.8, 0.15, 0.05]),  # Has an invalid value
        'outcome': np.random.normal(10, 2, n_rows),
        'age': np.random.randint(20, 60, n_rows),
        'city': np.random.choice(['A', 'B', 'C', None], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        're74': np.random.normal(5000, 1000, n_rows),
        're75': np.random.normal(5500, 1000, n_rows),
    })
    data.loc[0, 're74'] = np.nan  # Add a guaranteed NaN
    data = pd.concat([data, data.head(5)], ignore_index=True)  # Add duplicate rows
    data['city'] = data['city'].astype('object') # Ensure it's not categorical yet
    return data

def test_autocleaner_chaining_and_execution(sample_data_for_cleaning):
    """Tests the full pipeline of the AutoCleaner facade."""
    df = sample_data_for_cleaning.copy()
    initial_rows = len(df)
    initial_duplicates = df.duplicated().sum()

    # 1. Configure the AutoCleaner using the fluent API
    autocleaner = (
        AutoCleaner()
        .check_required_columns(required_columns=['treatment', 'outcome', 'age'])
        .check_column_types(expected_types={'age': 'int64', 'treatment': 'int64'})
        .check_for_missing_data(strategy='drop_rows', check_columns=['city', 're74'], max_missing_fraction=0.15)
        .infer_and_convert_categoricals(ignore_columns=['treatment'])
        .drop_duplicates()
    )

    # 2. Execute the cleaning process
    cleaned_df = autocleaner.clean(df)

    # 3. Assertions
    # Check shape and dropped rows (metadata is now logged, not returned)
    assert cleaned_df.shape[0] < initial_rows

    # Check for dropped duplicates
    assert cleaned_df.duplicated().sum() == 0

    # Check for dropped NA values
    assert cleaned_df['city'].isnull().sum() == 0
    assert cleaned_df['re74'].isnull().sum() == 0

    # Check for categorical conversion
    assert pd.api.types.is_categorical_dtype(cleaned_df['city'])

    # Check that original df is untouched
    assert df.shape[0] == initial_rows

    # Note: Metadata is now logged automatically rather than returned

def test_autocleaner_fill_missing_strategy(sample_data_for_cleaning):
    """Tests the 'fill' strategy for missing data."""
    df = sample_data_for_cleaning.copy()
    fill_value = -1

    # if max_missing_fraction is not set, the default is 0.10 which would cause the test to fail
    autocleaner = (
        AutoCleaner()
        .check_for_missing_data(strategy='fill', check_columns=['city'], fill_value=fill_value, max_missing_fraction=0.15)
    )
    cleaned_df = autocleaner.clean(df)

    assert cleaned_df['city'].isnull().sum() == 0
    assert (cleaned_df['city'] == fill_value).any()
    # Note: Metadata is now logged automatically rather than returned 


def test_autocleaner_unified_logging():
    """Test that the unified clean() method properly logs metadata while maintaining backward compatibility."""
    # Set up logging capture
    log_capture = StringIO()
    logger = logging.getLogger("pyautocausal.data_cleaner_interface.autocleaner.AutoCleaner")
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    try:
        # Create test data with various issues
        data = pd.DataFrame({
            'treat': [1, 0, 1, 0, 1, 1, 0],
            'y': [1.2, 2.3, None, 4.5, 5.6, 6.7, 7.8],  # Missing value
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],  # Should be categorical
            'duplicate_col': [1, 1, 1, 1, 1, 1, 1]  # Uniform values
        })
        
        # Add a duplicate row
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        
        # Create autocleaner
        autocleaner = (
            AutoCleaner()
            .check_required_columns(required_columns=["treat", "y"])
            .check_for_missing_data(strategy="drop_rows", check_columns=["y"], max_missing_fraction=0.15)
            .infer_and_convert_categoricals(ignore_columns=["treat", "y"])
            .drop_duplicates(max_duplicate_fraction=0.15) # if max_duplicate_fraction is not set, the default is 0.05 which would cause the test to fail, since we have 1 in 8 rows that are duplicates
        )
        
        # Test the unified clean() method
        cleaned_df = autocleaner.clean(data)
        
        # Verify clean method returns DataFrame only
        assert isinstance(cleaned_df, pd.DataFrame)
        
        # Verify data was properly cleaned
        assert len(cleaned_df) < len(data)  # Should have fewer rows due to missing data + duplicates
        assert cleaned_df['y'].isnull().sum() == 0  # No missing values
        assert not cleaned_df.duplicated().any()  # No duplicates
        
        # Verify categorical conversion (but not for treat/y which are ignored)
        assert pd.api.types.is_categorical_dtype(cleaned_df['category'])
        assert not pd.api.types.is_categorical_dtype(cleaned_df['treat'])
        assert not pd.api.types.is_categorical_dtype(cleaned_df['y'])
        
        # Verify logging behavior
        logged_content = log_capture.getvalue()
        assert "Data cleaning completed" in logged_content
        assert "operations performed" in logged_content
        assert "rows dropped" in logged_content
        
        # Note: Metadata details are captured in logs, not returned
        
    finally:
        # Clean up logging
        logger.removeHandler(handler)


def test_autocleaner_time_period_standardization():
    """Test the time period standardization functionality."""
    # Create test data with different time formats but consistent types
    data = pd.DataFrame({
        'unit': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'time': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
                 '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
                 '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
        'treatment': [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # First treatment in 2020-02-01 (unit 2)
        'outcome': [10, 12, 15, 18, 8, 11, 14, 16, 9, 10, 11, 14]
    })
    
    # Verify original data type
    assert data['time'].dtype == 'object'  # Date strings are stored as object type
    
    # Create autocleaner with time period standardization
    autocleaner = (
        AutoCleaner()
        .standardize_time_periods(treatment_column="treatment", time_column="time")
    )
    
    # Apply cleaning
    cleaned_df = autocleaner.clean(data)
    
    # Verify that time column was standardized
    unique_times = sorted(cleaned_df['time'].unique())
    expected_times = [1,2,3,4]  # 2020-02-01 becomes 0 (first treatment)
    assert unique_times == expected_times
    
    # CRITICAL: Verify the resultant data type is INTEGER, not Timedelta or any other type
    assert cleaned_df['time'].dtype == 'int64', f"Expected int64, got {cleaned_df['time'].dtype}"
    assert all(isinstance(val, (int, np.integer)) for val in cleaned_df['time']), "All time values should be integers"
    
    # Additional type safety checks  
    for time_val in cleaned_df['time']:
        assert not pd.api.types.is_timedelta64_dtype(type(time_val)), f"Time value {time_val} should not be Timedelta type"
        assert not pd.api.types.is_datetime64_dtype(type(time_val)), f"Time value {time_val} should not be datetime type"
    
    # Verify that the mapping is correct for specific cases
    # Original 2020-02-01 should become 0 (first treatment period)
    # Original 2020-01-01 should become -1 (before first treatment)
    # Original 2020-03-01 should become 1 (after first treatment)
    # Original 2020-04-01 should become 2 (two periods after first treatment)
    
    # Check specific rows to verify the mapping
    unit_2_data = cleaned_df[cleaned_df['unit'] == 2].sort_values('time')
    assert unit_2_data['time'].tolist() == [1,2,3,4]
    
    # Verify treatment data is preserved
    assert cleaned_df['treatment'].tolist() == data['treatment'].tolist()
    assert cleaned_df['outcome'].tolist() == data['outcome'].tolist()
    assert cleaned_df['unit'].tolist() == data['unit'].tolist()


def test_autocleaner_time_period_standardization_error_cases():
    """Test that time period standardization properly handles error cases."""
    
    # Test 1: No treatment data (should raise error)
    data_no_treatment = pd.DataFrame({
        'unit': [1, 1, 1],
        'time': [1, 2, 3],
        'treatment': [0, 0, 0]  # No treatment==1
    })
    
    autocleaner = AutoCleaner().standardize_time_periods()
    
    with pytest.raises(Exception) as exc_info:
        autocleaner.clean(data_no_treatment)
    
    assert "No treatment data found" in str(exc_info.value)
    
    # Test 2: Mixed data types (should raise error)
    data_mixed_types = pd.DataFrame({
        'unit': [1, 1, 1],
        'time': [1, '2020-02-01', 3],  # Mixed int and string
        'treatment': [0, 1, 1]
    })
    
    autocleaner = AutoCleaner().standardize_time_periods()
    
    with pytest.raises(Exception) as exc_info:
        autocleaner.clean(data_mixed_types)
    
    assert "mixed data types" in str(exc_info.value)


def test_autocleaner_time_period_standardization_with_integer_periods():
    """Test time period standardization with integer time periods."""
    data = pd.DataFrame({
        'unit': [1, 1, 1, 2, 2, 2],
        'time': [2, 3, 4, 2, 3, 4],
        'treatment': [0, 1, 1, 0, 0, 1],  # Treatment starts in period 2
        'outcome': [10, 15, 20, 12, 18, 25]
    })
    
    autocleaner = AutoCleaner().standardize_time_periods()
    cleaned_df = autocleaner.clean(data)
    
    # Expected mapping: 2->1, 3->2, 4->3
    expected_times = [1, 2, 3, 1, 2, 3]
    assert cleaned_df['time'].tolist() == expected_times 