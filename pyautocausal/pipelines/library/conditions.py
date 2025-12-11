from typing import Callable, Any
import pandas as pd
import numpy as np
from pyautocausal.persistence.parameter_mapper import make_transformable


def is_continuous(series: pd.Series) -> bool:
    """
    Check if a series contains continuous values (not just 0s and 1s).
    
    Args:
        series: pandas Series to check
        
    Returns:
        bool: True if series contains values other than 0 and 1
    """
    unique_values = series.unique()
    return (
        len(unique_values) > 2 or 
        (len(unique_values) == 2 and not all(v in [0, 1] for v in unique_values))
    )


def index_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to calculate relative time to treatment
    
    Args:
        df: DataFrame with 't', 'treat', and 'id_unit' columns
        
    Returns:
        DataFrame with added 'relative_time' column
        
    Raises:
        ValueError: If required columns are missing
    """
    if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
        raise ValueError("DataFrame must contain 't', 'treat', and 'id_unit' columns")
        
    # Find first treatment period for each unit
    treatment_starts = df[df['treat'] == 1].groupby('id_unit')['t'].min()
    
    # Create relative time column
    df = df.copy()
    df['relative_time'] = df.apply(
        lambda row: row['t'] - treatment_starts.get(row['id_unit'], np.inf)
        if row['id_unit'] in treatment_starts.index else -np.inf,
        axis=1
    )
    return df


# Time-related functions (moved from TimeConditions class)

@make_transformable
def has_minimum_periods(df: pd.DataFrame, min_periods: int = 3) -> bool:
    """
    Check if dataset has at least the specified number of time periods.
    
    Args:
        df: DataFrame with 't' column
        min_periods: Minimum number of periods required
        
    Returns:
        bool: True if dataset has at least min_periods time periods
    """
    if 't' not in df.columns:
        return False  # Single period case
    return df['t'].nunique() >= min_periods


@make_transformable
def has_multiple_periods(df: pd.DataFrame) -> bool:
    """
    Check if dataset has multiple time periods.
    
    Args:
        df: DataFrame with 't' column
        
    Returns:
        bool: True if dataset has multiple time periods
    """
    if 't' not in df.columns:
        return False  # Single period case
    return df['t'].nunique() > 1


@make_transformable
def has_single_period(df: pd.DataFrame) -> bool:
    """
    Check if dataset has only a single time period.
    
    Args:
        df: DataFrame with 't' column
        
    Returns:
        bool: True if dataset has only a single time period
    """
    if 't' not in df.columns:
        return True  # No time column means single period
    return df['t'].nunique() <= 1


@make_transformable
def has_staggered_treatment(df: pd.DataFrame) -> bool:
    """
    Check if treatment timing is staggered across units.

    Args:
        df: DataFrame with 'treat', 'id_unit', and 't' columns

    Returns:
        bool: True if treatment is staggered across units
    """
    required_cols = {'treat', 'id_unit', 't'}
    if not required_cols.issubset(df.columns):
        return False

    # Get treatment start time for each unit, if any
    treatment_starts = (
        df[df['treat'] == 1]
        .groupby('id_unit')['t']
        .min()
    )
    # More than one unique treatment start → staggered
    return treatment_starts.nunique(dropna=True) > 1


@make_transformable
def has_minimum_pre_periods(df: pd.DataFrame, min_periods: int = 3) -> bool:
    """
    Check if there are enough pre-treatment periods.
    
    Args:
        df: DataFrame with 't', 'treat', and 'id_unit' columns
        min_periods: Minimum number of pre-treatment periods required
        
    Returns:
        bool: True if dataset has at least min_periods pre-treatment periods
    """
    if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
        return False
    df = index_relative_time(df)
    return df[df['relative_time'] < 0]['relative_time'].nunique() >= min_periods


@make_transformable
def has_minimum_post_periods(df: pd.DataFrame, min_periods: int = 2) -> bool:
    """
    Check if there are enough post-treatment periods.
    
    Args:
        df: DataFrame with 't', 'treat', and 'id_unit' columns
        min_periods: Minimum number of post-treatment periods required
        
    Returns:
        bool: True if dataset has at least min_periods post-treatment periods
    """
    if not all(col in df.columns for col in ['t', 'treat', 'id_unit']):
        return False
    df = index_relative_time(df)
    return df[df['relative_time'] >= 0]['relative_time'].nunique() >= min_periods


# Treatment-related functions (moved from TreatmentConditions class)

@make_transformable
def has_multiple_treated_units(df: pd.DataFrame) -> bool:
    """
    Check if dataset has multiple treated units.
    
    Args:
        df: DataFrame with 'treat' and 'id_unit' columns
        
    Returns:
        bool: True if dataset has multiple treated units
    """
    return len(df[df['treat']==1]['id_unit'].unique()) > 1

@make_transformable
def has_single_treated_unit(df: pd.DataFrame) -> bool:
    """
    Check if dataset has exactly one treated unit.
    
    Args:
        df: DataFrame with 'treat' and 'id_unit' columns
        
    Returns:
        bool: True if dataset has exactly one treated unit
    """
    return len(df[df['treat']==1]['id_unit'].unique()) == 1

@make_transformable
def has_never_treated_units(df: pd.DataFrame) -> bool:
    """
    Check if dataset has never treated units.
    
    Args:
        df: DataFrame with 'treat' and 'id_unit' columns
        
    Returns:
        bool: True if dataset has units that are never treated
    """
    all_units = df['id_unit'].unique()
    ever_treated_units = df[df['treat'] == 1]['id_unit'].unique()
    return len(set(all_units) - set(ever_treated_units)) > 0

@make_transformable
def has_sufficient_never_treated_units(df: pd.DataFrame, min_percentage: float = 0.1) -> bool:
    """
    Check if dataset has sufficient never-treated units for Callaway & Sant'Anna estimation.
    
    A dataset should have a sufficient number of never-treated units to serve as a proper
    control group for the Callaway & Sant'Anna estimator. This function checks if at least
    a specified percentage of units are never treated.
    
    Args:
        df: DataFrame with 'treat' and 'id_unit' columns
        min_percentage: Minimum percentage of units that should be never-treated (default: 0.1 or 10%)
        
    Returns:
        bool: True if dataset has sufficient never-treated units
    """
    if not {'treat', 'id_unit'}.issubset(df.columns):
        return False
        
    all_units = df['id_unit'].unique()
    total_units = len(all_units)
    
    ever_treated_units = df[df['treat'] == 1]['id_unit'].unique()
    never_treated_units = set(all_units) - set(ever_treated_units)
    
    never_treated_count = len(never_treated_units)
    never_treated_percentage = never_treated_count / total_units
    
    return never_treated_percentage >= min_percentage

@make_transformable
def has_covariate_imbalance(df: pd.DataFrame, threshold: float = 0.5, imbalance_threshold: float = 0.25) -> bool:
    """
    Check if there is covariate imbalance between treatment and control groups.
    
    Args:
        df: DataFrame with 'treat' column and covariates
        threshold: Treatment threshold for continuous treatment
        imbalance_threshold: Standardized difference threshold above which covariates are considered imbalanced
        
    Returns:
        bool: True if covariates are imbalanced
    """
    if 'treat' not in df.columns:
        raise ValueError("DataFrame must contain 'treat' column")
            
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    covariates = [col for col in numeric_cols if col not in ['treat', 't', 'id_unit']]
    
    if not covariates:
        return False
    
    # Define treatment groups based on continuous or binary treatment
    if is_continuous(df['treat']):
        treated_mask = df['treat'] > threshold
    else:
        treated_mask = df['treat'] == 1
        
    for col in covariates:
        treated_mean = df[treated_mask][col].mean()
        control_mean = df[~treated_mask][col].mean()
        treated_var = df[treated_mask][col].var()
        control_var = df[~treated_mask][col].var()
        
        std_diff = abs(treated_mean - control_mean) / np.sqrt((treated_var + control_var) / 2)
        
        if std_diff > imbalance_threshold:
            return True
            
    return False


def has_binary_treatment_and_outcome(df: pd.DataFrame) -> bool:
    """Enhanced check with safety validations."""
    # Check for required columns
    if 'y' not in df.columns or 'treat' not in df.columns:
        return False
    
    # Ensure no time column (cross-sectional only)
    if 't' in df.columns:
        return False
    
    # Verify binary
    return df['y'].nunique() == 2 and df['treat'].nunique() == 2


