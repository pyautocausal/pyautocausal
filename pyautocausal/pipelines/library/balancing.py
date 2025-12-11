import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from pyautocausal.pipelines.library.base_weighter import BaseWeighter
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.specifications import DiDSpec, BaseSpec
from typing import List


@make_transformable
def compute_synthetic_control_weights(spec: DiDSpec,
                    unit_col: str = 'id_unit', 
                    time_col: str = 't',
                    outcome_col: str = 'y',
                    treatment_col: str = 'treat',
                    covariates: Optional[List[str]] = None,
                    pre_treatment_periods: Optional[List] = None,
                    **kwargs) -> DiDSpec:
        """
        Compute synthetic control weights.
        
        Args:
            spec: DiD specification dataclass with data and column information
            unit_col: Column name for unit identifier (overrides spec.unit_col if provided)
            time_col: Column name for time period (overrides spec.time_col if provided)
            outcome_col: Column name for outcome variable (overrides spec.outcome_col if provided)
            treatment_col: Column name for treatment indicator (overrides spec.treatment_cols[0] if provided)
            covariates: List of covariates to use (if None, uses outcome in pre-treatment periods)
            pre_treatment_periods: List of pre-treatment periods (if None, inferred from data)
            **kwargs: Additional parameters
            
        Returns:
            DiDSpec with synthetic control weights added as a column to the data
        """
        # Extract data from inputs
        if isinstance(spec, DiDSpec) and hasattr(spec, 'data'):
            df = spec.data
        else:
            raise ValueError("Inputs must contain a 'data' field with a DataFrame")
        
        # Use values from spec by default, but allow overriding with provided parameters
        unit_col = spec.unit_col
        time_col = spec.time_col
        outcome_col = spec.outcome_col
        
        # sort of hacky, think about how to elegantly handle multiple vs one treatment column
        treatment_col = spec.treatment_cols[0] 

        required_columns = [unit_col, time_col, outcome_col, treatment_col]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Identify treated unit(s)
        treated_units = df.loc[df[treatment_col] == 1, unit_col].unique()
        if len(treated_units) == 0:
            raise ValueError("No treated units found in data")
        if len(treated_units) > 1:
            raise ValueError("Synthetic control requires exactly one treated unit")
        
        treated_unit = treated_units[0]
        
        # Identify control periods for the treated unit
        control_periods = df[(df[unit_col] == treated_unit) & (df[treatment_col] == 0)][time_col].unique()
        
        # Check if we have other control units
        other_control_units = df[(df[unit_col] != treated_unit) & (df[treatment_col] == 0)][unit_col].unique()
        
        # If there are no other control units and no control periods for the treated unit, raise an error
        if len(other_control_units) == 0 and len(control_periods) == 0:
            raise ValueError("No control units or control periods found for synthetic control")
        
        # If treated unit is also a control unit in some periods, use those periods
        if len(control_periods) > 0:
            control_units = np.array([treated_unit])
        else:
            # Otherwise use other control units
            control_units = other_control_units
        
        # Determine pre-treatment periods
        if pre_treatment_periods is None:
            # Get all periods before treatment starts for the treated unit
            treat_start = df[(df[unit_col] == treated_unit) & (df[treatment_col] == 1)][time_col].min()
            pre_treatment_periods = sorted(df[df[time_col] < treat_start][time_col].unique())
        
        if len(pre_treatment_periods) == 0:
            raise ValueError("No pre-treatment periods identified")
        
        # Get pre-treatment outcomes for treated unit
        treated_pre = df[(df[unit_col] == treated_unit) & 
                         (df[time_col].isin(pre_treatment_periods))][outcome_col].values
        
        # Get pre-treatment outcomes for control units
        control_pre = np.zeros((len(control_units), len(pre_treatment_periods)))
        
        for i, unit in enumerate(control_units):
            unit_data = df[(df[unit_col] == unit) & 
                           (df[time_col].isin(pre_treatment_periods))][outcome_col].values
            control_pre[i, :] = unit_data
        
        # For single unit case, use all 1.0 for weights
        if len(control_units) == 1 and control_units[0] == treated_unit:
            unit_weights = np.ones(len(control_units))
        else:
            # For multiple control units, use equal weights
            unit_weights = np.ones(len(control_units)) / len(control_units)
        
        # Compute synthetic control as weighted average of control units
        synthetic_pre = unit_weights @ control_pre
        
        # Calculate fit metrics
        pre_rmse = np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2))
        pre_mae = np.mean(np.abs(treated_pre - synthetic_pre))
        
        # Create full weights array for all units in the dataset
        all_units = df[unit_col].unique()
        all_weights = np.zeros(len(all_units))
        
        # Set weights for control units
        for i, unit in enumerate(control_units):
            unit_idx = np.where(all_units == unit)[0][0]
            all_weights[unit_idx] = unit_weights[i]
        
        # Create a new copy of inputs to avoid modifying the original
        result_df = df.copy()
        
        # Expand weights to match DataFrame length
        expanded_weights = np.zeros(len(result_df))
        for i, unit in enumerate(all_units):
            unit_mask = result_df[unit_col] == unit
            expanded_weights[unit_mask] = all_weights[i]
        
        # Store expanded weights in the result
        result_df['weights'] = expanded_weights
        
        # Create a diagnostic report to help with troubleshooting
        sc_diagnostics = {
            'treated_unit': treated_unit,
            'control_units': control_units.tolist(),
            'pre_treatment_periods': pre_treatment_periods,
            'pre_rmse': pre_rmse,
            'pre_mae': pre_mae,
            'num_units': len(all_units),
            'num_controls': len(control_units)
        }
        
        # Return a new DiDSpec with the updated dataframe
        return DiDSpec(
            data=result_df,
            formula=spec.formula,
            outcome_col=spec.outcome_col,
            treatment_cols=spec.treatment_cols,
            control_cols=spec.control_cols,
            time_col=spec.time_col,
            unit_col=spec.unit_col,
            post_col=spec.post_col,
            include_unit_fe=spec.include_unit_fe,
            include_time_fe=spec.include_time_fe
        )


@make_transformable
def compute_balance_tests(spec: Union[DiDSpec, 'StaggeredDiDSpec', 'EventStudySpec', 'CrossSectionalSpec']) -> Union[DiDSpec, 'StaggeredDiDSpec', 'EventStudySpec', 'CrossSectionalSpec']:
    """
    Compute balance tests on control variables for any specification type.
    
    For panel data specifications (DiD, Staggered, Event Study):
    - Tests balance in the last pre-treatment period
    
    For cross-sectional specifications:
    - Tests balance across all data
    
    Args:
        spec: Any specification object (DiDSpec, StaggeredDiDSpec, etc.)
        
    Returns:
        Same specification object with balance test results added
    """
    # Extract balance testing data based on specification type
    balance_data = _extract_balance_data(spec)
    
    # Get control variables
    control_cols = getattr(spec, 'control_cols', [])
    if not control_cols:
        # Auto-detect control columns
        exclude_cols = [spec.outcome_col, spec.treatment_cols[0]]
        if hasattr(spec, 'unit_col'):
            exclude_cols.append(spec.unit_col)
        if hasattr(spec, 'time_col'):
            exclude_cols.append(spec.time_col)
        
        numeric_cols = balance_data.select_dtypes(include=[np.number]).columns
        control_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Compute balance statistics
    balance_stats = []
    treatment_col = spec.treatment_cols[0]
    
    for covariate in control_cols:
        # Split by treatment status
        treated_data = balance_data[balance_data[treatment_col] == 1][covariate]
        control_data = balance_data[balance_data[treatment_col] == 0][covariate]
        
        # Basic statistics
        treated_mean = treated_data.mean()
        treated_std = treated_data.std()
        treated_n = len(treated_data)
        
        control_mean = control_data.mean()
        control_std = control_data.std()
        control_n = len(control_data)
        
        # Difference and statistical test
        diff = treated_mean - control_mean
        t_stat, p_value = stats.ttest_ind(treated_data, control_data, equal_var=False)
        
        # Standard error for confidence intervals
        se_diff = np.sqrt((treated_std**2 / treated_n) + (control_std**2 / control_n))
        
        balance_stats.append({
            'covariate': covariate,
            'treated_mean': treated_mean,
            'treated_std': treated_std,
            'treated_n': treated_n,
            'control_mean': control_mean,
            'control_std': control_std,
            'control_n': control_n,
            'diff': diff,
            'se_diff': se_diff,
            't_stat': t_stat,
            'p_value': p_value
        })
    
    # Add balance results to spec
    spec.balance_stats = pd.DataFrame(balance_stats)
    spec.balance_data = balance_data
    
    return spec


def _extract_balance_data(spec) -> pd.DataFrame:
    """
    Extract appropriate data for balance testing based on specification type.
    
    Args:
        spec: Any specification object
        
    Returns:
        DataFrame with data appropriate for balance testing
    """
    data = spec.data
    
    # Check if we have panel data (DiD, Staggered, Event Study)
    if hasattr(spec, 'time_col') and hasattr(spec, 'unit_col'):
        # Panel data: extract pre-treatment data
        return _extract_pretreatment_data(data, spec.treatment_cols[0], spec.unit_col, spec.time_col)
    else:
        # Cross-sectional data: use all data
        return data.copy()


def _extract_pretreatment_data(data: pd.DataFrame, treatment_col: str, unit_col: str, time_col: str) -> pd.DataFrame:
    """
    Extract pre-treatment data for balancing tests.
    
    For each unit, finds the last period before treatment begins.
    For never-treated units, uses the last available period.
    
    Args:
        data: Full panel DataFrame
        treatment_col: Treatment column name
        unit_col: Unit identifier column
        time_col: Time column name
        
    Returns:
        DataFrame with one observation per unit from pre-treatment period
    """
    balance_data = []
    
    for unit in data[unit_col].unique():
        unit_data = data[data[unit_col] == unit].sort_values(time_col)
        
        # Find first treatment period for this unit
        treated_periods = unit_data[unit_data[treatment_col] == 1]
        
        if len(treated_periods) > 0:
            # Unit is treated at some point - get last pre-treatment period
            first_treatment_period = treated_periods[time_col].min()
            pre_treatment_data = unit_data[unit_data[time_col] < first_treatment_period]
            
            if len(pre_treatment_data) > 0:
                # Use the last pre-treatment period
                last_pre_period = pre_treatment_data.iloc[-1:].copy()
                # Set treatment to 1 to indicate this unit will be treated
                last_pre_period[treatment_col] = 1
                balance_data.append(last_pre_period)
        else:
            # Never-treated unit - use last available period
            last_period = unit_data.iloc[-1:].copy()
            # Ensure treatment is 0 for never-treated units
            last_period[treatment_col] = 0
            balance_data.append(last_period)
    
    if not balance_data:
        raise ValueError("No pre-treatment data available for balance testing")
    
    return pd.concat(balance_data, ignore_index=True)

