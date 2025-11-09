import pandas as pd
import statsmodels.api as sm
import io
from typing import Optional, Any, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from pyautocausal.persistence.output_config import OutputType, OutputConfig
from pyautocausal.persistence.parameter_mapper import make_transformable
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from pyautocausal.pipelines.library.synthdid.utils import panel_matrices
import numpy as np
import patsy
from enum import Enum

@dataclass
class BaseSpec:
    """Base specification with common fields."""
    data: pd.DataFrame
    formula: str


@dataclass
class CrossSectionalSpec(BaseSpec):
    """Univariate specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]

@dataclass
class DiDSpec(BaseSpec):
    """Difference-in-Differences specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    post_col: str
    include_unit_fe: bool
    include_time_fe: bool
    model: Optional[Any] = None

@dataclass
class EventStudySpec(BaseSpec):
    """Event Study specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    relative_time_col: str
    event_cols: List[str]
    reference_period: int
    model: Optional[Any] = None


@dataclass
class StaggeredDiDSpec(BaseSpec):
    """Staggered Difference-in-Differences specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    treatment_time_col: str
    model: Optional[Any] = None


@dataclass
class SynthDIDSpec(BaseSpec):
    """Synthetic Difference-in-Differences specification."""
    outcome_col: str
    treatment_cols: List[str]
    control_cols: List[str]
    time_col: str
    unit_col: str
    Y: np.ndarray  # Outcome matrix (N x T)
    N0: int  # Number of control units
    T0: int  # Number of pre-treatment periods
    X: Optional[np.ndarray] = None  # Covariates matrix (N x T x C)
    model: Optional[Any] = None

@dataclass
class UpliftSpec(BaseSpec):
    """Uplift modeling specification for binary treatment and outcomes."""
    outcome_col: str
    treatment_cols: List[str]  # List to match existing pattern
    control_cols: List[str]    # Renamed from feature_cols to match convention
    unit_col: str              # Added to match existing specs
    model: Optional[Any] = None  # IMPORTANT: Required for compatibility with existing pipeline
    propensity_score: Optional[np.ndarray] = None
    cate_estimates: Optional[Dict[str, np.ndarray]] = None
    ate_estimate: Optional[float] = None
    ate_ci: Optional[Tuple[float, float]] = None
    model_type: Optional[str] = None
    models: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, float]] = None


def validate_and_prepare_data(
    data: pd.DataFrame,
    outcome_col: str,
    treatment_cols: List[str],
    required_columns: List[str] = None,
    control_cols: Optional[List[str]] = None,
    excluded_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and prepare dataframe for specification creation.
    
    Args:
        data: DataFrame to validate and prepare
        outcome_col: Name of outcome column
        treatment_cols: Names of treatment columns
        required_columns: List of additional required columns (beyond outcome and treatment)
        control_cols: Explicitly provided control columns (if None, will be auto-determined)
        excluded_cols: Columns to exclude when auto-determining control columns
        
    Returns:
        Tuple of (cleaned dataframe, control columns list)
    """
    # Handle control_cols=None appropriately
    if control_cols is None:
        control_cols = []  # Default empty list instead of raising error
    
    # Build complete list of required columns
    all_required = [outcome_col] + treatment_cols
    if required_columns:
        all_required.extend(required_columns)
    
    # Check if required columns exist
    missing_columns = [col for col in all_required if col not in data.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing the following required columns: {missing_columns}. All required columns are: {all_required}")
    
    # Clean data (make a copy to avoid modifying the original)
    cleaned_data = data.copy().dropna()
    
    # Determine control columns if not provided
    if not control_cols:
        # Get numeric columns as potential controls
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        # Determine columns to exclude
        to_exclude = [outcome_col] + treatment_cols
        if excluded_cols:
            to_exclude.extend(excluded_cols)
            
        # Filter numeric columns to only include those not in excluded list
        control_cols = [col for col in numeric_cols if col not in to_exclude]
        
    return cleaned_data, control_cols


@make_transformable
def create_uplift_specification(
    data: pd.DataFrame,
    outcome_col: str = 'y',
    treatment_cols: List[str] = ['treat'],
    unit_col: str = 'id_unit',
    control_cols: Optional[List[str]] = None
) -> UpliftSpec:
    """
    Create uplift modeling specification following PyAutoCausal patterns.
    
    Uses validate_and_prepare_data to handle control column auto-detection
    and data validation, matching existing specification functions.
    """
    # Validate and prepare data using existing utility
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols,
        excluded_cols=[unit_col]  # Exclude ID from controls
    )
    
    # Additional validation for uplift modeling
    treatment_col = treatment_cols[0]  # Use first treatment
    if data[outcome_col].nunique() != 2:
        raise ValueError(f"Outcome {outcome_col} must be binary for uplift modeling")
    if data[treatment_col].nunique() != 2:
        raise ValueError(f"Treatment {treatment_col} must be binary for uplift modeling")
    
    formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(control_cols)
    
    return UpliftSpec(
        data=data,
        formula=formula,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols,
        unit_col=unit_col
    )


@make_transformable
def create_cross_sectional_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    control_cols: Optional[List[str]] = None
) -> BaseSpec:
    """
    Create a standard regression specification.
    
    Args:
        data: DataFrame with outcome, treatment, and controls
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names (only first is used for now)
        control_cols: List of control variable columns
        
    Returns:
        BaseSpec object with specification information
    """
    # TODO:Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols
    )
    
    # Standard-specific validation: check if treatment column is binary
    n_unique_treatment = data[treatment_col].nunique()
    if n_unique_treatment > 2:
        raise ValueError(f"Treatment column {treatment_col} must be binary")
    elif n_unique_treatment < 2:
        raise ValueError(f"Treatment column {treatment_col} must have at least two unique values")  
    
    # Make treatment column binary (0/1)
    data[treatment_col] = np.where(data[treatment_col] == data[treatment_col].unique()[0], 0, 1)
    
    # Create formula
    formula = (f"{outcome_col} ~ {treatment_col} + " + " + ".join(control_cols) 
              if control_cols else f"{outcome_col} ~ {treatment_col}")
    
    # return CrossSectionalSpec with treatment_cols
    return CrossSectionalSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,  # Use the list
        control_cols=control_cols,
        data=data,
        formula=formula
    )


@make_transformable
def create_did_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    post_col: Optional[str] = None,
    treatment_time_col: Optional[str] = None,
    include_unit_fe: bool = True,
    include_time_fe: bool = True,
    control_cols: Optional[List[str]] = None
) -> DiDSpec:
    """
    Create a DiD specification.
    
    Args:
        df: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_col: Name of treatment column
        time_col: Name of time column
        unit_col: Name of unit identifier column
        post_col: Name of post-treatment indicator column
        treatment_time_col: Name of treatment timing column
        include_unit_fe: Whether to include unit fixed effects
        include_time_fe: Whether to include time fixed effects
        control_cols: List of control variable columns
        
    Returns:
        DiDSpec object with DiD specification information
    """
    # TODO: Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols,
        excluded_cols=[time_col, unit_col]
    )
    
    # Create post-treatment indicator if not provided
    if post_col is None:
        if treatment_time_col is not None:
            # Ensure never-treated units have 0 in treatment_time_col, not NaN
            data[treatment_time_col] = data[treatment_time_col].fillna(0)
            # Create post indicator based on treatment timing
            data['post'] = (data[time_col] >= data[treatment_time_col]).astype(int)
            post_col = 'post'
        else:
            # Try to infer post periods for treated units
            treat_start = data[data[treatment_col] == 1][time_col].min()
            data['post'] = (data[time_col] >= treat_start).astype(int) if pd.notna(treat_start) else 0
            post_col = 'post'
    
    # Create interaction term
    data['treat_post'] = data[treatment_col] * data[post_col]
    
    # Construct formula
    formula_parts = [outcome_col, "~", "treat_post"]
    
    if not include_unit_fe and not include_time_fe:
        # Base model with just the interaction and controls
        formula_parts.extend(["+", treatment_col, "+", post_col])
        if control_cols:
            formula_parts.extend(["+ " + " + ".join(control_cols)])
    else:
        # Model with fixed effects using linearmodels syntax
        if include_unit_fe:
            formula_parts.append("+ EntityEffects")
        if include_time_fe:
            formula_parts.append("+ TimeEffects")
        if control_cols:
            formula_parts.extend(["+ " + " + ".join(control_cols)])
    
    formula = " ".join(formula_parts)
    
    print(formula)

    # Create and return specification
    return DiDSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        time_col=time_col,
        unit_col=unit_col,
        post_col=post_col,
        include_unit_fe=include_unit_fe,
        include_time_fe=include_time_fe,
        control_cols=control_cols,
        data=data,
        formula=formula
    )


@make_transformable
def create_event_study_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    reference_period: int = -1,
    relative_time_col: Optional[str] = None,
    pre_periods: int = 3,
    post_periods: int = 3,
    control_cols: Optional[List[str]] = None
) -> EventStudySpec:
    """
    Create an Event Study specification.
    
    Args:
        data: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names
        time_col: Name of time column
        unit_col: Name of unit identifier column
        relative_time_col: Name of relative time column
        pre_periods: Number of pre-treatment periods to include
        post_periods: Number of post-treatment periods to include
        control_cols: List of control variable columns
        
    Returns:
        EventStudySpec object with event study specification information
    """
    # Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    # Ensure 'post' is excluded if it exists to avoid issues with event dummies + time FEs
    current_excluded_cols = [time_col, unit_col]
    if 'post' in data.columns:
        current_excluded_cols.append('post')

    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols, # Pass user-provided or None
        excluded_cols=current_excluded_cols
    )
    
    # Infer treatment timing from the data (same approach as staggered specification)
    treatment_times = data[data[treatment_col] == 1].groupby(unit_col)[time_col].min()
    data['treatment_time'] = data[unit_col].map(treatment_times)

    # Create relative time variable (same approach as staggered specification)
    data['relative_time'] = pd.NA
    mask = ~data['treatment_time'].isna()
    data.loc[mask, 'relative_time'] = (data.loc[mask, time_col] - data.loc[mask, 'treatment_time']).astype(int)
    data.loc[~mask, 'relative_time'] = 0  # Never treated units get 0
    
    # Filter to include only the specified pre and post periods
    if pre_periods > 0 or post_periods > 0:
        min_period = -pre_periods if pre_periods > 0 else None
        max_period = post_periods if post_periods > 0 else None
        
        # Keep rows that either have relative_time==0 (never treated) or are within the specified range
        if min_period is not None and max_period is not None:
            data = data[(data['relative_time'] == 0) | 
                       ((data['relative_time'] >= min_period) & (data['relative_time'] <= max_period))]
        elif min_period is not None:
            data = data[(data['relative_time'] == 0) | (data['relative_time'] >= min_period)]
        elif max_period is not None:
            data = data[(data['relative_time'] == 0) | (data['relative_time'] <= max_period)]

    print(data['relative_time'].value_counts())
    
    # Create event time dummies manually to avoid issues with negative signs in column names
    # Get unique relative time values (excluding the reference period)
    unique_times = sorted([t for t in data['relative_time'].dropna().unique() if t != reference_period])
    
    event_cols = []
    for t in unique_times:
        # Create formula-friendly column names
        if t < 0:
            col_name = f'event_pre{abs(t)}'  # event_pre1, event_pre2, etc.
        elif t > 0:
            col_name = f'event_post{t}'      # event_post1, event_post2, etc.
        else:
            col_name = 'event_period0'       # event_period0 for t=0
        
        # Create the dummy variable
        data[col_name] = (data['relative_time'] == t).astype(int)
        event_cols.append(col_name)
    
    # Construct formula
    formula_parts = [outcome_col, "~"]
    if event_cols:
        formula_parts.append(" + ".join(event_cols))
    else:
        formula_parts.append("1")  # Intercept only if no event columns
    formula_parts.append("+ EntityEffects")
    formula_parts.append("+ TimeEffects")
    
    if control_cols:
        formula_parts.append("+ " + " + ".join(control_cols))
    
    formula = " ".join(formula_parts)
    
    print(formula)

    # Create and return the EventStudySpec object
    return EventStudySpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols,
        time_col=time_col,
        unit_col=unit_col,
        relative_time_col='relative_time',
        event_cols=event_cols,
        reference_period=reference_period,
        data=data,
        formula=formula
    )


@make_transformable
def create_staggered_did_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    control_cols: Optional[List[str]] = None,
    pre_periods: int = 4,
    reference_period: int = -1
) -> StaggeredDiDSpec:
    """
    Create a Staggered DiD specification.
    
    Follows the cohort-based approach for staggered treatment adoption.
    
    Args:
        data: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names
        time_col: Name of time column
        unit_col: Name of unit identifier column
        treatment_time_col: Name of treatment timing column
        control_cols: List of control variable columns
        pre_periods: Number of pre-treatment periods to include for testing parallel trends
        
    Returns:
        StaggeredDiDSpec object with specification information
    """
    # Use first treatment column for now (may extend to multiple in future)
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols,
        excluded_cols=[time_col, unit_col]
    )

    # Infer treatment timing from the data
    treatment_times = data[data[treatment_col] == 1].groupby(unit_col)[time_col].min()
    data['treatment_time'] = data[unit_col].map(treatment_times)

    # Create relative time variable 
    data['relative_time'] = pd.NA
    mask = ~data['treatment_time'].isna()
    data.loc[mask, 'relative_time'] = (data.loc[mask, time_col] - data.loc[mask, 'treatment_time']).astype(int)
    data.loc[~mask, 'relative_time'] = 0  # Never treated units get 0 (separate from reference period)
    
    # Create event time dummies manually to avoid issues with negative signs in column names
    # Get unique relative time values (excluding the reference period and never-treated period)
    unique_times = sorted([t for t in data['relative_time'].dropna().unique() if t != reference_period])
    
    event_cols = []
    for t in unique_times:
        # Create formula-friendly column names
        if t < 0:
            col_name = f'event_pre{abs(t)}'  # event_pre1, event_pre2, etc.
        elif t >= 0:
            col_name = f'event_post{t}'      # event_post1, event_post2, etc.
        
        # Create the dummy variable
        data[col_name] = (data['relative_time'] == t).astype(int)
        event_cols.append(col_name)

    # Construct formula
    formula = f"{outcome_col} ~ {' + '.join(event_cols)} + EntityEffects + TimeEffects"

    if control_cols:
        formula += f" + {' + '.join(control_cols)}"
    
    print(formula)

    # Create and return specification
    return StaggeredDiDSpec(
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        time_col=time_col,
        unit_col=unit_col,
        treatment_time_col='treatment_time',
        control_cols=control_cols,
        data=data,
        formula=formula
    )


@make_transformable
def create_synthdid_specification(
    data: pd.DataFrame, 
    outcome_col: str = 'y', 
    treatment_cols: List[str] = ['treat'],
    time_col: str = 't',
    unit_col: str = 'id_unit',
    control_cols: Optional[List[str]] = None
) -> SynthDIDSpec:
    """
    Create a Synthetic Difference-in-Differences specification.
    
    Args:
        data: DataFrame with outcome, treatment, time, and unit identifiers
        outcome_col: Name of outcome column
        treatment_cols: List of treatment column names (only first is used)
        time_col: Name of time column
        unit_col: Name of unit identifier column
        control_cols: List of control variable columns
        
    Returns:
        SynthDIDSpec object with synthetic DiD specification information
    """
    
    
    # Use first treatment column for now
    treatment_col = treatment_cols[0]
    
    # Validate and prepare data
    data, control_cols = validate_and_prepare_data(
        data=data,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        required_columns=[time_col, unit_col],
        control_cols=control_cols,
        excluded_cols=[time_col, unit_col]
    )
    
    # Check that we have exactly one treated unit
    treated_units = data[data[treatment_col] == 1][unit_col].unique()
    if len(treated_units) == 0:
        raise ValueError("No treated units found in data")
    if len(treated_units) > 1:
        raise ValueError("Synthetic DiD requires exactly one treated unit")
    
    # Convert to panel format expected by synthdid
    try:
        panel_result = panel_matrices(
            data, 
            unit=unit_col, 
            time=time_col, 
            outcome=outcome_col, 
            treatment=treatment_col
        )
        
        Y = panel_result['Y']
        N0 = panel_result['N0']
        T0 = panel_result['T0']
        
        # Handle covariates if provided
        X = None
        if control_cols:
            # Create 3D covariate matrix
            n_units, n_times = Y.shape
            X = np.zeros((n_units, n_times, len(control_cols)))
            
            for k, cov_col in enumerate(control_cols):
                # Reshape covariate data to matrix form
                cov_matrix = data.pivot_table(
                    index=unit_col, 
                    columns=time_col, 
                    values=cov_col, 
                    fill_value=0
                ).values
                X[:, :, k] = cov_matrix
        
    except Exception as e:
        raise ValueError(f"Error converting data to panel format: {str(e)}")
    
    # Create a simple formula for compatibility
    formula = f"{outcome_col} ~ {treatment_col}"
    if control_cols:
        formula += " + " + " + ".join(control_cols)
    
    return SynthDIDSpec(
        data=data,
        formula=formula,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        control_cols=control_cols,
        time_col=time_col,
        unit_col=unit_col,
        Y=Y,
        N0=N0,
        T0=T0,
        X=X
    )


def spec_constructor(spec: Any) -> str:
    """Convert any specification object to a constructor string representation.
    
    Args:
        spec: Any specification object (DiDSpec, CrossSectionalSpec, etc.)
        
    Returns:
        A string representation calling the constructor with appropriate arguments
    """
    # Get the class name for the constructor call
    class_name = spec.__class__.__name__
    class_module = spec.__class__.__module__
    
    # Gather attributes excluding DataFrame objects
    attrs = []
    for key, value in spec.__dict__.items():
        if isinstance(value, pd.DataFrame):
            # Just reference the data variable
            attrs.append(f"    {key}=data_PLACEHOLDER")
        elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            # Format list of strings more cleanly
            items = ", ".join([repr(x) for x in value])
            attrs.append(f"    {key}=[{items}]")
        elif isinstance(value, str):
            # Add quotes around string values
            attrs.append(f'    {key}="{value}"')
        elif value is None:
            # Handle None values
            attrs.append(f"    {key}=None")
        else:
            # For other types (booleans, numbers, etc.)
            attrs.append(f"    {key}={value}")

    # add import statement for the class
    

    # Return the constructor call as a string with proper indentation
    return f"{class_name}(\n" + ",\n".join(attrs) + "\n)"









