"""
Utility functions for the synthdid package
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any


def collapsed_form(Y: np.ndarray, N0: int, T0: int) -> np.ndarray:
    """
    Collapse Y to an (N0+1) x (T0+1) matrix by averaging the last N1=N-N0 rows 
    and T1=T-T0 columns.
    
    Parameters:
    -----------
    Y : np.ndarray
        The observation matrix of shape (N, T)
    N0 : int
        The number of control units 
    T0 : int
        The number of pre-treatment time steps
    
    Returns:
    --------
    np.ndarray
        The collapsed matrix of shape (N0+1, T0+1)
    """
    N, T = Y.shape
    
    # Extract and average the blocks
    control_pre = Y[:N0, :T0]  
    control_post = Y[:N0, T0:].mean(axis=1, keepdims=True, dtype=np.float64)
    treated_pre = Y[N0:, :T0].mean(axis=0, keepdims=True, dtype=np.float64)
    treated_post = Y[N0:, T0:].mean(dtype=np.float64)
    
    # Combine the blocks
    top = np.hstack([control_pre, control_post])
    bottom = np.hstack([treated_pre, np.array([[treated_post]])])
    
    return np.vstack([top, bottom])


def contract3(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Contract a 3D tensor X with a vector v along the third dimension.
    
    Parameters:
    -----------
    X : np.ndarray
        A 3D tensor of shape (N, T, C)
    v : np.ndarray
        A vector of shape (C,)
    
    Returns:
    --------
    np.ndarray
        The contracted tensor of shape (N, T)
    """
    if X.size == 0 or v.size == 0:
        return np.zeros(X.shape[:2])
    
    return np.tensordot(X, v, axes=([2], [0]))


def pairwise_sum_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return the component-wise sum of decreasing vectors in which NA (np.nan) is 
    taken to mean that the vector has stopped decreasing and we can use the last 
    non-NA element. Where both are NA, leave as NA.
    
    Parameters:
    -----------
    x : np.ndarray
        First vector
    y : np.ndarray
        Second vector
    
    Returns:
    --------
    np.ndarray
        The pairwise sum with length of max(len(x), len(y))
    """
    # For different length arrays, use the longer one's length
    max_len = max(len(x), len(y))
    
    # Create padded copies of both arrays
    x_padded = np.full(max_len, np.nan, dtype=np.float64)
    y_padded = np.full(max_len, np.nan, dtype=np.float64)
    
    # Copy the original data
    x_padded[:len(x)] = x
    y_padded[:len(y)] = y
    
    # Handle NaN values by replacing with minimum
    na_x = np.isnan(x_padded)
    na_y = np.isnan(y_padded)
    
    if np.any(~na_x):  # If there are any non-NaN values
        last_val_x = np.min(x_padded[~na_x])
        x_padded[na_x] = last_val_x
    
    if np.any(~na_y):  # If there are any non-NaN values
        last_val_y = np.min(y_padded[~na_y])
        y_padded[na_y] = last_val_y
    
    # Compute pairwise sum
    result = x_padded + y_padded
    
    # Restore NaNs where both inputs had NaNs
    result[na_x & na_y] = np.nan
    
    return result


def panel_matrices(panel: pd.DataFrame, 
                  unit: Union[int, str] = 0,
                  time: Union[int, str] = 1,
                  outcome: Union[int, str] = 2,
                  treatment: Union[int, str] = 3,
                  treated_last: bool = True) -> Dict[str, Any]:
    """
    Convert a long (balanced) panel to a wide matrix format required by synthdid estimators.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        A data frame with columns for units, time, outcome, and treatment indicator
    unit : Union[int, str], default=0
        The column number/name corresponding to the unit identifier
    time : Union[int, str], default=1
        The column number/name corresponding to the time identifier
    outcome : Union[int, str], default=2
        The column number/name corresponding to the outcome identifier
    treatment : Union[int, str], default=3
        The column number/name corresponding to the treatment status
    treated_last : bool, default=True
        Should we sort the rows of Y and W so treated units are last
    
    Returns:
    --------
    Dict[str, Any]
        A dictionary with entries 'Y': the data matrix, 'N0': the number of control units,
        'T0': the number of time periods before treatment, 'W': the matrix of treatment indicators
    """
    # Handle column selection
    def index_to_name(x, df):
        if isinstance(x, int) and 0 <= x < len(df.columns):
            return df.columns[x]
        return x
    
    unit_col = index_to_name(unit, panel)
    time_col = index_to_name(time, panel)
    outcome_col = index_to_name(outcome, panel)
    treatment_col = index_to_name(treatment, panel)
    
    keep = [unit_col, time_col, outcome_col, treatment_col]
    
    if not all(col in panel.columns for col in keep):
        raise ValueError("Column identifiers should be either integer or column names in `panel`.")
    
    panel_subset = panel[keep].copy()
    
    # Check for NAs
    if panel_subset.isna().any().any():
        raise ValueError("Missing values in `panel`.")
    
    # Check treatment status
    if len(panel_subset[treatment_col].unique()) == 1:
        raise ValueError("There is no variation in treatment status.")
    
    if not all(val in [0, 1] for val in panel_subset[treatment_col].unique()):
        raise ValueError("The treatment status should be in 0 or 1.")
    
    # A balanced panel requires every cell once, not merely no duplicates.
    if panel_subset.duplicated([unit_col, time_col]).any():
        raise ValueError("Input panel must have exactly one observation per unit and time")
    unique_units = np.sort(panel_subset[unit_col].unique())
    unique_times = np.sort(panel_subset[time_col].unique())
    num_units, num_years = len(unique_units), len(unique_times)
    if len(panel_subset) != num_units * num_years:
        raise ValueError("Input panel must be balanced: an observation is required for every unit at every time")
    Y = panel_subset.pivot(index=unit_col, columns=time_col, values=outcome_col).reindex(
        index=unique_units, columns=unique_times).to_numpy(dtype=float)
    W = panel_subset.pivot(index=unit_col, columns=time_col, values=treatment_col).reindex(
        index=unique_units, columns=unique_times).to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError("Panel outcomes must be finite numeric values")
    w = np.any(W == 1, axis=1)
    N0 = int(np.sum(~w))
    if N0 == 0 or N0 == num_units:
        raise ValueError("Synthetic DiD requires both never-treated donors and treated units")
    # T0 is a COUNT: columns [:T0] are pre, columns [T0:] are post.
    T0 = int(np.flatnonzero(np.any(W == 1, axis=0))[0])
    if T0 == 0:
        raise ValueError("Synthetic DiD requires at least one pre-treatment period")
    if not (np.all(W[:, :T0] == 0) and np.all(W[w, T0:] == 1)):
        raise ValueError("Treatment adoption must be simultaneous and permanent")
    unit_order = np.argsort(w, kind='stable') if treated_last else np.arange(num_units)
    if not treated_last and (w[:N0].any() or not w[N0:].all()):
        raise ValueError("Controls must precede treated units when treated_last=False")
    
    # Create row and column names
    Y_with_names = pd.DataFrame(Y[unit_order, :], index=unique_units[unit_order], columns=unique_times)
    W_with_names = pd.DataFrame(W[unit_order, :], index=unique_units[unit_order], columns=unique_times)
    
    return {
        'Y': Y_with_names.values,
        'N0': N0,
        'T0': T0,
        'W': W_with_names.values,
        'Y_df': Y_with_names,
        'W_df': W_with_names
    }


def random_low_rank(n_0: int = 100, 
                   n_1: int = 10, 
                   T_0: int = 120, 
                   T_1: int = 20, 
                   tau: float = 1.0, 
                   sigma: float = 0.5, 
                   rank: int = 2, 
                   rho: float = 0.7) -> Dict[str, Any]:
    """
    Generate random low-rank data for testing.
    
    Parameters:
    -----------
    n_0 : int, default=100
        Number of control units
    n_1 : int, default=10
        Number of treated units
    T_0 : int, default=120
        Number of pre-treatment time periods
    T_1 : int, default=20
        Number of post-treatment time periods
    tau : float, default=1.0
        Treatment effect
    sigma : float, default=0.5
        Noise level
    rank : int, default=2
        Rank of the underlying process
    rho : float, default=0.7
        Autocorrelation parameter
    
    Returns:
    --------
    Dict[str, Any]
        A dictionary with entries 'Y': the data matrix, 'L': the low-rank component, 
        'N0': the number of control units, 'T0': the number of pre-treatment periods
    """
    n = n_0 + n_1
    T = T_0 + T_1
    
    # Create covariance matrix with autocorrelation rho
    indices = np.arange(T)
    var = np.exp(-rho * np.abs(indices[:, np.newaxis] - indices[np.newaxis, :]))
    
    # Treatment matrix
    W = np.zeros((n, T))
    W[n_0:, T_0:] = 1
    
    # Low-rank component
    U = np.random.poisson(np.sqrt(np.arange(1, n+1))[:, np.newaxis] / np.sqrt(n), size=(n, rank))
    V = np.random.poisson(np.sqrt(np.arange(1, T+1))[:, np.newaxis] / np.sqrt(T), size=(T, rank))
    
    # Fixed effects
    alpha = 10 * np.random.random(n)[:, np.newaxis] * np.ones((n, T))
    beta = 10 * np.arange(1, T+1)[np.newaxis, :] / T * np.ones((n, T))
    
    # Low-rank mean
    mu = U @ V.T + alpha + beta
    
    # Generate multivariate normal errors with autocorrelation
    L = np.linalg.cholesky(var)
    Z = np.random.normal(0, 1, size=(n, T))
    error = Z @ L.T * sigma
    
    # Complete data matrix
    Y = mu + tau * W + error
    
    # Add row and column names
    row_names = [str(i+1) for i in range(n)]
    col_names = [str(i+1) for i in range(T)]
    Y_df = pd.DataFrame(Y, index=row_names, columns=col_names)
    L_df = pd.DataFrame(mu, index=row_names, columns=col_names)
    
    return {
        'Y': Y, 
        'L': mu, 
        'N0': n_0, 
        'T0': T_0,
        'Y_df': Y_df,
        'L_df': L_df
    }
