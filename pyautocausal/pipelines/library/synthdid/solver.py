"""
Solver functions for the synthdid package
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from .utils import contract3


def fw_step(A: np.ndarray, x: np.ndarray, b: np.ndarray, eta: float, alpha: Optional[float] = None) -> np.ndarray:
    """
    A Frank-Wolfe step for ||Ax - b||^2 + eta * ||x||^2 with x in unit simplex.
    
    Parameters:
    -----------
    A : np.ndarray
        Matrix of shape (M, N)
    x : np.ndarray
        Vector of shape (N,)
    b : np.ndarray
        Vector of shape (M,)
    eta : float
        Regularization parameter
    alpha : Optional[float], default=None
        Step size. If None, use line search
    
    Returns:
    --------
    np.ndarray
        Updated weights vector
    """
    # Ensure high precision for numerical stability
    A_f64 = A.astype(np.float64, copy=False)
    x_f64 = x.astype(np.float64, copy=False)
    b_f64 = b.astype(np.float64, copy=False)
    
    Ax = A_f64 @ x_f64
    half_grad = (Ax - b_f64).T @ A_f64 + eta * x_f64
    
    # Handle case where half_grad might be all NaNs or contains NaNs
    if np.all(np.isnan(half_grad)) or np.isnan(half_grad).any():
        # If all elements are NaN, we cannot determine a direction. Return current x.
        # If some elements are NaN, replace them with a large positive value so they won't be selected
        half_grad = np.where(np.isnan(half_grad), np.inf, half_grad)
        # If still all infinite/NaN after replacement, return x
        if np.all(np.isinf(half_grad)) or np.all(np.isnan(half_grad)):
            return x_f64
    
    i = np.argmin(half_grad)
    
    if alpha is not None:
        x_new = x_f64 * (1 - alpha)
        x_new[i] += alpha
        return x_new
    else:
        d_x = -x_f64.copy()
        d_x[i] = 1 - x_f64[i]
        
        if np.all(d_x == 0):
            return x_f64
        
        d_err = A_f64[:, i] - Ax
        
        # Use high precision for the division
        num = -np.sum(half_grad * d_x, dtype=np.float64)
        den = np.sum(d_err**2, dtype=np.float64) + eta * np.sum(d_x**2, dtype=np.float64)
        
        # Handle potential numerical instability
        if den == 0 or np.isnan(den) or np.isnan(num):
            step = 0.0
        else:
            step = num / den
            
        # Ensure step is within [0, 1]
        constrained_step = min(1, max(0, step))
        
        return x_f64 + constrained_step * d_x


def sc_weight_fw(Y: np.ndarray, zeta: float, intercept: bool = True, 
                lambda_init: Optional[np.ndarray] = None, min_decrease: float = 1e-3, 
                max_iter: int = 1000) -> Dict[str, Any]:
    """
    A Frank-Wolfe solver for synthetic control weights using exact line search.
    
    Parameters:
    -----------
    Y : np.ndarray
        Matrix of shape (N0, T0+1)
    zeta : float
        Regularization parameter
    intercept : bool, default=True
        Whether to center the data
    lambda_init : Optional[np.ndarray], default=None
        Initial value for lambda weights
    min_decrease : float, default=1e-3
        Stopping criterion threshold
    max_iter : int, default=1000
        Maximum number of iterations
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with 'lambda' (weights) and 'vals' (optimization values)
    """
    T0 = Y.shape[1] - 1
    N0 = Y.shape[0]
    
    # Convert to float64 for highest precision matching R
    Y = Y.astype(np.float64, copy=False)
    
    if lambda_init is None:
        lambda_weights = np.ones(T0, dtype=np.float64) / T0
    else:
        lambda_weights = lambda_init.astype(np.float64, copy=True)
    
    if intercept:
        # Center each column using float64 for means calculation to match R's behavior
        Y_means = Y.mean(axis=0, dtype=np.float64)
        Y = Y - Y_means
    
    # Set up for the Frank-Wolfe algorithm
    vals = np.full(max_iter, np.nan, dtype=np.float64)
    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N0 * zeta**2
    
    # Corrected loop logic to match R
    # In R: t goes from 1 to max.iter.
    # vals[t] is the objective value using lambda from iteration t.
    # Condition: (t < 2 || vals[t-1] - vals[t] > min_decrease^2)
    # loop continues if t=1 OR (objective value from iter t-1 MINUS objective value from iter t > threshold)
    
    final_t_iterations_done = 0
    for t_iter_num in range(1, max_iter + 1):  # t_iter_num is R's t (1-indexed)
        final_t_iterations_done = t_iter_num
        
        # Use fw_step to update lambda weights
        lambda_p = fw_step(A, lambda_weights, b, eta)  # lambda_weights is from prev iter or init
        lambda_weights = lambda_p  # lambda_weights is now for current iter t_iter_num
        
        # Calculate error and objective value components
        with np.errstate(invalid='ignore'):
            err = Y @ np.concatenate([lambda_weights, [-1.0]])  # Y is N0x(T0+1)
            current_mse_term = np.sum(err**2, dtype=np.float64) / N0
            current_penalty_term = zeta**2 * np.sum(lambda_weights**2, dtype=np.float64)
            vals[t_iter_num - 1] = current_penalty_term + current_mse_term  # Store in 0-indexed vals
        
        if t_iter_num >= 2:  # Can only check decrease if we have at least two values
            # Previous value was vals[t_iter_num - 2], Current value is vals[t_iter_num - 1]
            if (vals[t_iter_num - 2] - vals[t_iter_num - 1]) <= min_decrease**2:
                break  # Break if decrease is not sufficient
        # If t_iter_num is 1, R's (t<2) is true, so loop must continue without checking decrease.
    
    return {
        'lambda': lambda_weights,
        'vals': vals[:final_t_iterations_done]  # Slice up to number of iterations actually done
    }


def sc_weight_fw_covariates(Y: np.ndarray, X: Optional[np.ndarray] = None, 
                           zeta_lambda: float = 0.0, zeta_omega: float = 0.0,
                           lambda_intercept: bool = True, omega_intercept: bool = True,
                           min_decrease: float = 1e-3, max_iter: int = 1000,
                           lambda_init: Optional[np.ndarray] = None, 
                           omega_init: Optional[np.ndarray] = None,
                           beta_init: Optional[np.ndarray] = None,
                           update_lambda: bool = True, update_omega: bool = True) -> Dict[str, Any]:
    """
    A Frank-Wolfe + Gradient solver for lambda, omega, and beta when there are covariates.
    
    Parameters:
    -----------
    Y : np.ndarray
        Matrix of shape (N0+1, T0+1)
    X : np.ndarray, default=None
        3D array of covariates of shape (N0+1, T0+1, C)
    zeta_lambda : float, default=0.0
        Regularization parameter for lambda
    zeta_omega : float, default=0.0
        Regularization parameter for omega
    lambda_intercept : bool, default=True
        Whether to center the data for lambda estimation
    omega_intercept : bool, default=True
        Whether to center the data for omega estimation
    min_decrease : float, default=1e-3
        Stopping criterion threshold
    max_iter : int, default=1000
        Maximum number of iterations
    lambda_init : Optional[np.ndarray], default=None
        Initial value for lambda weights
    omega_init : Optional[np.ndarray], default=None
        Initial value for omega weights
    beta_init : Optional[np.ndarray], default=None
        Initial value for beta coefficients
    update_lambda : bool, default=True
        Whether to update lambda weights
    update_omega : bool, default=True
        Whether to update omega weights
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with 'lambda', 'omega', 'beta', and 'vals'
    """
    T0 = Y.shape[1] - 1
    N0 = Y.shape[0] - 1
    
    Y = Y.astype(np.float64, copy=False)
    
    # Create empty X if not provided
    if X is None:
        X = np.zeros((Y.shape[0], Y.shape[1], 0))
    
    # Check dimensions
    assert Y.ndim == 2, "Y must be a 2D array"
    assert X.ndim == 3, "X must be a 3D array"
    assert Y.shape[0] == X.shape[0] and Y.shape[1] == X.shape[1], "Y and X must have the same first two dimensions"
    
    # Initialize weights
    if lambda_init is None:
        lambda_weights = np.ones(T0) / T0
    else:
        lambda_weights = lambda_init.copy()
    
    if omega_init is None:
        omega_weights = np.ones(N0) / N0
    else:
        omega_weights = omega_init.copy()
    
    if beta_init is None:
        beta = np.zeros(X.shape[2])
    else:
        beta = beta_init.copy()
    
    # Define a function to update weights
    def update_weights(Y, lambda_weights, omega_weights):
        # Update lambda weights
        Y_lambda = Y[:N0, :].astype(np.float64, copy=False)
        if lambda_intercept:
            # Calculate means using float64 explicitly
            Y_lambda_means = Y_lambda.mean(axis=0, dtype=np.float64)
            Y_lambda = Y_lambda - Y_lambda_means
            
        if update_lambda:
            lambda_weights = fw_step(Y_lambda[:, :T0], lambda_weights, Y_lambda[:, T0], N0 * zeta_lambda**2)
        
        err_lambda = Y_lambda @ np.concatenate([lambda_weights, [-1.0]], dtype=np.float64)
        
        # Update omega weights (using transposed data)
        Y_omega = Y[:, :T0].T.astype(np.float64, copy=False)
        if omega_intercept:
            # Calculate means using float64 explicitly
            Y_omega_means = Y_omega.mean(axis=0, dtype=np.float64)
            Y_omega = Y_omega - Y_omega_means
            
        if update_omega:
            omega_weights = fw_step(Y_omega[:, :N0], omega_weights, Y_omega[:, N0], T0 * zeta_omega**2)
        
        err_omega = Y_omega @ np.concatenate([omega_weights, [-1.0]], dtype=np.float64)
        
        # Calculate loss value with explicit float64
        val = (
            zeta_omega**2 * np.sum(omega_weights**2, dtype=np.float64) + 
            zeta_lambda**2 * np.sum(lambda_weights**2, dtype=np.float64) + 
            np.sum(err_omega**2, dtype=np.float64) / T0 + 
            np.sum(err_lambda**2, dtype=np.float64) / N0
        )
        
        return {
            'val': val,
            'lambda': lambda_weights,
            'omega': omega_weights,
            'err_lambda': err_lambda,
            'err_omega': err_omega
        }
    
    # Main optimization loop
    vals = np.full(max_iter, np.nan)
    t = 0
    Y_beta = Y - contract3(X, beta)
    weights = update_weights(Y_beta, lambda_weights, omega_weights)
    
    while t < max_iter and (t < 2 or abs(vals[t-1] - vals[t]) > min_decrease**2):
        t += 1
        
        # Update beta with gradient step
        if X.shape[2] > 0:
            grad_beta = np.zeros(X.shape[2])
            for i in range(X.shape[2]):
                Xi = X[:, :, i]
                grad_beta[i] = (
                    weights['err_lambda'].T @ Xi[:N0, :] @ np.concatenate([weights['lambda'], [-1]]) / N0 +
                    weights['err_omega'].T @ Xi[:, :T0].T @ np.concatenate([weights['omega'], [-1]]) / T0
                )
            
            alpha = 1 / t
            beta = beta - alpha * grad_beta
            Y_beta = Y - contract3(X, beta)
        
        # Update weights
        weights = update_weights(Y_beta, weights['lambda'], weights['omega'])
        vals[t-1] = weights['val']
    
    return {
        'lambda': weights['lambda'],
        'omega': weights['omega'],
        'beta': beta,
        'vals': vals[:t]
    }


def sparsify_function(v: np.ndarray) -> np.ndarray:
    """
    Maps a numeric vector to a (presumably sparser) numeric vector of the same shape.
    Matches R's behavior: v[v <= max(v)/4] = 0; v/sum(v)
    
    Parameters:
    -----------
    v : np.ndarray
        Input vector
    
    Returns:
    --------
    np.ndarray
        Sparsified vector, potentially containing NaNs if sum is zero.
    """
    # Handle empty inputs
    if v.size == 0:
        return v

    # Ensure float64 precision to match R
    v_copy = v.astype(np.float64, copy=True)
    
    # Handle all-zero input case - R returns all zeros
    if np.all(v_copy == 0) or np.all(np.isnan(v_copy)):
        return v_copy
    
    # Find max value, ignoring NaNs
    max_v = np.nanmax(v_copy)
    
    # Set values <= max/4 to zero
    threshold = max_v / 4.0
    v_copy[v_copy <= threshold] = 0.0
    
    # Calculate sum for normalization
    sum_v = np.sum(v_copy)
    
    # Handle zero sum case like R does - will produce NaNs
    # Let division by zero produce np.nan naturally, similar to R
    with np.errstate(divide='ignore', invalid='ignore'):
        result = v_copy / sum_v
        
    return result 