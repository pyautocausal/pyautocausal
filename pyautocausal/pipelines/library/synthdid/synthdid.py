"""
Main estimators for the synthdid package
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass

from .utils import collapsed_form, contract3, pairwise_sum_decreasing
from .solver import sc_weight_fw, sc_weight_fw_covariates, sparsify_function


@dataclass
class SynthDIDEstimate:
    """
    Class to store the synthetic difference-in-differences estimate.
    """
    estimate: float
    weights: Dict[str, Any]
    setup: Dict[str, Any]
    opts: Dict[str, Any]
    estimator: str = "synthdid_estimate"
    
    def __float__(self):
        return float(self.estimate)
    
    def __repr__(self):
        return f"{self.estimator}: {float(self.estimate):.4f}"


def synthdid_estimate(Y: np.ndarray, 
                     N0: int, 
                     T0: int, 
                     X: Optional[np.ndarray] = None,
                     noise_level: Optional[float] = None,
                     eta_omega: float = None,
                     eta_lambda: float = 1e-6,
                     zeta_omega: Optional[float] = None,
                     zeta_lambda: Optional[float] = None,
                     omega_intercept: bool = True,
                     lambda_intercept: bool = True,
                     weights: Optional[Dict[str, np.ndarray]] = None,
                     update_omega: Optional[bool] = None,
                     update_lambda: Optional[bool] = None,
                     min_decrease: Optional[float] = None,
                     max_iter: int = 10000,
                     sparsify: Optional[Callable] = sparsify_function,
                     max_iter_pre_sparsify: int = 100) -> SynthDIDEstimate:
    """
    Computes the synthetic diff-in-diff estimate for an average treatment effect on a treated block.
    
    Parameters:
    -----------
    Y : np.ndarray
        The observation matrix of shape (N, T)
    N0 : int
        The number of control units
    T0 : int
        The number of pre-treatment time steps
    X : Optional[np.ndarray], default=None
        An optional 3-D array of time-varying covariates. Shape should be (N, T, C) for C covariates.
    noise_level : Optional[float], default=None
        An estimate of the noise standard deviation. Defaults to the standard deviation of first differences of Y.
    eta_omega : float, default=None
        Determines the tuning parameter zeta.omega = eta.omega * noise.level. Defaults to (N_tr * T_post)^(1/4).
    eta_lambda : float, default=1e-6
        Analogous for lambda. Defaults to an 'infinitesimal' value 1e-6.
    zeta_omega : Optional[float], default=None
        If passed, overrides the default zeta.omega = eta.omega * noise.level.
    zeta_lambda : Optional[float], default=None
        Analogous for lambda.
    omega_intercept : bool, default=True
        Binary. Use an intercept when estimating omega.
    lambda_intercept : bool, default=True
        Binary. Use an intercept when estimating lambda.
    weights : Optional[Dict[str, np.ndarray]], default=None
        A dictionary with fields lambda and omega. If non-null weights['lambda'] is passed,
        we use them instead of estimating lambda weights. Same for weights['omega'].
    update_omega : Optional[bool], default=None
        If True, solve for omega using the passed value of weights['omega'] only as an initialization.
        If False, use it exactly as passed. Defaults to False if a non-null value of weights['omega'] is passed.
    update_lambda : Optional[bool], default=None
        Analogous for lambda.
    min_decrease : Optional[float], default=None
        Tunes a stopping criterion for the weight estimator. Stop after an iteration results in a decrease
        in penalized MSE smaller than min_decrease^2.
    max_iter : int, default=10000
        A fallback stopping criterion for the weight estimator. Stop after this number of iterations.
    sparsify : Optional[Callable], default=sparsify_function
        A function mapping a numeric vector to a (presumably sparser) numeric vector, which must sum to one.
        If not None, try to estimate sparse weights via a second round of Frank-Wolfe optimization.
    max_iter_pre_sparsify : int, default=100
        Analogous to max_iter, but for the pre-sparsification first-round of optimization.
    
    Returns:
    --------
    SynthDIDEstimate
        An object containing the average treatment effect estimate and other information.
    """
    # Check inputs
    if Y.shape[0] <= N0 or Y.shape[1] <= T0:
        raise ValueError("Y must have more rows than N0 and more columns than T0")
    
    # Set up X if not provided
    if X is None:
        X = np.zeros((Y.shape[0], Y.shape[1], 0))
    
    # Check X dimensions
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("X must have the same first two dimensions as Y")
    
    # Calculate dimensions
    N1 = Y.shape[0] - N0
    T1 = Y.shape[1] - T0
    
    # Set default noise level
    if noise_level is None:
        # Calculate standard deviation of first differences
        diffs = np.diff(Y[:N0, :T0], axis=1)
        if diffs.size == 0:  # Handle case with only one pre-treatment period
            noise_level = 0.0
        else:
            noise_level = np.std(diffs, ddof=1) # Match R's sd() which uses ddof=1
    
    # Set default eta_omega
    if eta_omega is None:
        eta_omega = (N1 * T1) ** (1/4) if (N1 * T1) > 0 else 1.0 # Ensure positive base for power
    
    # Set default zeta parameters
    if zeta_omega is None:
        zeta_omega = eta_omega * noise_level
    
    if zeta_lambda is None:
        zeta_lambda = eta_lambda * noise_level
    
    # Set default min_decrease
    if min_decrease is None:
        min_decrease = 1e-5 * noise_level
    
    # Initialize weights if not provided
    if weights is None:
        weights = {'lambda': None, 'omega': None, 'beta': None}
    
    # Set default update flags
    if update_lambda is None:
        update_lambda = weights.get('lambda') is None
    
    if update_omega is None:
        update_omega = weights.get('omega') is None
    
    # Check that we have weights or will estimate them
    if not update_lambda and weights.get('lambda') is None:
        raise ValueError("Either update_lambda must be True or weights['lambda'] must be provided")
    
    if not update_omega and weights.get('omega') is None:
        raise ValueError("Either update_omega must be True or weights['omega'] must be provided")
    
    # Set max_iter_pre_sparsify to max_iter if no sparsify function
    if sparsify is None:
        max_iter_pre_sparsify = max_iter
    
    # Estimate weights
    if X.shape[2] == 0:  # No covariates
        weights['vals'] = None
        weights['lambda_vals'] = None
        weights['omega_vals'] = None
        
        if update_lambda:
            Yc = collapsed_form(Y, N0, T0)
            lambda_opt = sc_weight_fw(
                Yc[:N0, :], 
                zeta=zeta_lambda, 
                intercept=lambda_intercept, 
                lambda_init=weights.get('lambda'),
                min_decrease=min_decrease, 
                max_iter=max_iter_pre_sparsify
            )
            
            if sparsify is not None:
                lambda_opt = sc_weight_fw(
                    Yc[:N0, :], 
                    zeta=zeta_lambda, 
                    intercept=lambda_intercept, 
                    lambda_init=sparsify(lambda_opt['lambda']),
                    min_decrease=min_decrease, 
                    max_iter=max_iter
                )
            
            weights['lambda'] = lambda_opt['lambda']
            weights['lambda_vals'] = lambda_opt['vals']
            weights['vals'] = lambda_opt['vals']
        
        if update_omega:
            Yc = collapsed_form(Y, N0, T0)
            omega_opt = sc_weight_fw(
                Yc[:, :T0].T, 
                zeta=zeta_omega, 
                intercept=omega_intercept, 
                lambda_init=weights.get('omega'),
                min_decrease=min_decrease, 
                max_iter=max_iter_pre_sparsify
            )
            
            if sparsify is not None:
                omega_opt = sc_weight_fw(
                    Yc[:, :T0].T, 
                    zeta=zeta_omega, 
                    intercept=omega_intercept, 
                    lambda_init=sparsify(omega_opt['lambda']),
                    min_decrease=min_decrease, 
                    max_iter=max_iter
                )
            
            weights['omega'] = omega_opt['lambda']
            weights['omega_vals'] = omega_opt['vals']
            
            if weights['vals'] is None:
                weights['vals'] = omega_opt['vals']
            else:
                weights['vals'] = pairwise_sum_decreasing(weights['vals'], omega_opt['vals'])
    else:
        # With covariates
        Yc = collapsed_form(Y, N0, T0)
        Xc = np.zeros((Yc.shape[0], Yc.shape[1], X.shape[2]))
        
        for i in range(X.shape[2]):
            Xc[:, :, i] = collapsed_form(X[:, :, i], N0, T0)
        
        opt_weights = sc_weight_fw_covariates(
            Yc, Xc,
            zeta_lambda=zeta_lambda,
            zeta_omega=zeta_omega,
            lambda_intercept=lambda_intercept,
            omega_intercept=omega_intercept,
            min_decrease=min_decrease,
            max_iter=max_iter,
            lambda_init=weights.get('lambda'),
            omega_init=weights.get('omega'),
            beta_init=weights.get('beta'),
            update_lambda=update_lambda,
            update_omega=update_omega
        )
        
        weights = {
            'lambda': opt_weights['lambda'],
            'omega': opt_weights['omega'],
            'beta': opt_weights['beta'],
            'vals': opt_weights['vals']
        }
    
    # Apply sparsify post-optimization if specified
    if sparsify is not None and X.shape[2] == 0: # only sparsify if no covariates, to match R
        if weights.get('lambda') is not None:
            weights['lambda'] = sparsify(weights['lambda'].astype(np.float64))
            # The sparsify function should handle normalization and NaNs.
            # If result is all NaNs, or original was empty, it should be fine.

        if weights.get('omega') is not None:
            weights['omega'] = sparsify(weights['omega'].astype(np.float64))
            # Similar to lambda, sparsify handles this.

    # Compute X.beta - use float64 for intermediate calculations to match R precision
    X_beta = contract3(X, weights.get('beta', np.array([])))
    
    # Compute estimate - using float64 for intermediate vectors to match R precision
    omega_vec = np.concatenate([-weights['omega'], np.ones(N1) / N1]).astype(np.float64)
    lambda_vec = np.concatenate([-weights['lambda'], np.ones(T1) / T1]).astype(np.float64)
    
    # Matrix multiplication with explicit promotion to float64
    Y_float64 = Y.astype(np.float64)
    X_beta_float64 = X_beta.astype(np.float64)
    estimate = omega_vec @ (Y_float64 - X_beta_float64) @ lambda_vec
    
    # Create result object
    opts = {
        'zeta_omega': zeta_omega,
        'zeta_lambda': zeta_lambda,
        'omega_intercept': omega_intercept,
        'lambda_intercept': lambda_intercept,
        'update_omega': update_omega,
        'update_lambda': update_lambda,
        'min_decrease': min_decrease,
        'max_iter': max_iter
    }
    
    setup = {
        'Y': Y,
        'X': X,
        'N0': N0,
        'T0': T0
    }
    
    return SynthDIDEstimate(
        estimate=estimate,
        weights=weights,
        setup=setup,
        opts=opts,
        estimator="synthdid_estimate"
    )


def sc_estimate(Y: np.ndarray, 
               N0: int, 
               T0: int, 
               eta_omega: float = 1e-6, 
               # omega_intercept is explicitly a parameter for sc_estimate
               # lambda_intercept can be passed via kwargs
               **kwargs) -> SynthDIDEstimate:
    """
    Synthetic control estimate.
    Takes all the same parameters as synthdid_estimate, but uses the synthetic control estimator.
    """
    # Ensure float64 precision for data
    Y = Y.astype(np.float64, copy=False)
    
    # For SC, lambda is fixed to zeros
    sc_fixed_lambda = np.zeros(T0, dtype=np.float64)
    
    # Get any omega weights from kwargs or weights
    sc_weights = {'lambda': sc_fixed_lambda}
    
    # Handle omega weights from provided weights parameter
    weights_param = kwargs.get('weights', {})
    if isinstance(weights_param, dict) and 'omega' in weights_param:
        sc_weights['omega'] = weights_param['omega']
        # Set update_omega to False since we're using provided weights
        kwargs['update_omega'] = False
    
    # Remove weights from kwargs to avoid confusion in synthdid_estimate
    kwargs.pop('weights', None)
    
    # Explicitly set omega_intercept for SC
    current_omega_intercept = kwargs.pop('omega_intercept', False)  # Default False for SC
    
    # Remove other synthdid_estimate specific args if set by mistake
    kwargs.pop('update_lambda', None)
    
    # Add special handling for R-compatible estimates
    if 'zeta_omega' not in kwargs and 'noise_level' in kwargs:
        zeta_omega = eta_omega * kwargs['noise_level']
        kwargs['zeta_omega'] = zeta_omega
    
    # Handle update_omega - if not explicitly set and no omega weights provided,
    # then we need to estimate them by setting update_omega=True
    if 'update_omega' not in kwargs and 'omega' not in sc_weights:
        kwargs['update_omega'] = True

    estimate = synthdid_estimate(
        Y, N0, T0, eta_omega=eta_omega,
        weights=sc_weights, 
        omega_intercept=current_omega_intercept, 
        # lambda_intercept will be taken from kwargs if present, otherwise use default
        update_lambda=False,  # Lambda is always fixed for SC
        **kwargs
    )
    
    estimate.estimator = "sc_estimate"
    return estimate


def did_estimate(Y: np.ndarray, 
                N0: int, 
                T0: int, 
                # omega_intercept and lambda_intercept can be passed via kwargs
                **kwargs) -> SynthDIDEstimate:
    """
    Difference-in-differences estimate.
    Takes all the same parameters as synthdid_estimate, but uses the DiD estimator.
    """
    did_weights = {
        'lambda': np.ones(T0) / T0 if T0 > 0 else np.array([]),
        'omega': np.ones(N0) / N0 if N0 > 0 else np.array([])
    }
    
    # Remove other synthdid_estimate specific args from kwargs
    kwargs.pop('weights', None)
    kwargs.pop('update_lambda', None)
    kwargs.pop('update_omega', None)

    # omega_intercept and lambda_intercept will be taken from kwargs if present, 
    # otherwise synthdid_estimate defaults (True, True)
    estimate = synthdid_estimate(
        Y, N0, T0, weights=did_weights, 
        update_lambda=False,  
        update_omega=False,   
        **kwargs
    )
    
    estimate.estimator = "did_estimate"
    return estimate


def synthdid_effect_curve(estimate: SynthDIDEstimate) -> np.ndarray:
    """
    Outputs the effect curve that was averaged to produce the estimate.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate object returned by synthdid_estimate
    
    Returns:
    --------
    np.ndarray
        The effect curve
    """
    setup = estimate.setup
    weights = estimate.weights
    X_beta = contract3(setup['X'], weights.get('beta', np.array([])))
    
    N1 = setup['Y'].shape[0] - setup['N0']
    T1 = setup['Y'].shape[1] - setup['T0']
    
    # Vector of weights
    omega_vec = np.concatenate([-weights['omega'], np.ones(N1) / N1])
    
    # Calculate tau_sc (synthetic control trajectory)
    tau_sc = omega_vec @ (setup['Y'] - X_beta)
    
    # Calculate effect curve
    pre_trend = tau_sc[:setup['T0']] @ weights['lambda']
    tau_curve = tau_sc[setup['T0']:] - pre_trend
    
    return tau_curve


def synthdid_placebo(estimate: SynthDIDEstimate, treated_fraction: Optional[float] = None) -> SynthDIDEstimate:
    """
    Computes a placebo variant of the estimator using pre-treatment data only.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate object returned by synthdid_estimate
    treated_fraction : Optional[float], default=None
        The fraction of pre-treatment data to use as a placebo treatment period.
        Defaults to None, which indicates that it should be the fraction of 
        post-treatment to pre-treatment data.
    
    Returns:
    --------
    SynthDIDEstimate
        The placebo estimate
    """
    setup = estimate.setup
    opts = estimate.opts
    weights = estimate.weights
    estimator = estimate.estimator
    
    # Determine the treated fraction
    if treated_fraction is None:
        treated_fraction = 1 - setup['T0'] / setup['Y'].shape[1]
    
    # Calculate the new T0 for the placebo test
    placebo_T0 = int(setup['T0'] * (1 - treated_fraction))
    
    # Create a subset of the data for the placebo test
    Y_placebo = setup['Y'][:, :setup['T0']]
    X_placebo = setup['X'][:, :setup['T0'], :] if setup['X'].size > 0 else None
    
    # Call the appropriate estimator
    if estimator == "synthdid_estimate":
        return synthdid_estimate(Y_placebo, setup['N0'], placebo_T0, X_placebo, **opts)
    elif estimator == "sc_estimate":
        return sc_estimate(Y_placebo, setup['N0'], placebo_T0, X=X_placebo, **opts)
    elif estimator == "did_estimate":
        return did_estimate(Y_placebo, setup['N0'], placebo_T0, X=X_placebo, **opts)
    else:
        raise ValueError(f"Unknown estimator type: {estimator}") 