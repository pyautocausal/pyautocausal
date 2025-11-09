"""
Variance estimation functions for the synthdid package
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import warnings

from .utils import contract3
from .synthdid import SynthDIDEstimate, synthdid_estimate, sc_estimate, did_estimate


def bootstrap_se(estimate: SynthDIDEstimate, replications: int = 200, 
                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute bootstrap standard errors for a synthdid estimate.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate for which to compute standard errors
    replications : int, default=200
        Number of bootstrap replications
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with standard error ('se'), confidence interval ('ci'), 
        and bootstrap distribution ('boot_vals')
    """
    setup = estimate.setup
    
    # Get original dimensions
    N0 = setup['N0']
    T0 = setup['T0']
    N = setup['Y'].shape[0]
    T = setup['Y'].shape[1]
    N1 = N - N0
    T1 = T - T0
    
    # Store bootstrap estimates
    boot_vals = np.zeros(replications)
    
    # Prepare estimator
    estimator_name = estimate.estimator
    estimator_fun = {
        "synthdid_estimate": synthdid_estimate,
        "sc_estimate": sc_estimate,
        "did_estimate": did_estimate
    }[estimator_name]
    
    # Run bootstrap
    for i in range(replications):
        # Sample units with replacement
        control_units = np.random.choice(N0, N0, replace=True)
        treated_units = np.random.choice(np.arange(N0, N), N1, replace=True)
        units = np.concatenate([control_units, treated_units])
        
        # Sample time periods with replacement
        pre_periods = np.random.choice(T0, T0, replace=True)
        post_periods = np.random.choice(np.arange(T0, T), T1, replace=True)
        periods = np.concatenate([pre_periods, post_periods])
        
        # Create bootstrap sample
        Y_boot = setup['Y'][units, :][:, periods]
        
        # Create covariates if they exist
        X_boot = None
        if setup['X'].size > 0:
            X_boot = setup['X'][units, :, :][:, periods, :]
        
        # Estimate with bootstrap sample
        boot_est = estimator_fun(Y_boot, N0, T0, X=X_boot, **estimate.opts)
        boot_vals[i] = float(boot_est.estimate)
    
    # Compute standard error
    se = np.std(boot_vals)
    
    # Compute confidence interval
    quantiles = np.quantile(boot_vals, [alpha/2, 1-alpha/2])
    ci = (quantiles[0], quantiles[1])
    
    return {
        'se': se,
        'ci': ci,
        'boot_vals': boot_vals
    }


def jackknife_se(estimate: SynthDIDEstimate, 
                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute jackknife standard errors for a synthdid estimate.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate for which to compute standard errors
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with standard error ('se'), confidence interval ('ci'), 
        and jackknife values ('jack_vals')
    """
    setup = estimate.setup
    orig_est = float(estimate.estimate)
    
    # Get original dimensions
    N0 = setup['N0']
    T0 = setup['T0']
    N = setup['Y'].shape[0]
    T = setup['Y'].shape[1]
    
    # Prepare estimator
    estimator_name = estimate.estimator
    estimator_fun = {
        "synthdid_estimate": synthdid_estimate,
        "sc_estimate": sc_estimate,
        "did_estimate": did_estimate
    }[estimator_name]
    
    # Store jackknife estimates
    jack_vals_units = np.zeros(N)
    jack_vals_times = np.zeros(T)
    
    # Leave-one-unit-out estimates
    for i in range(N):
        # Skip if removing would leave no treated or control units
        if i < N0 and N0 == 1:
            warnings.warn("Can't jackknife with only one control unit")
            jack_vals_units[i] = orig_est
            continue
        if i >= N0 and N - N0 == 1:
            warnings.warn("Can't jackknife with only one treated unit")
            jack_vals_units[i] = orig_est
            continue
        
        # Get units without unit i
        units = np.arange(N) != i
        
        # Adjust N0 if removing a control unit
        N0_adj = N0 - 1 if i < N0 else N0
        
        # Create jackknife sample
        Y_jack = setup['Y'][units, :]
        
        # Create covariates if they exist
        X_jack = None
        if setup['X'].size > 0:
            X_jack = setup['X'][units, :, :]
        
        # Estimate with jackknife sample
        jack_est = estimator_fun(Y_jack, N0_adj, T0, X=X_jack, **estimate.opts)
        jack_vals_units[i] = float(jack_est.estimate)
    
    # Leave-one-time-out estimates
    for j in range(T):
        # Skip if removing would leave no pre or post periods
        if j < T0 and T0 == 1:
            warnings.warn("Can't jackknife with only one pre-treatment period")
            jack_vals_times[j] = orig_est
            continue
        if j >= T0 and T - T0 == 1:
            warnings.warn("Can't jackknife with only one post-treatment period")
            jack_vals_times[j] = orig_est
            continue
        
        # Get periods without period j
        periods = np.arange(T) != j
        
        # Adjust T0 if removing a pre-treatment period
        T0_adj = T0 - 1 if j < T0 else T0
        
        # Create jackknife sample
        Y_jack = setup['Y'][:, periods]
        
        # Create covariates if they exist
        X_jack = None
        if setup['X'].size > 0:
            X_jack = setup['X'][:, periods, :]
        
        # Estimate with jackknife sample
        jack_est = estimator_fun(Y_jack, N0, T0_adj, X=X_jack, **estimate.opts)
        jack_vals_times[j] = float(jack_est.estimate)
    
    # Combine unit and time jackknife estimates (pseudo-values)
    n_total = N + T
    pseudo_vals = np.concatenate([
        n_total * orig_est - (n_total - 1) * jack_vals_units,
        n_total * orig_est - (n_total - 1) * jack_vals_times
    ])
    
    # Compute standard error
    se = np.std(pseudo_vals) / np.sqrt(n_total)
    
    # Normal approximation for confidence interval
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    ci = (orig_est - z * se, orig_est + z * se)
    
    return {
        'se': se,
        'ci': ci,
        'jack_vals_units': jack_vals_units,
        'jack_vals_times': jack_vals_times,
        'pseudo_vals': pseudo_vals
    }


def placebo_se(estimate: SynthDIDEstimate, replications: int = 200,
              alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute placebo-based standard errors for a synthdid estimate.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate for which to compute standard errors
    replications : int, default=200
        Number of placebo replications
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with standard error ('se'), confidence interval ('ci'), 
        and placebo distribution ('placebo_vals')
    """
    setup = estimate.setup
    orig_est = float(estimate.estimate)
    
    # Get original dimensions
    N0 = setup['N0']
    T0 = setup['T0']
    
    # Store placebo estimates
    placebo_vals = np.zeros(replications)
    
    # Prepare estimator
    estimator_name = estimate.estimator
    estimator_fun = {
        "synthdid_estimate": synthdid_estimate,
        "sc_estimate": sc_estimate,
        "did_estimate": did_estimate
    }[estimator_name]
    
    # Run placebo tests with random treatment timing
    for i in range(replications):
        # Randomly select a treatment period in the pre-treatment window
        if T0 <= 1:
            warnings.warn("Not enough pre-treatment periods for placebo test")
            placebo_vals[i] = 0
            continue
            
        placebo_T0 = np.random.randint(1, T0)
        
        # Restrict to pre-treatment period for placebo
        Y_placebo = setup['Y'][:, :T0]
        
        # Create covariates if they exist
        X_placebo = None
        if setup['X'].size > 0:
            X_placebo = setup['X'][:, :T0, :]
        
        # Estimate with placebo treatment timing
        placebo_est = estimator_fun(Y_placebo, N0, placebo_T0, X=X_placebo, **estimate.opts)
        placebo_vals[i] = float(placebo_est.estimate)
    
    # Compute standard error
    se = np.std(placebo_vals)
    
    # Normal approximation for confidence interval
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    ci = (orig_est - z * se, orig_est + z * se)
    
    return {
        'se': se,
        'ci': ci,
        'placebo_vals': placebo_vals
    }


def synthdid_se(estimate: SynthDIDEstimate, 
               method: str = "jackknife", 
               replications: int = 200,
               alpha: float = 0.05) -> Union[float, Dict[str, Any]]:
    """
    Compute standard errors for a synthdid estimate.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate for which to compute standard errors
    method : str, default="jackknife"
        Method to use for standard error computation.
        Options: "bootstrap", "jackknife", "placebo"
    replications : int, default=200
        Number of replications for bootstrap or placebo methods
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha)
    
    Returns:
    --------
    Union[float, Dict[str, Any]]
        If return_all=False, returns just the standard error.
        If return_all=True, returns a dictionary with standard error ('se'),
        confidence interval ('ci'), and method-specific values.
    """
    if method == "bootstrap":
        return bootstrap_se(estimate, replications, alpha)
    elif method == "jackknife":
        return jackknife_se(estimate, alpha)
    elif method == "placebo":
        return placebo_se(estimate, replications, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")


def vcov(estimate: SynthDIDEstimate, 
        method: str = "jackknife", 
        replications: int = 200) -> Union[float, Dict[str, Any]]:
    """
    Alias for synthdid_se that returns the variance (se^2) for compatibility with R.
    
    Parameters:
    -----------
    estimate : SynthDIDEstimate
        The estimate for which to compute variance
    method : str, default="jackknife"
        Method to use for variance computation.
        Options: "bootstrap", "jackknife", "placebo"
    replications : int, default=200
        Number of replications for bootstrap or placebo methods
    
    Returns:
    --------
    Union[float, Dict[str, Any]]
        The variance estimate or dictionary with all results
    """
    result = synthdid_se(estimate, method, replications)
    result['vcov'] = result['se']**2
    return result 