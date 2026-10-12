"""Unit-level inference for synthetic DiD (Arkhangelsky et al., Algorithms 2–4).

Reference: https://github.com/synth-inference/synthdid/blob/master/R/vcov.R
The normal intervals rely on the paper's sampling and regularity assumptions;
placebo inference additionally assumes comparable noise for donors and treated
units. Time periods are kept together to preserve within-unit dependence.
"""
import copy
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from .synthdid import SynthDIDEstimate, synthdid_estimate


def _validate(alpha, replications=None):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if replications is not None and (not isinstance(replications, (int, np.integer)) or replications < 2):
        raise ValueError("At least two replications are required")


def _normalize(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(len(weights)) / len(weights)


def _refit(estimate, units, n_control, *, fixed_weights=False):
    """Refit an entire unit sample using the original regularization and T0."""
    setup = estimate.setup
    opts = copy.deepcopy(estimate.opts)
    weights = copy.deepcopy(estimate.weights)
    # Sample order always places its controls first. For placebo, units are
    # exclusively original controls, including the new pseudo-treated units.
    weights['omega'] = _normalize(estimate.weights['omega'][units[:n_control]])
    if fixed_weights:
        opts.update(update_omega=False, update_lambda=False)
    X = setup['X'][units] if setup['X'].size else None
    # Call the base estimator with the original options, which also represent
    # fixed DID weights or the SC zero pre-period contrast correctly.
    return float(synthdid_estimate(setup['Y'][units], n_control, setup['T0'],
                                  X=X, weights=weights, noise_level=0.0, **opts))


def _result(estimate, se, alpha, method, **samples):
    center = float(estimate)
    margin = norm.ppf(1 - alpha / 2) * se
    return {'status': 'asymptotic', 'method': method, 'se': float(se),
            'ci': (center - margin, center + margin), 'confidence_level': 1 - alpha,
            **samples}


def bootstrap_se(estimate: SynthDIDEstimate, replications: int = 200,
                 alpha: float = 0.05, random_state=None) -> Dict[str, Any]:
    """Algorithm 2: resample units and refit; requires multiple treated units."""
    _validate(alpha, replications)
    N0, N = estimate.setup['N0'], len(estimate.setup['Y'])
    if N - N0 < 2:
        raise ValueError("Bootstrap inference requires at least two treated units; use placebo")
    rng = np.random.default_rng(random_state)
    values = []
    while len(values) < replications:
        units = np.sort(rng.choice(N, size=N, replace=True))
        controls = int(np.sum(units < N0))
        if 0 < controls < N:
            values.append(_refit(estimate, units, controls))
    values = np.asarray(values)
    return _result(estimate, np.std(values, ddof=0), alpha, 'unit bootstrap', boot_vals=values)


def jackknife_se(estimate: SynthDIDEstimate, alpha: float = 0.05) -> Dict[str, Any]:
    """Algorithm 3: leave one unit out, keeping and renormalizing fitted weights."""
    _validate(alpha)
    N0, N = estimate.setup['N0'], len(estimate.setup['Y'])
    if N - N0 < 2 or np.count_nonzero(estimate.weights['omega']) < 2:
        raise ValueError("Jackknife requires two treated units and two positive-weight controls")
    values = np.array([_refit(estimate, np.delete(np.arange(N), i),
                             N0 - int(i < N0), fixed_weights=True) for i in range(N)])
    se = np.sqrt((N - 1) / N * np.sum((values - values.mean()) ** 2))
    return _result(estimate, se, alpha, 'fixed-weight unit jackknife', jack_vals_units=values)


def placebo_se(estimate: SynthDIDEstimate, replications: int = 200,
               alpha: float = 0.05, random_state=None) -> Dict[str, Any]:
    """Algorithm 4: assign pseudo-treatment to donors at the actual boundary."""
    _validate(alpha, replications)
    N0 = estimate.setup['N0']
    N1 = len(estimate.setup['Y']) - N0
    if N0 <= N1:
        raise ValueError("Placebo inference requires more controls than treated units")
    rng = np.random.default_rng(random_state)
    values = np.array([_refit(estimate, rng.permutation(N0), N0 - N1)
                       for _ in range(replications)])
    return _result(estimate, np.std(values, ddof=0), alpha,
                   'control-unit placebo', placebo_vals=values)


def synthdid_se(estimate: SynthDIDEstimate, method: str = 'jackknife',
                replications: int = 200, alpha: float = 0.05,
                random_state=None) -> Dict[str, Any]:
    """Return standard error, normal interval and resampling diagnostics."""
    if method == 'bootstrap':
        return bootstrap_se(estimate, replications, alpha, random_state)
    if method == 'jackknife':
        return jackknife_se(estimate, alpha)
    if method == 'placebo':
        return placebo_se(estimate, replications, alpha, random_state)
    raise ValueError(f"Unknown method: {method}")


def vcov(estimate: SynthDIDEstimate, method: str = 'jackknife',
         replications: int = 200, random_state=None) -> Dict[str, Any]:
    """Return the inference result with variance in the ``vcov`` field."""
    result = synthdid_se(estimate, method, replications, random_state=random_state)
    result['vcov'] = result['se'] ** 2
    return result
