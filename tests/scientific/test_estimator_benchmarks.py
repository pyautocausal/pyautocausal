"""Prespecified release benchmarks; these do not establish general validity.

Coverage tolerances are fixed at >=80% over 30 or 40 seeds for nominal 95%
intervals, a coarse guard against severe undercoverage rather than a claim of
exact finite-sample calibration. Each experiment below states its DGP and target.
"""
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library.estimators import (
    fit_double_ml_binary, fit_s_learner, fit_t_learner, fit_x_learner,
)
from pyautocausal.pipelines.library.specifications import create_uplift_specification
from pyautocausal.pipelines.library.synthdid.synthdid import synthdid_estimate
from pyautocausal.pipelines.library.synthdid.vcov import bootstrap_se, jackknife_se, placebo_se


@pytest.mark.scientific
def test_cross_fitted_aipw_randomized_binary_ate_coverage(record_property):
    # Independent units; Bernoulli(.5) assignment; binary covariate; outcome
    # probabilities .2 + .1 X + .3 D. True ATE=.3, good overlap, and both
    # conditional outcome models are represented by a one-split forest.
    estimates, covered = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed + 800)
        n = 1600
        x, t = rng.binomial(1, .5, size=(2, n))
        y = rng.binomial(1, .2 + .1 * x + .3 * t)
        data = pd.DataFrame({'id_unit': np.arange(n), 'x': x, 'treat': t, 'y': y})
        result = fit_double_ml_binary(create_uplift_specification(data, control_cols=['x']))
        estimates.append(result.ate_estimate)
        covered.append(result.ate_ci[0] <= .3 <= result.ate_ci[1])
    record_property("mean_estimate", float(np.mean(estimates)))
    record_property("coverage", float(np.mean(covered)))
    assert abs(np.mean(estimates) - .3) < .025
    assert np.mean(covered) >= .8, f'ATE coverage {sum(covered)}/40'


@pytest.mark.scientific
@pytest.mark.parametrize('learner', [fit_s_learner, fit_t_learner, fit_x_learner])
def test_meta_learner_recovers_prespecified_heterogeneous_effects(learner):
    # Four balanced (X,D) cells each contain exact binary response frequencies.
    # p0=.2, tau(0)=.1, tau(1)=.5, so sample-average CATE=.3.
    rows = []
    for x in [0, 1]:
        for t in [0, 1]:
            p = .2 + t * (.1 + .4 * x)
            for i in range(1000):
                rows.append({'id_unit': len(rows), 'x': x, 'treat': t,
                             'y': int(i < round(p * 1000))})
    data = pd.DataFrame(rows)
    result = learner(create_uplift_specification(data, control_cols=['x']))
    cate = next(iter(result.cate_estimates.values()))
    assert cate[data.x == 0].mean() == pytest.approx(.1, abs=.04)
    assert cate[data.x == 1].mean() == pytest.approx(.5, abs=.04)
    assert result.ate_estimate == pytest.approx(.3, abs=.04)
    assert result.ate_ci is None  # This benchmark does not validate inference.


@pytest.mark.scientific
def test_synthetic_did_unit_inference_coverage_with_serial_dependence(record_property):
    # Independent units, additive unit/time effects, normal AR(.4) disturbances,
    # simultaneous constant effect 2, 40 donors + 20 treated, 10 pre + 5 post.
    # Donors and treated share a common noise law (required for placebo SE).
    covered = {method: [] for method in ['bootstrap', 'jackknife', 'placebo']}
    estimates = []
    for seed in range(30):
        rng = np.random.default_rng(seed + 1900)
        n, T, N0, T0 = 60, 15, 40, 10
        error = rng.normal(size=(n, T))
        for t in range(1, T):
            error[:, t] += .4 * error[:, t - 1]
        Y = rng.normal(size=(n, 1)) + np.arange(T)[None, :] * .15 + error
        Y[N0:, T0:] += 2
        est = synthdid_estimate(Y, N0, T0)
        estimates.append(float(est))
        results = {
            'bootstrap': bootstrap_se(est, replications=80, random_state=seed),
            'jackknife': jackknife_se(est),
            'placebo': placebo_se(est, replications=80, random_state=seed),
        }
        for method, result in results.items():
            covered[method].append(result['ci'][0] <= 2 <= result['ci'][1])
    record_property("mean_estimate", float(np.mean(estimates)))
    assert abs(np.mean(estimates) - 2) < .1
    for method, coverage in covered.items():
        record_property(method + "_coverage", float(np.mean(coverage)))
        assert np.mean(coverage) >= .8, f'{method} coverage {sum(coverage)}/30'
