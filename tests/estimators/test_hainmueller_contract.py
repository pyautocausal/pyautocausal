"""Synthetic-control input contracts must survive the vendored backend."""
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library.estimators import (
    fit_hainmueller_synth_estimator, fit_hainmueller_placebo_test,
)
from pyautocausal.pipelines.library.specifications import create_synthdid_specification


def panel():
    return pd.DataFrame([
        {'id_unit': str(unit), 't': time,
         'treat': int(unit == 0 and time >= 4),
         'y': (unit + 1) * time + unit + 5 * (unit == 0 and time >= 4),
         'x': 10 * unit + 3 * time}
        for unit in range(4) for time in range(6)
    ])


def fitted(data, controls):
    spec = create_synthdid_specification(data, control_cols=controls)
    return fit_hainmueller_synth_estimator(spec).hainmueller_model.original_data


def assert_same_fit(first, second):
    np.testing.assert_array_equal(first.control_units, second.control_units)
    for attribute in ['treated_outcome_all', 'control_outcome_all',
                      'treated_covariates', 'control_covariates', 'w', 'synth_outcome']:
        np.testing.assert_allclose(getattr(first, attribute), getattr(second, attribute), atol=1e-9)


def test_row_permutation_preserves_units_periods_and_fitted_outcomes():
    original = panel()
    shuffled = original.sample(frac=1, random_state=7)
    before = shuffled.copy(deep=True)
    reference = fitted(original, ['x'])
    actual = fitted(shuffled, ['x'])
    assert_same_fit(reference, actual)
    # Verify matrix meaning, independently of the fit on ordered rows.
    np.testing.assert_array_equal(actual.control_outcome_all,
                                  original.loc[original.id_unit != '0'].pivot(
                                      index='t', columns='id_unit', values='y').to_numpy())
    pd.testing.assert_frame_equal(shuffled, before)


@pytest.mark.parametrize('controls', [[], ['x']])
def test_only_requested_controls_enter_the_fitted_predictor_set(controls):
    original = panel()
    with_extra = original.assign(unused_numeric=np.nan, another_unused=1e12)
    # Missing or extreme values in excluded columns must neither remove rows
    # nor contaminate the fitted predictors.
    kept = ['id_unit', 't', 'treat', 'y', *controls]
    reference = fitted(original[kept], controls)
    actual = fitted(with_extra, controls)
    assert actual.covariates == ['y', *controls]
    assert_same_fit(reference, actual)


def test_fractional_periods_do_not_truncate_the_treatment_boundary():
    original = panel().drop(columns='x')
    fractional = original.assign(t=original.t / 10)
    reference = fitted(original, [])
    actual = fitted(fractional, [])
    assert actual.treatment_period == pytest.approx(.4)
    assert actual.periods_pre_treatment == 4
    assert_same_fit(reference, actual)


@pytest.mark.parametrize('constant_outcome', [False, True])
def test_constant_and_zero_predictors_produce_finite_placebo_paths(constant_outcome):
    data = panel().assign(zero=0., constant=7.)
    if constant_outcome:
        data.y = 5 * data.treat
    spec = create_synthdid_specification(data, control_cols=['zero', 'constant'])
    result = fit_hainmueller_placebo_test(fit_hainmueller_synth_estimator(spec))
    model = result.hainmueller_model.original_data
    assert np.isfinite(model.in_space_placebos).all()
    assert np.isfinite(model.in_time_placebo_outcome).all()
    assert len(model.in_space_placebos) == 3
    assert all(np.asarray(path).size == 6 for path in model.in_space_placebos)


@pytest.mark.parametrize('placebo', ['in-space', 'in-time'])
def test_placebo_retains_best_fit_instead_of_last_objective_evaluation(placebo):
    from types import SimpleNamespace
    from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods.optimize import Optimize
    optimizer = Optimize()
    optimizer.method = 'SC'
    data = SimpleNamespace()
    treated = np.array([[.25], [.75]])
    controls = np.array([[0., 1.], [0., 1.]])
    outcomes = np.array([[0., 1.]])
    # The first predictor fits the actual treated outcome exactly; the second
    # evaluation is a worse pre-period fit and must not replace the optimum.
    for importance in [np.array([1., 0.]), np.array([0., 1.])]:
        optimizer.total_loss(importance, np.array([[.25]]), treated, outcomes,
                             controls, treated - controls, 0, placebo, data)
    weights = getattr(data, placebo.replace('-', '_') + '_placebo_w')
    np.testing.assert_allclose(weights.ravel(), [.75, .25], atol=1e-6)


@pytest.mark.parametrize('placebo', [False, 'in-space', 'in-time'])
def test_solver_failure_reports_deliberate_error_without_reusing_old_weights(monkeypatch, placebo):
    from types import SimpleNamespace
    from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods.optimize import Optimize
    import cvxpy
    monkeypatch.setattr(cvxpy.Problem, 'solve', lambda *args, **kwargs: np.inf)
    optimizer = Optimize()
    optimizer.method = 'SC'
    data = SimpleNamespace(n_covariates=1, w=None, min_loss=np.inf,
                           in_space_placebo_w=np.array([[.5], [.5]]),
                           in_time_placebo_w=np.array([[.5], [.5]]))
    treated = np.array([[.25]])
    controls = np.array([[0., 1.]])
    with pytest.raises(ValueError, match='could not find finite feasible donor weights'):
        optimizer.optimize(treated, treated, controls, controls,
                           treated - controls, data, placebo, 0, steps=1)
