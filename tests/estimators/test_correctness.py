"""Regression tests for previously silent changes to causal estimands."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library import estimators as fit
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, create_event_study_specification,
    create_staggered_did_specification, create_synthdid_specification,
    create_uplift_specification,
)
from pyautocausal.pipelines.library.synthdid.utils import panel_matrices
from pyautocausal.pipelines.library.synthdid.synthdid import (
    synthdid_estimate, synthdid_effect_curve, did_estimate,
)
from pyautocausal.pipelines.library.synthdid.vcov import (
    bootstrap_se, jackknife_se, placebo_se,
)
from pyautocausal.pipelines.library.synthdid.plot import plot_synthdid


def panel(effect=9, n_treated=1):
    return pd.DataFrame([
        {'id_unit': f'u{u}', 't': t, 'treat': int(u < n_treated and t >= 3),
         'y': u + t + effect * (u < n_treated and t >= 3), 'x': 10 * u + t}
        for u in range(5) for t in range(5)
    ])


def test_cross_sectional_treatment_sign_survives_every_row_permutation():
    original = pd.DataFrame({'y': [10., 10., 0., 0.], 'treat': [1, 1, 0, 0]})
    for order in itertools.permutations(range(4)):
        data = original.iloc[list(order)]
        spec = create_cross_sectional_specification(data, control_cols=[])
        assert fit.fit_ols(spec).model.params['treat'] == pytest.approx(10)
        pd.testing.assert_frame_equal(data, original.iloc[list(order)])


def test_explicit_label_and_boolean_treatment_contract():
    data = pd.DataFrame({'y': [10, 10, 0, 0], 'treat': ['yes', 'yes', 'no', 'no']})
    with pytest.raises(ValueError, match='treatment_value'):
        create_cross_sectional_specification(data)
    spec = create_cross_sectional_specification(data, treatment_value='yes')
    assert fit.fit_ols(spec).model.params['treat'] == pytest.approx(10)
    data['treat'] = [True, True, False, False]
    assert fit.fit_ols(create_cross_sectional_specification(data)).model.params['treat'] == pytest.approx(10)


def test_control_contract_and_missing_data_policy():
    data = pd.DataFrame({'y': [0, 1, 3, 4], 'treat': [0, 0, 1, 1],
                         'x': [1, 2, 1, 2], 'irrelevant': [None] * 4})
    assert create_cross_sectional_specification(data).control_cols == ['x']
    assert create_cross_sectional_specification(data, control_cols=[]).control_cols == []
    assert len(create_cross_sectional_specification(data).data) == 4
    with pytest.raises(ValueError, match='missing required columns'):
        create_cross_sectional_specification(data, control_cols=['unknown'])
    data.loc[0, 'x'] = np.nan
    with pytest.raises(ValueError, match='Missing required values'):
        create_cross_sectional_specification(data, missing_strategy='reject')
    result = create_cross_sectional_specification(data)
    assert result.data.attrs['preparation']['rows_dropped'] == 1


@pytest.mark.parametrize('n_treated', [1, 2])
def test_synthetic_did_boundary_and_row_order(n_treated):
    data = panel(n_treated=n_treated)
    for seed in range(3):
        spec = create_synthdid_specification(data.sample(frac=1, random_state=seed), control_cols=[])
        assert spec.T0 == 3
        model = fit.fit_synthdid_estimator(spec).model
        assert float(model) == pytest.approx(9)
        np.testing.assert_allclose(synthdid_effect_curve(model), [9, 9])


def test_synthetic_covariates_follow_outcome_unit_order():
    spec = create_synthdid_specification(panel(), control_cols=['x'])
    np.testing.assert_array_equal(spec.X[:, :, 0],
                                  np.array([10 * u + np.arange(5) for u in [1, 2, 3, 4, 0]]))


@pytest.mark.parametrize('defect, match', [
    ('missing_cell', 'balanced'), ('duplicate', 'exactly one'),
    ('reversal', 'permanent'), ('first_period', 'pre-treatment'),
    ('no_donors', 'donors'), ('staggered', 'simultaneous'),
])
def test_invalid_synthetic_design_fails_before_estimation(defect, match):
    data = panel()
    if defect == 'missing_cell':
        data = data.iloc[1:]
    elif defect == 'duplicate':
        data = pd.concat([data, data.iloc[[0]]])
    elif defect == 'reversal':
        data.loc[(data.id_unit == 'u0') & (data.t == 4), 'treat'] = 0
    elif defect == 'first_period':
        data.loc[data.id_unit == 'u0', 'treat'] = 1
    elif defect == 'no_donors':
        data.loc[data.t >= 3, 'treat'] = 1
    else:
        data.loc[(data.id_unit == 'u1') & (data.t == 4), 'treat'] = 1
    with pytest.raises(ValueError, match=match):
        create_synthdid_specification(data, control_cols=[])


def test_synthetic_last_period_and_explicit_noise_boundary():
    data = panel()
    data.treat = ((data.id_unit == 'u0') & (data.t == 4)).astype(int)
    data.y = data.t + 9 * data.treat
    spec = create_synthdid_specification(data, control_cols=[])
    assert spec.T0 == 4
    assert float(fit.fit_synthdid_estimator(spec).model) == pytest.approx(9)
    with pytest.raises(ValueError, match='differences'):
        synthdid_estimate(np.array([[1, 2], [1, 5]]), 1, 1)
    assert float(synthdid_estimate(np.array([[1, 2], [1, 5]]), 1, 1, noise_level=0)) == pytest.approx(3)


def test_two_period_did_and_fixed_contrasts_do_not_require_noise_estimation():
    Y = np.array([[1., 2.], [1., 5.]])
    assert float(did_estimate(Y, 1, 1)) == pytest.approx(3)
    estimate = synthdid_estimate(Y, 1, 1,
                                weights={'omega': np.ones(1), 'lambda': np.ones(1)})
    assert float(estimate) == pytest.approx(3)
    # Asking to fit either weight vector still needs a supplied noise level.
    with pytest.raises(ValueError, match='differences'):
        synthdid_estimate(Y, 1, 1, weights={'omega': np.ones(1)}, update_lambda=True)


def test_synthetic_plot_reports_actual_signed_effect_and_treatment_boundary():
    spec = create_synthdid_specification(panel(effect=-9), control_cols=[])
    est = fit.fit_synthdid_estimator(spec).model
    before = np.random.get_state()
    fig, ax = plot_synthdid(est, ci_level=0, show_guides=False)
    curve = next(line for line in ax.lines if line.get_label() == 'Effect curve')
    np.testing.assert_array_equal(curve.get_xdata(), [3, 4])
    np.testing.assert_allclose(curve.get_ydata(), [-9, -9])
    boundary = next(line for line in ax.lines if line.get_label() == 'Treatment begins')
    np.testing.assert_array_equal(boundary.get_xdata(), [3, 3])
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    plt.close(fig)


def test_synthetic_fixed_weights_are_not_sparsified_or_mutated():
    Y = np.random.default_rng(381).normal(size=(8, 7))
    weights = {'omega': np.array([.01, .02, .07, .3, .6]), 'lambda': np.array([.01, .09, .3, .6])}
    est = synthdid_estimate(Y, 5, 4, weights=weights)
    np.testing.assert_array_equal(est.weights['omega'], weights['omega'])
    np.testing.assert_array_equal(est.weights['lambda'], weights['lambda'])
    assert 'vals' not in weights


def test_synthetic_inference_matches_r_reference_point_and_jackknife():
    # Values generated by unmodified synth-inference/synthdid R functions:
    # R/{utils,solver,synthdid,vcov}.R, retrieved 2026-10-07.
    # Direct check also verified both fitted weight vectors to machine precision.
    Y = np.random.default_rng(381).normal(size=(8, 7)) + np.arange(8)[:, None] + np.arange(7)[None, :]
    Y[5:, 4:] += 2
    est = synthdid_estimate(Y, 5, 4)
    assert float(est) == pytest.approx(0.30084457615458027, abs=1e-12)
    assert jackknife_se(est)['se'] == pytest.approx(0.63641129175261912, abs=1e-12)


def test_synthetic_resampling_keeps_actual_periods_and_uses_only_donors_for_placebos(monkeypatch):
    import importlib
    module = importlib.import_module("pyautocausal.pipelines.library.synthdid.vcov")
    Y = np.arange(8 * 5).reshape(8, 5).astype(float)
    est = did_estimate(Y, 5, 3)
    calls = []
    original = module.synthdid_estimate
    def capture(Y, N0, T0, **kwargs):
        calls.append((Y.copy(), N0, T0))
        return original(Y, N0, T0, **kwargs)
    monkeypatch.setattr(module, 'synthdid_estimate', capture)
    placebo_se(est, 8, random_state=2)
    assert all(T0 == 3 and N0 == 2 and len(y) == 5 for y, N0, T0 in calls)
    assert all(np.all(y[:, 0] < 25) for y, _, _ in calls)
    assert all(np.all(np.diff(y, axis=1) == 1) for y, _, _ in calls)
    calls.clear()
    bootstrap_se(est, 8, random_state=2)
    assert all(T0 == 3 and len(y) == 8 and np.all(np.diff(y, axis=1) == 1)
               for y, _, T0 in calls)


def test_synthetic_inference_unavailable_designs_are_explicit():
    est = synthdid_estimate(np.arange(20).reshape(4, 5), 3, 3)
    for method in [bootstrap_se, jackknife_se]:
        with pytest.raises(ValueError, match='treated'):
            method(est)
    with pytest.raises(ValueError, match='replications'):
        placebo_se(est, replications=1)


def uplift_data():
    return pd.DataFrame({'y': [0] * 800 + [1] * 200 + [0] * 200 + [1] * 800,
                         'treat': [0] * 1000 + [1] * 1000, 'x': 1.,
                         'id_unit': np.arange(2000)})


@pytest.mark.parametrize('learner', [fit.fit_s_learner, fit.fit_t_learner, fit.fit_x_learner])
def test_meta_learners_preserve_constant_effect_and_natural_probabilities(learner):
    spec = learner(create_uplift_specification(uplift_data(), control_cols=['x']))
    cate = next(iter(spec.cate_estimates.values()))
    assert np.isfinite(cate).all()
    assert np.std(cate) < 1e-12
    assert spec.ate_estimate == pytest.approx(0.6, abs=0.025)
    assert spec.ate_estimate == pytest.approx(cate.mean())
    assert spec.ate_ci is None
    assert spec.inference['status'] == 'unavailable'
    assert spec.experimental


@pytest.mark.parametrize('learner', [fit.fit_t_learner, fit.fit_x_learner, fit.fit_double_ml_binary])
def test_uplift_supports_single_outcome_class_in_each_arm(learner):
    data = uplift_data()
    data.y = data.treat
    spec = learner(create_uplift_specification(data, control_cols=[]))
    assert spec.ate_estimate == pytest.approx(1)


def test_aipw_returns_ate_without_invented_individual_effects():
    spec = fit.fit_double_ml_binary(create_uplift_specification(uplift_data(), control_cols=['x']))
    assert spec.ate_estimate == pytest.approx(.6, abs=.01)
    assert spec.cate_estimates is None
    assert spec.ate_ci[0] < .6 < spec.ate_ci[1]
    assert spec.inference['status'] == 'asymptotic'
    assert len(spec.models['fold_models']) == 5


@pytest.mark.parametrize('defect, match', [('one_arm', 'both binary arms'),
                                         ('overlap', 'overlap'), ('repeated', 'independent')])
def test_aipw_invalid_design_is_rejected(defect, match):
    data = uplift_data()
    spec = create_uplift_specification(data, control_cols=['x'])
    if defect == 'one_arm':
        spec.data.treat = 0
    elif defect == 'overlap':
        spec.data.x = 100 * spec.data.treat
    else:
        spec.data.id_unit = 1
    with pytest.raises(ValueError, match=match):
        fit.fit_double_ml_binary(spec)


def test_event_study_never_treated_rows_have_no_event_dummies():
    spec = create_event_study_specification(panel(), control_cols=[])
    controls = spec.data.loc[spec.data.id_unit != 'u0', spec.event_cols]
    assert not controls.to_numpy().any()


def test_parallel_estimators_do_not_overwrite_shared_specification_results():
    spec = create_uplift_specification(uplift_data(), control_cols=['x'])
    first = fit.fit_t_learner(spec)
    second = fit.fit_double_ml_binary(spec)
    assert first is not second and first is not spec and second is not spec
    assert first.model_type == 't-learner'
    assert first.cate_estimates is not None
    assert second.model_type == 'double-ml'
    assert spec.model is None


def test_synthetic_refits_match_official_r_resampling_reference():
    from pyautocausal.pipelines.library.synthdid.vcov import _refit
    Y = np.random.default_rng(381).normal(size=(8, 7)) + np.arange(8)[:, None] + np.arange(7)[None, :]
    Y[5:, 4:] += 2
    est = synthdid_estimate(Y, 5, 4)
    # Bootstrap and placebo sample fits from unmodified R synthdid_estimate,
    # using R vcov.R's Algorithms 2/4 sampling and initial-weight rules.
    for units, n_control, expected in [
        ([0, 0, 1, 3, 5, 5, 6, 7], 4, 0.4356768318225927),
        ([0, 1, 1, 2, 3, 4, 4, 7], 7, 0.37010284195825843),
        ([4, 1, 2, 0, 3], 2, 0.58956248289243285),
        ([1, 3, 4, 0, 2], 2, 0.7923449758003358),
    ]:
        assert _refit(est, np.array(units), n_control) == pytest.approx(expected, abs=1e-12)


def test_synthetic_covariate_solver_matches_official_r_reference():
    rng = np.random.default_rng(92)
    X = rng.normal(scale=.25, size=(8, 7))
    Y = rng.normal(size=(8, 7)) + 2 * X
    Y[5:, 4:] += 2
    est = synthdid_estimate(Y, 5, 4, X=X)
    # Unmodified R synthdid_estimate including sc.weight.fw.covariates.
    assert float(est) == pytest.approx(2.6405636648830888, abs=1e-12)
    assert est.weights['beta'][0] == pytest.approx(.12140539100077039, abs=1e-12)


def test_cross_sectional_public_pipeline_keeps_positive_treatment_effect(tmp_path):
    from pyautocausal.pipelines.example_graph import create_cross_sectional_graph
    rng = np.random.default_rng(4)
    treatment = np.repeat([1, 0], 100)
    # Exact balanced X distributions prevent adjustment from changing the effect.
    x = np.tile(rng.normal(size=100), 2)
    data = pd.DataFrame({'id_unit': np.arange(200), 'treat': treatment,
                         'x': x, 'y': 10 * treatment + 2 * x})
    graph = create_cross_sectional_graph(tmp_path)
    graph.fit(df=data)
    fitted = graph.get('ols_stand').get_result_value()
    assert fitted.model.params['treat'] == pytest.approx(10)
    assert (tmp_path / 'text').is_dir()


@pytest.mark.parametrize('learner', [fit.fit_s_learner, fit.fit_t_learner, fit.fit_x_learner])
def test_zero_outcomes_remain_zero_without_artificial_heterogeneity(learner):
    data = uplift_data()
    data.y = 0
    result = learner(create_uplift_specification(data, control_cols=[]))
    assert result.ate_estimate == 0
    np.testing.assert_array_equal(next(iter(result.cate_estimates.values())), np.zeros(len(data)))


@pytest.mark.parametrize('columns', [[], ['treat', 'other']])
def test_unsupported_treatment_column_counts_raise_clear_error(columns):
    with pytest.raises(ValueError, match='Exactly one treatment'):
        create_cross_sectional_specification(uplift_data(), treatment_cols=columns)


class _PlaceboRecorder:
    def __init__(self):
        self.calls = []

    def in_space_placebo(self, n_placebo):
        self.calls.append(('space', n_placebo))

    def in_time_placebo(self, period):
        self.calls.append(('time', period))


@pytest.mark.parametrize('random_state', [42, 9])
def test_hainmueller_placebo_period_is_reproducible_without_global_rng_mutation(random_state):
    spec = create_synthdid_specification(panel(), control_cols=[])
    spec.hainmueller_model = _PlaceboRecorder()
    before = np.random.get_state()
    first = fit.fit_hainmueller_placebo_test(spec, random_state=random_state)
    second = fit.fit_hainmueller_placebo_test(spec, random_state=random_state)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert first.hainmueller_model.calls == second.hainmueller_model.calls
    assert first.hainmueller_placebo_metadata == second.hainmueller_placebo_metadata
    assert first.hainmueller_placebo_metadata['random_state'] == random_state
    period = first.hainmueller_placebo_metadata['placebo_period']
    assert period in [1, 2]  # Both have observations before and after pseudo-adoption.
    assert first.hainmueller_model.calls[-1] == ('time', period)
    assert spec.hainmueller_model.calls == []
    assert not hasattr(spec, 'hainmueller_placebo_metadata')


def test_hainmueller_placebo_rejects_no_pretreatment_comparison():
    data = panel()
    data.treat = ((data.id_unit == 'u0') & (data.t >= 1)).astype(int)
    spec = create_synthdid_specification(data, control_cols=[])
    spec.hainmueller_model = _PlaceboRecorder()
    with pytest.raises(ValueError, match='two observed pre-treatment'):
        fit.fit_hainmueller_placebo_test(spec)
    assert spec.hainmueller_model.calls == []
