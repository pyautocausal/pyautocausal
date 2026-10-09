"""C&S wrapper parity and a conditional-parallel-trends benchmark."""
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library import estimators as fit
from pyautocausal.pipelines.library.csdid.att_gt import ATTgt
from pyautocausal.pipelines.library.specifications import create_staggered_did_specification


def staggered_panel(seed=19):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(240):
        # Both cohorts and donors overlap in X but have different covariate
        # means, so unconditional parallel trends is deliberately false.
        cohort = 3 if unit < 80 else 4 if unit < 160 else 0
        x = rng.normal(loc=0.8 if cohort else 0)
        for time in range(1, 6):
            d = int(cohort > 0 and time >= cohort)
            rows.append({'id_unit': unit + 1, 't': time, 'x': x,
                         'treat': d, 'y': unit * .01 + time * (x + 1) + 2 * d + rng.normal(scale=.08)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize('wrapper, control_group', [
    (fit.fit_callaway_santanna_estimator, 'nevertreated'),
    (fit.fit_callaway_santanna_nyt_estimator, 'notyettreated'),
])
@pytest.mark.parametrize('controls', [[], ['x']])
def test_both_cs_wrappers_match_direct_backend_and_honor_controls(wrapper, control_group, controls):
    data = staggered_panel()
    spec = create_staggered_did_specification(data, control_cols=controls)
    raw = spec.data.copy()
    raw.treatment_time = raw.treatment_time.fillna(0)
    formula = 'y ~ x' if controls else 'y ~ 1'
    # Direct backend is independent of wrapper preparation and construction.
    direct = ATTgt(yname='y', tname='t', idname='id_unit', gname='treatment_time',
                   data=raw, xformla=formula, control_group=[control_group],
                   panel=True, allow_unbalanced_panel=True, anticipation=0,
                   cband=True, biters=1000, alp=.05).fit(
                       est_method='dr', base_period='varying', bstrap=True, random_state=42)
    result = wrapper(spec).model['att_gt_object']
    assert result.dp['xformla'] == formula
    assert result.dp['control_group'] == control_group
    np.testing.assert_allclose(result.results['att'], direct.results['att'], atol=1e-12)
    np.testing.assert_allclose(result.results['se'], direct.results['se'], atol=1e-12)
    results = pd.DataFrame(result.results)
    post = results.loc[results.year >= results.group, 'att']
    if controls:
        np.testing.assert_allclose(post, 2, atol=.08)
    else:
        assert np.mean(post) > 2.5


def test_cs_accepts_string_ids_and_zero_based_times_without_changing_effects():
    data = staggered_panel()
    original = fit.fit_callaway_santanna_estimator(
        create_staggered_did_specification(data, control_cols=['x'])).model['att_gt_object']
    data.id_unit = 'label_' + data.id_unit.astype(str)
    data.t -= 1
    recoded = fit.fit_callaway_santanna_estimator(
        create_staggered_did_specification(data, control_cols=['x'])).model['att_gt_object']
    np.testing.assert_allclose(recoded.results['att'], original.results['att'], atol=1e-12)


def test_cs_not_yet_treated_works_without_never_treated_units():
    data = staggered_panel()
    data.loc[data.id_unit > 160, 'treat'] = (data.loc[data.id_unit > 160, 't'] >= 5).astype(int)
    spec = create_staggered_did_specification(data, control_cols=['x'])
    result = fit.fit_callaway_santanna_nyt_estimator(spec).model['att_gt_object']
    assert result.dp['control_group'] == 'notyettreated'
    assert np.isfinite(result.results['att']).all()
    with pytest.raises(ValueError, match='never-treated'):
        fit.fit_callaway_santanna_estimator(create_staggered_did_specification(data, control_cols=['x']))


def test_cs_rejects_unsupported_covariates_before_backend_can_silently_drop_them():
    data = staggered_panel()
    data['category'] = 'a'
    spec = create_staggered_did_specification(data, control_cols=['category'])
    with pytest.raises(ValueError, match='numeric columns'):
        fit.fit_callaway_santanna_estimator(spec)


@pytest.fixture(scope='module', params=[
    fit.fit_callaway_santanna_estimator,
    fit.fit_callaway_santanna_nyt_estimator,
])
def cs_reserved_name_reference(request):
    wrapper = request.param
    reference = wrapper(create_staggered_did_specification(
        staggered_panel(), control_cols=['x'])).model['att_gt_object']
    return wrapper, reference


@pytest.mark.parametrize('name', [
    'rowid', 'w', 'w1', 'C', 'G_m', 'y_main', 'y0', 'y1', 'dy',
    'Intercept', 'never_treated',
])
def test_cs_reserved_covariate_names_preserve_adjustment(cs_reserved_name_reference, name):
    wrapper, reference = cs_reserved_name_reference
    data = staggered_panel().rename(columns={'x': name})
    # Even the default internal alias can already exist in user data.
    data['__pyautocausal_cs_0'] = -123.
    spec = create_staggered_did_specification(data, control_cols=[name])
    original = spec.data.copy(deep=True)
    fitted = wrapper(spec)
    result = fitted.model['att_gt_object']
    alias = fitted.model['column_mapping'][name]
    assert alias not in data.columns
    assert result.dp['xformla'] == f'y ~ {alias}'
    np.testing.assert_allclose(result.results['att'], reference.results['att'], atol=1e-12)
    np.testing.assert_allclose(result.results['se'], reference.results['se'], atol=1e-12)
    np.testing.assert_allclose(result.dp['data'][alias], data[name])
    pd.testing.assert_frame_equal(spec.data, original)
    assert fitted.control_cols == [name]
    assert spec.model is None


def assert_global_random_state_unchanged(before):
    after = np.random.get_state()
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]


@pytest.mark.parametrize('wrapper', [
    fit.fit_callaway_santanna_estimator,
    fit.fit_callaway_santanna_nyt_estimator,
])
def test_cs_bootstrap_is_seeded_for_fit_and_every_aggregation(wrapper):
    spec = create_staggered_did_specification(staggered_panel(), control_cols=['x'])
    before = np.random.get_state()
    default = wrapper(spec).model
    repeated = wrapper(spec, random_state=42).model
    other_seed = wrapper(spec, random_state=91).model
    assert_global_random_state_unchanged(before)
    assert default['random_state'] == repeated['random_state'] == 42
    assert other_seed['random_state'] == 91
    assert default['bootstrap_iterations'] == 1000

    for key in ('att', 'se', 'c', 'l_se', 'u_se'):
        np.testing.assert_array_equal(
            default['att_gt_object'].results[key], repeated['att_gt_object'].results[key])
    np.testing.assert_array_equal(
        default['att_gt_object'].results['att'], other_seed['att_gt_object'].results['att'])
    assert not np.array_equal(
        default['att_gt_object'].results['se'], other_seed['att_gt_object'].results['se'])

    for aggregation in ('overall_effect', 'dynamic_effects', 'group_effects'):
        first, again, different = (model[aggregation].atte
                                   for model in (default, repeated, other_seed))
        for key in ('overall_att', 'overall_se', 'att_egt', 'se_egt', 'crit_val_egt'):
            if first[key] is None:
                assert again[key] is None
            else:
                np.testing.assert_array_equal(first[key], again[key])
        np.testing.assert_array_equal(first['overall_att'], different['overall_att'])
        assert not np.array_equal(first['overall_se'], different['overall_se'])
    assert spec.model is None


@pytest.mark.parametrize('parallel', [False, True])
def test_multiplier_bootstrap_local_stream_is_reproducible(parallel):
    from pyautocausal.pipelines.library.csdid.utils.mboot import run_multiplier_bootstrap

    influences = np.random.default_rng(83).normal(size=(2501, 2))
    before = np.random.get_state()
    first = run_multiplier_bootstrap(influences, 12, pl=parallel, cores=2, rng=17)
    repeated = run_multiplier_bootstrap(influences, 12, pl=parallel, cores=2, rng=17)
    assert_global_random_state_unchanged(before)
    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (12, 2)
    if not parallel:
        # The change only localizes randomness: draws still use the same
        # Rademacher multiplier mean for each influence-function column.
        rng = np.random.default_rng(17)
        expected = np.array([
            np.mean(influences * rng.choice([1, -1], size=(len(influences), 1)), axis=0)
            for _ in range(12)
        ])
        np.testing.assert_array_equal(first, expected)
