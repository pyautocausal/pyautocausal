"""Degenerate empirical inference must retain effects without false precision."""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library import estimators as fit
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.csdid.utils.mboot import mboot
from pyautocausal.pipelines.library.plots import callaway_santanna_comprehensive_results
from pyautocausal.pipelines.library.specifications import create_staggered_did_specification


def deterministic_panel(single_cohort=False):
    rows = []
    for unit in range(120):
        cohort = 3 if unit < 40 else (3 if single_cohort else 4) if unit < 80 else 0
        for time in range(1, 6):
            treat = int(cohort > 0 and time >= cohort)
            rows.append({'id_unit': unit, 't': time, 'treat': treat,
                         'y': unit * .01 + time + 2 * treat})
    return pd.DataFrame(rows)


@pytest.mark.parametrize('wrapper', [fit.fit_callaway_santanna_estimator,
                                     fit.fit_callaway_santanna_nyt_estimator])
@pytest.mark.parametrize('single_cohort', [False, True])
def test_deterministic_effect_keeps_point_estimates_and_marks_inference_unavailable(
        wrapper, single_cohort, capsys):
    fitted_spec = wrapper(create_staggered_did_specification(
        deterministic_panel(single_cohort), control_cols=[]))
    model = fitted_spec.model
    attgt = model['att_gt_object']
    post = np.asarray(attgt.results['year']) >= np.asarray(attgt.results['group'])
    np.testing.assert_allclose(np.asarray(attgt.results['att'])[post], 2, atol=1e-12)
    assert attgt.inference['status'] == 'unavailable'
    assert np.isnan(attgt.results['se']).all()
    assert np.isnan(attgt.results['l_se']).all()
    assert np.isnan(attgt.results['u_se']).all()
    assert not np.asarray(attgt.results['sig']).any()
    assert 'Inference unavailable' in capsys.readouterr().out

    # The public run report must preserve the unavailable-inference contract
    # without leaking NaN values into JSON or losing the known point estimate.
    graph = ExecutableGraph()
    graph.create_input_node('fitted', type(fitted_spec))
    graph.fit(fitted=fitted_spec)
    report = json.loads(json.dumps(graph.execution_report(), allow_nan=False))
    analysis = report['nodes']['fitted']['analysis']
    for key in ['group_time_effects', 'overall_effect', 'dynamic_effects', 'group_effects']:
        assert analysis['inference'][key]['status'] == 'unavailable'
    assert analysis['overall_effect']['overall_att'] == pytest.approx(2)
    assert analysis['overall_effect']['overall_se'] is None
    assert analysis['overall_effect']['inference']['status'] == 'unavailable'
    formatted = callaway_santanna_comprehensive_results(fitted_spec)
    assert '2.0000' in formatted
    assert '95% CI:     unavailable' in formatted
    assert 'Inference unavailable:' in formatted
    assert 'nan' not in formatted.lower()
    assert 'Significant:' not in formatted

    figures = [attgt.plot_attgt()]
    for key in ['overall_effect', 'dynamic_effects', 'group_effects']:
        aggregation = model[key]
        assert aggregation.atte['overall_att'] == pytest.approx(2)
        assert np.isnan(aggregation.atte['overall_se']).all()
        assert aggregation.atte['inference']['status'] == 'unavailable'
        if key != 'overall_effect':
            assert np.isnan(aggregation.atte['se_egt']).all()
            figures.append(aggregation.plot_aggte())
    for figure in figures:
        figure.canvas.draw()
        assert any('intervals unavailable' in text.get_text()
                   for axis in figure.axes for text in axis.texts)
        plt.close(figure)


def bootstrap_params(n):
    return {'data': pd.DataFrame({'id': np.arange(n), 'time': 1}),
            'idname': 'id', 'tname': 'time', 'clustervars': None,
            'biters': 1000, 'alp': .05, 'panel': True,
            'true_repeated_cross_sections': False}


def test_partial_degeneracy_retains_valid_dimensions_without_zero_width_bands():
    varying = np.random.default_rng(83).normal(size=(100, 1))
    influences = np.column_stack([np.zeros(100), varying, np.full(100, np.nan)])
    params = bootstrap_params(100)
    partial = mboot(influences, params, rng=42)
    reference = mboot(varying, params, rng=42)
    assert partial['inference']['status'] == 'partial'
    assert np.isnan(partial['se'][[0, 2]]).all()
    assert partial['se'][1] == pytest.approx(reference['se'][0], abs=1e-14)
    assert partial['crit_val'] == pytest.approx(reference['crit_val'], abs=1e-14)
    assert np.isfinite(partial['crit_val'])


def test_zero_bootstrap_iqr_is_unavailable_instead_of_dividing_by_zero(monkeypatch):
    from importlib import import_module
    module = import_module('pyautocausal.pipelines.library.csdid.utils.mboot')
    monkeypatch.setattr(module, 'run_multiplier_bootstrap',
                        lambda influences, biters, *args, **kwargs: np.ones((biters, 1)))
    result = mboot(np.ones((100, 1)), bootstrap_params(100), rng=42)
    assert result['inference']['status'] == 'unavailable'
    assert np.isnan(result['se']).all()
    assert np.isnan(result['crit_val'])


def test_partial_inference_formatter_keeps_supported_overall_interval():
    data = deterministic_panel()
    noise = np.random.default_rng(21).normal(size=data.id_unit.nunique())
    data.y += noise[data.id_unit] * (data.t == 5)
    fitted_spec = fit.fit_callaway_santanna_estimator(
        create_staggered_did_specification(data, control_cols=[]))
    model = fitted_spec.model
    assert model['att_gt_object'].inference['status'] == 'partial'
    assert np.isfinite(model['overall_effect'].atte['overall_se']).all()
    original_summary = model['att_gt_object'].summary2.copy(deep=True)
    formatted = callaway_santanna_comprehensive_results(fitted_spec)
    assert 'Group-time inference partial:' in formatted
    assert '95% CI:     [' in formatted
    assert 'unavailable' in formatted
    assert 'nan' not in formatted.lower()
    pd.testing.assert_frame_equal(model['att_gt_object'].summary2, original_summary)
