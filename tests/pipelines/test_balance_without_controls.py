import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyautocausal.pipelines.example_graph import create_panel_graph
from pyautocausal.pipelines.library.balancing import compute_balance_tests
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.output import write_balance_summary_table
from pyautocausal.pipelines.library.plots import plot_balance_coefficients


def test_explicit_no_controls_does_not_add_unused_numeric_variables():
    frame = pd.DataFrame({'y': [1., 3., 2., 4.], 'treat': [0, 1, 0, 1],
                          'unused_covariate': [0., 2., 1., 3.]})
    spec = create_cross_sectional_specification(frame, control_cols=[])
    compute_balance_tests(spec)
    assert spec.balance_stats.empty
    assert 'covariate' in spec.balance_stats.columns
    assert spec.balance_diagnostics['control_cols'] == []
    assert 'No control covariates were specified' in write_balance_summary_table(spec)
    figure = plot_balance_coefficients(spec)
    assert not figure.axes[0].lines
    assert 'unavailable' in figure.axes[0].get_title()
    plt.close(figure)


def test_public_panel_graph_saves_no_controls_diagnostic_plot(tmp_path):
    rows = []
    rng = np.random.default_rng(12)
    for unit in range(16):
        for time in range(2):
            treat = int(unit < 8 and time == 1)
            rows.append({'id_unit': unit, 't': time, 'treat': treat,
                         'y': float(unit + time + 3 * treat + rng.normal(scale=.1))})
    graph = create_panel_graph(tmp_path).fit(df=pd.DataFrame(rows))
    spec = graph.get('did_spec_balance').get_result_value()
    assert spec.control_cols == []
    assert spec.balance_stats.empty
    assert graph.is_execution_finished()
    figure = graph.get('did_spec_balance_plot').get_result_value()
    assert not figure.axes[0].lines
    assert 'No control covariates were specified' in figure.axes[0].texts[0].get_text()
    assert (tmp_path / 'plots' / 'did_balance_plot.png').exists()
    assert 'balance unavailable' in (tmp_path / 'text' / 'did_balance_results.txt').read_text()
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['nodes']['did_spec_balance']['analysis']['balance_diagnostics']['status'] == 'unavailable'
    plt.close(figure)


def test_synthetic_control_scales_predictors_across_units():
    from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods.main import DataProcessor
    treated = np.array([[2.], [7.]])
    controls = np.array([[1., 3.], [7., 7.]])
    scaled_treated, scaled_controls = DataProcessor()._rescale_covariate_variance(treated, controls, 2)
    combined = np.concatenate([scaled_treated, scaled_controls], axis=1)
    assert np.isfinite(combined).all()
    np.testing.assert_allclose(combined[0].std(), 1.)
    np.testing.assert_allclose(combined[1], [7., 7., 7.])
    # Outcome-only models have one predictor; scaling must still be finite.
    one_treated, one_controls = DataProcessor()._rescale_covariate_variance(treated[:1], controls[:1], 1)
    assert np.isfinite(one_treated).all() and np.isfinite(one_controls).all()
    np.testing.assert_allclose(np.concatenate([one_treated, one_controls], axis=1).std(), 1.)


def test_synthetic_control_objective_compares_matching_predictors():
    from types import SimpleNamespace
    from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods.optimize import Optimize
    optimizer = Optimize()
    optimizer.method = 'SC'
    treated = np.array([[.25], [125.]])
    controls = np.array([[0., 1.], [100., 200.]])
    outcomes = np.array([[0., 1.]])
    data = SimpleNamespace(min_loss=np.inf, control_outcome_all=outcomes,
                           control_covariates=controls)
    loss = optimizer.total_loss(np.array([.5, .5]), np.array([[.25]]), treated,
                                outcomes, controls, treated - controls, 0, False, data)
    # Both predictors are matched exactly by the same convex combination.
    # Broadcasting instead targets a mixture of unrelated predictor scales.
    np.testing.assert_allclose(data.w.ravel(), [.75, .25], atol=1e-6)
    np.testing.assert_allclose(controls @ data.w, treated, atol=1e-6)
    assert float(np.asarray(loss).item()) < 1e-12


@pytest.mark.parametrize('panel', ['original', 'pointwise', 'cumulative', 'in-time placebo'])
def test_synthetic_control_annotations_use_scalars_and_draw(panel):
    from types import SimpleNamespace
    from matplotlib.text import Annotation
    from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods.plot import Plot
    plotter = Plot()
    plotter.original_data = SimpleNamespace(
        dataset=pd.DataFrame({'t': [1, 2, 3, 4]}), time='t', outcome_var='y',
        synth_outcome=np.array([[1., 2., 2.5, 3.]]),
        treated_outcome=np.array([[1.], [2.]]),
        treated_outcome_all=np.array([[1.], [2.], [4.], [5.]]),
        treatment_period=3, periods_all=4, periods_pre_treatment=2,
        in_time_placebo_outcome=np.array([[1., 2., 3., 4.]]),
        placebo_treatment_period=2, placebo_periods_pre_treatment=1,
    )
    figure = plotter.plot([panel])
    figure.canvas.draw()
    annotations = [text for axis in figure.axes for text in axis.texts if isinstance(text, Annotation)]
    assert annotations
    assert all(isinstance(annotation.xy[1], float) for annotation in annotations)
    plt.close(figure)
