import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.persistence.output_types import OutputType
from pyautocausal.orchestration.graph import ExecutableGraph


def test_png_saves_supplied_figure_not_current_figure(tmp_path):
    wanted, ax = plt.subplots()
    ax.plot([0, 1], [10, 0], color='red')
    other, other_ax = plt.subplots()
    other_ax.plot([0, 1], [0, 1], color='blue')
    expected = io.BytesIO()
    wanted.savefig(expected, format='png')
    LocalOutputHandler(tmp_path).save('figure', wanted, OutputType.PNG)
    assert (tmp_path / 'figure.png').read_bytes() == expected.getvalue()
    plt.close(wanted)
    plt.close(other)


@pytest.mark.parametrize('value', [3, 3.5, True, None, [1, False, None],
                                   {'values': np.array([1., 2.]), 'n': np.int64(2)}])
def test_json_supports_scalars_lists_and_numpy(tmp_path, value):
    LocalOutputHandler(tmp_path).save('result', value, OutputType.JSON)
    loaded = json.loads((tmp_path / 'result.json').read_text())
    if isinstance(value, dict):
        assert loaded == {'values': [1., 2.], 'n': 2}
    else:
        assert loaded == value


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {'v': np.float64('-inf')}])
def test_json_rejects_nonfinite_without_damaging_existing_result(tmp_path, value):
    output = tmp_path / 'result.json'
    output.write_text('{"previous": 1}')
    with pytest.raises(ValueError, match='Non-finite'):
        LocalOutputHandler(tmp_path).save('result', value, OutputType.JSON)
    assert json.loads(output.read_text()) == {'previous': 1}


@pytest.mark.parametrize('annotation, value', [(int, 3), (float, 2.5), (bool, True),
                                              (list[int], [1, 2]), (np.float64, np.float64(2.5))])
def test_inferred_json_output_runs_through_public_graph(tmp_path, annotation, value):
    def estimate():
        return value
    estimate.__annotations__['return'] = annotation
    graph = ExecutableGraph().configure_runtime(output_path=tmp_path)
    graph.create_node('estimate', estimate, save_node=True).execute_graph()
    assert json.loads((tmp_path / 'estimate.json').read_text()) == value


def test_uplift_summary_handles_unavailable_inference_and_real_cate_dict():
    from types import SimpleNamespace
    from pyautocausal.pipelines.library.output import write_uplift_summary
    spec = SimpleNamespace(model={}, ate_estimate=0.2, ate_ci=None,
                           cate_estimates={'t_learner': np.array([0.1, 0.3])},
                           experimental=True, inference={'status': 'unavailable'},
                           evaluation_metrics=None)
    summary = write_uplift_summary(spec)
    assert 'Mean CATE: 0.2000' in summary
    assert 'Confidence interval: unavailable' in summary
    assert 'Experimental estimator' in summary
    spec.cate_estimates = None
    summary = write_uplift_summary(spec)
    assert 'Conditional treatment effects: unavailable' in summary


def test_average_effect_only_cannot_produce_uplift_curve():
    from types import SimpleNamespace
    from pyautocausal.pipelines.library.plots import uplift_curve_plot_adaptive
    spec = SimpleNamespace(cate_estimates=None, ate_estimate=0.2)
    with pytest.raises(ValueError, match='Conditional treatment effects are unavailable'):
        uplift_curve_plot_adaptive(spec)


def test_explicit_output_format_defaults_filename_to_node_name(tmp_path):
    from pyautocausal.persistence.output_config import OutputConfig
    graph = ExecutableGraph().configure_runtime(output_path=tmp_path)
    graph.create_node('effect', lambda: 2, save_node=True,
                      output_config=OutputConfig(OutputType.JSON))
    graph.execute_graph()
    assert json.loads((tmp_path / 'effect.json').read_text()) == 2
    assert not (tmp_path / 'None.json').exists()


def test_json_preserves_timestamp_values_and_index(tmp_path):
    timestamp = pd.Timestamp('2026-10-08T12:34:56.123456789+02:00')
    data = pd.DataFrame({'observed_at': [timestamp]}, index=pd.DatetimeIndex([timestamp]))
    LocalOutputHandler(tmp_path).save('timestamps', data, OutputType.JSON)
    loaded = json.loads((tmp_path / 'timestamps.json').read_text())
    assert loaded == {'observed_at': {timestamp.isoformat(): timestamp.isoformat()}}


@pytest.mark.parametrize('data', [
    pd.DataFrame({'a': [1, 2]}, index=[0, 0]),
    pd.Series([1, 2], index=[0, 0]),
    pd.DataFrame([[1, 2]], columns=['a', 'a']),
    pd.DataFrame({'a': [1, 2]}, index=[0, '0']),
])
def test_json_rejects_ambiguous_pandas_labels_without_overwriting(tmp_path, data):
    output = tmp_path / 'result.json'
    output.write_text('{"previous": 1}')
    with pytest.raises(ValueError, match='unique|collide'):
        LocalOutputHandler(tmp_path).save('result', data, OutputType.JSON)
    assert json.loads(output.read_text()) == {'previous': 1}


@pytest.mark.parametrize('value', [pd.NaT, pd.NA, np.datetime64('NaT')])
def test_json_rejects_missing_pandas_scalars(tmp_path, value):
    with pytest.raises(ValueError, match='Missing pandas values'):
        LocalOutputHandler(tmp_path).save('missing', value, OutputType.JSON)
    assert not (tmp_path / 'missing.json').exists()
