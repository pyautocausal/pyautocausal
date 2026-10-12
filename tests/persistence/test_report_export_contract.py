"""End-to-end checks for concise, truthful public report exports."""
import json

import nbformat
import pandas as pd
import pytest

from pyautocausal import ExecutableGraph, export_outputs
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.persistence.notebook_runner import convert_notebook_to_html


def test_compact_public_export_hides_code_but_executes_it(tmp_path):
    def average(data) -> float:
        # COMPACT_IMPLEMENTATION_MARKER
        return float(data.y.mean())

    data = pd.DataFrame({'y': [2., 4.]})
    graph = ExecutableGraph().create_input_node('data', pd.DataFrame)
    graph.create_node('average', average, ['data'], display_function=print).fit(data=data)
    export_outputs(graph, df=data, output_path=tmp_path, compact=True)

    notebook = nbformat.read(tmp_path / 'pipeline_execution.ipynb', as_version=4)
    html = (tmp_path / 'pipeline_execution.html').read_text()
    assert any('COMPACT_IMPLEMENTATION_MARKER' in cell.source for cell in notebook.cells)
    assert 'COMPACT_IMPLEMENTATION_MARKER' not in html
    assert '<pre>3.0\n</pre>' in html
    assert 'Execution diagnostics and provenance' in html
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['export']['state'] == 'completed'
    assert report['nodes']['average']['state'] == 'completed'


def test_static_html_conversion_respects_hidden_input_tags(tmp_path):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'STATIC_IMPLEMENTATION_MARKER = 7',
        metadata={'tags': ['hide-input']},
        outputs=[nbformat.v4.new_output('stream', name='stdout', text='visible saved result')],
    )])
    path = tmp_path / 'saved.ipynb'
    nbformat.write(notebook, path)
    html = convert_notebook_to_html(path).read_text()
    assert 'STATIC_IMPLEMENTATION_MARKER' not in html
    assert 'visible saved result' in html


def test_dataframe_report_replay_preserves_labels_types_and_index(tmp_path):
    def count_units(data) -> int:
        assert str(data.observed.dtype) == 'datetime64[ns, Europe/Berlin]'
        assert str(data.group.dtype) == 'category'
        assert data.index.name == 'original_row'
        assert data.index.tolist() == [10, 20, 30]
        return int(data.id_unit.nunique())

    data = pd.DataFrame({
        'id_unit': ['001', '01', '1'],
        'observed': pd.date_range('2024-01-01', periods=3, tz='Europe/Berlin').as_unit('ns'),
        'group': pd.Categorical(['a', 'b', 'a']),
    }, index=pd.Index([10, 20, 30], name='original_row'))
    graph = ExecutableGraph().create_input_node('data', pd.DataFrame)
    graph.create_node('units', count_units, ['data'], display_function=print).fit(data=data)
    export_outputs(graph, df=data, output_path=tmp_path)
    pd.testing.assert_frame_equal(data, pd.read_parquet(tmp_path / 'data.csv.parquet'))
    assert '<pre>3\n</pre>' in (tmp_path / 'pipeline_execution.html').read_text()
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['export']['data']['loader'] == 'pd.read_parquet'
    assert report['export']['state'] == 'completed'


def test_csv_loader_rejects_changed_input_instead_of_reporting_success(tmp_path):
    data = pd.DataFrame({'id_unit': ['001', '01', '1'], 'y': [1., 2., 3.]})
    path = tmp_path / 'source.csv'
    data.to_csv(path, index=False)
    graph = ExecutableGraph().create_input_node('data', pd.DataFrame).fit(data=data)
    with pytest.raises(ValueError, match='CSV replay would change.*pass df='):
        export_outputs(graph, data_path=path, output_path=tmp_path)
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['export']['state'] == 'failed'
    assert report['export']['stage'] == 'data'


@pytest.mark.parametrize('stage', ['notebook', 'html'])
def test_export_failure_is_raised_and_recorded(tmp_path, monkeypatch, stage):
    data = pd.DataFrame({'y': [1., 2.]})
    graph = ExecutableGraph().create_input_node('data', pd.DataFrame).fit(data=data)
    previous_html = tmp_path / 'pipeline_execution.html'
    previous_html.write_text('previous valid report')

    def fail(*args, **kwargs):
        raise RuntimeError('deliberate export failure')

    method = 'export_notebook' if stage == 'notebook' else 'export_and_run_to_html'
    monkeypatch.setattr(NotebookExporter, method, fail)
    with pytest.raises(RuntimeError, match='deliberate export failure'):
        export_outputs(graph, df=data, output_path=tmp_path)
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['export']['state'] == 'failed'
    assert report['export']['stage'] == stage
    assert report['export']['error'] == {
        'type': 'RuntimeError', 'message': 'deliberate export failure',
    }
    assert report['nodes']['data']['state'] == 'completed'
    assert previous_html.read_text() == 'previous valid report'
