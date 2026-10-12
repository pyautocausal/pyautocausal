import json
import sys
import nbformat
import numpy as np
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.run_context import RunContext
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.persistence.notebook_runner import run_notebook_and_create_html
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.estimators import fit_ols


def treatment_estimate(fit) -> float:
    return float(fit.model.params['treat'])


def _execute_export(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, str(path), 'exec'), namespace)
    return notebook, namespace


def test_exported_public_estimate_matches_run_with_explicit_controls(tmp_path):
    rng = np.random.default_rng(41)
    treatment = np.tile([0, 1], 50)
    x = rng.normal(size=100) + treatment
    frame = pd.DataFrame({'treat': treatment, 'x': x,
                          'y': 3 + 10 * treatment + 2 * x,
                          'id_unit': np.arange(100)})
    data_path = tmp_path / 'data.csv'
    frame.to_csv(data_path, index=False)
    context = RunContext(metadata={'control_cols': ['x']})
    graph = ExecutableGraph().configure_runtime(output_path=tmp_path, run_context=context)
    graph.create_input_node('data', pd.DataFrame)
    graph.create_node('specification', create_cross_sectional_specification.get_function(), ['data'])
    graph.create_node('fit', fit_ols.get_function(), ['specification'])
    graph.create_node('estimate', treatment_estimate, ['fit'], save_node=True)
    graph.fit(data=frame)
    path = tmp_path / 'analysis.ipynb'
    NotebookExporter(graph).export_notebook(path, str(data_path), 'pd.read_csv')
    notebook, namespace = _execute_export(path)
    assert abs(namespace['estimate_output'] - 10) < 1e-10
    assert abs(namespace['estimate_output'] - graph.get('estimate').get_result_value()) < 1e-10
    report = notebook.metadata.pyautocausal
    assert report['nodes']['specification']['analysis']['control_cols'] == ['x']
    assert report['nodes']['specification']['analysis']['rows'] == 100
    assert report['run_context']['control_cols'] == ['x']
    assert json.loads((tmp_path / 'estimate.json').read_text()) == graph.get('estimate').get_result_value()


def test_exported_join_uses_only_completed_alternative(tmp_path):
    def first():
        return 12
    def second():
        return 50
    def join(number, multiplier=1):
        return number * multiplier
    graph = ExecutableGraph().configure_runtime(run_context=RunContext({'multiplier': 3}))
    graph.create_node('a', first).create_node('b', second)
    graph.get('b').mark_passed()
    graph.create_node('result', join, {'number': ['a', 'b']})
    graph.execute_graph()
    path = tmp_path / 'join.ipynb'
    NotebookExporter(graph).export_notebook(path)
    notebook, namespace = _execute_export(path)
    assert namespace['result_output'] == 36
    assert notebook.metadata.pyautocausal['nodes']['b']['state'] == 'passed'
    assert all('b_output' not in cell.source for cell in notebook.cells if cell.cell_type == 'code')


def test_current_interpreter_kernel_ignores_stale_user_python3(tmp_path, monkeypatch):
    kernels = tmp_path / 'jupyter' / 'kernels' / 'python3'
    kernels.mkdir(parents=True)
    kernel = kernels / 'kernel.json'
    stale = {'argv': ['/nonexistent/stale/python', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
             'display_name': 'stale', 'language': 'python'}
    kernel.write_text(json.dumps(stale))
    monkeypatch.setenv('JUPYTER_PATH', str(tmp_path / 'jupyter'))
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        f"import sys\nassert sys.executable == {sys.executable!r}\nprint('correct-current-interpreter')")])
    path = tmp_path / 'interpreter.ipynb'
    nbformat.write(notebook, path)
    html = run_notebook_and_create_html(path)
    assert 'correct-current-interpreter' in html.read_text()
    assert json.loads(kernel.read_text()) == stale


def test_compact_export_keeps_execution_diagnostics_and_descriptions(tmp_path):
    def meaning() -> int:
        """A documented result for the reader."""
        return 42
    graph = ExecutableGraph().create_node('answer', meaning)
    graph.execute_graph()
    path = tmp_path / 'compact.ipynb'
    NotebookExporter(graph).export_notebook(path, compact=True)
    notebook, namespace = _execute_export(path)
    assert namespace['answer_output'] == 42
    assert any(cell.metadata.get('jupyter', {}).get('source_hidden') for cell in notebook.cells)
    assert any('A documented result for the reader.' in cell.source
               for cell in notebook.cells if cell.cell_type == 'markdown')
    assert notebook.metadata.pyautocausal['nodes']['answer']['state'] == 'completed'
    yaml_path = graph.to_yaml(tmp_path / 'graph.yaml')
    restored = ExecutableGraph.from_yaml(yaml_path)
    assert restored.get('answer').node_description == graph.get('answer').node_description


def test_notebook_missing_extra_has_install_instruction(tmp_path, monkeypatch):
    import pyautocausal.persistence.notebook_runner as runner
    import pytest
    path = tmp_path / 'example.ipynb'
    nbformat.write(nbformat.v4.new_notebook(), path)
    monkeypatch.setattr(runner, 'find_spec', lambda package: None)
    with pytest.raises(ImportError, match=r"pyautocausal\[notebooks\]"):
        runner.run_notebook_and_create_html(path)
    with pytest.raises(ImportError, match=r"pyautocausal\[notebooks\]"):
        runner.convert_notebook_to_html(path)


def test_merged_graph_notebook_replays_passthrough_and_lambda(tmp_path):
    def source() -> int:
        return 3
    left = ExecutableGraph().create_node('source', source)
    right = ExecutableGraph().create_input_node('number', int)
    right.create_node('answer', lambda number: number * 2, ['number'])
    left.merge_with(right, left.get('source') >> right.get('number'))
    left.execute_graph()
    path = tmp_path / 'merged.ipynb'
    NotebookExporter(left).export_notebook(path)
    _, namespace = _execute_export(path)
    assert namespace['answer_output'] == 6


def test_notebook_replays_explicit_binding_from_decision(tmp_path):
    def source():
        return 3
    def positive(number):
        return number > 0
    def answer(value):
        return value * 2
    graph = ExecutableGraph().create_node('source', source)
    graph.create_decision_node('decision', positive, ['source'])
    graph.create_node('answer', answer, {'value': 'decision'})
    graph.when_true('decision', 'answer')
    graph.execute_graph()
    path = tmp_path / 'decision.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    assert namespace['answer_output'] == graph.get('answer').get_result_value() == 6


def test_notebook_preserves_bindings_through_nested_decisions(tmp_path):
    def first():
        return 3
    def second():
        return 5
    def positive(left, right):
        return left < right
    def answer(left, right):
        return left * 10 + right
    graph = ExecutableGraph().create_node('a', first).create_node('b', second)
    graph.create_decision_node('check', positive, {'left': 'a', 'right': 'b'})
    graph.create_decision_node('check_again', positive, ['check'])
    graph.create_node('answer', answer, ['check_again'])
    graph.when_true('check', 'check_again').when_true('check_again', 'answer')
    graph.execute_graph()
    path = tmp_path / 'nested_decisions.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    assert namespace['answer_output'] == graph.get('answer').get_result_value() == 35


def test_notebook_named_optional_binding_preserves_required_context(tmp_path):
    def source():
        return 3
    def result(data, multiplier=2):
        return data * multiplier
    graph = ExecutableGraph().configure_runtime(run_context=RunContext({'data': 10}))
    graph.create_node('factor', source).create_node('result', result, {'multiplier': 'factor'})
    graph.execute_graph()
    path = tmp_path / 'binding.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    assert namespace['result_output'] == graph.get('result').get_result_value() == 30


def test_notebook_preserves_array_shape_dtype_and_numerical_behavior(tmp_path):
    def calculate(coefficients):
        return coefficients * 2
    values = np.array([[1, 2], [3, 4]], dtype=np.int16)
    graph = ExecutableGraph().configure_runtime(run_context=RunContext({'coefficients': values}))
    graph.create_node('result', calculate).execute_graph()
    path = tmp_path / 'array.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    result = namespace['result_output']
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, graph.get('result').get_result_value())


def test_notebook_preserves_datetime_scalar_and_nested_tuple_context(tmp_path):
    import datetime
    from zoneinfo import ZoneInfo
    def identity(values):
        return values
    values = (
        pd.Timestamp('2024-03-30 12:00:00.123456789', tz='Europe/Berlin'),
        pd.Timedelta('2 days 3ns'),
        datetime.datetime(2024, 10, 27, 2, 30, tzinfo=ZoneInfo('Europe/Berlin'), fold=1),
        datetime.date(2024, 2, 29),
        datetime.timedelta(days=2, microseconds=3),
        np.datetime64('2024-01-01T12:23:34.123456789'),
        np.timedelta64(13, 'ns'),
        np.int16(7),
        {'nested': (1, (2,))},
    )
    graph = ExecutableGraph().configure_runtime(run_context=RunContext({'values': values}))
    graph.create_node('result', identity).execute_graph()
    path = tmp_path / 'typed-context.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    result = namespace['result_output']
    assert type(result) is tuple
    assert result == values
    assert [type(value) for value in result] == [type(value) for value in values]
    assert str(result[0].tz) == 'Europe/Berlin'
    assert result[2].fold == 1
    assert result[5].dtype == values[5].dtype
    assert result[6].dtype == values[6].dtype


def test_notebook_rejects_unsupported_consumed_context_without_changing_graph(tmp_path):
    import pytest
    def identity(value):
        return value
    class Settings:
        pass
    context = RunContext({'value': Settings()})
    graph = ExecutableGraph().configure_runtime(run_context=context)
    graph.create_node('result', identity).execute_graph()
    with pytest.raises(ValueError, match="Cannot export run-context value 'value'.*Settings"):
        NotebookExporter(graph).export_notebook(tmp_path / 'unsupported.ipynb')
    assert graph.run_context is context


def test_notebook_ignores_context_values_not_used_by_the_executed_functions(tmp_path):
    def answer():
        return 42
    graph = ExecutableGraph().configure_runtime(run_context=RunContext({'unused': object()}))
    graph.create_node('result', answer).execute_graph()
    path = tmp_path / 'unused-context.ipynb'
    NotebookExporter(graph).export_notebook(path)
    _, namespace = _execute_export(path)
    assert namespace['result_output'] == 42


def test_notebook_rejects_ambiguous_single_loader_for_multiple_inputs(tmp_path):
    import pytest
    def difference(treated, control):
        return treated.y.mean() - control.y.mean()
    graph = ExecutableGraph().create_input_node('treated', pd.DataFrame)
    graph.create_input_node('control', pd.DataFrame)
    graph.create_node('result', difference, {'treated': 'treated', 'control': 'control'})
    graph.fit(treated=pd.DataFrame({'y': [5.]}), control=pd.DataFrame({'y': [2.]}))
    path = tmp_path / 'multiple-inputs.ipynb'
    with pytest.raises(ValueError, match='single data_path/loading_function.*multiple inputs'):
        NotebookExporter(graph).export_notebook(path, 'treated.csv', 'pd.read_csv')
    assert not path.exists()
    # Manual export still gives each input its own editable cell.
    NotebookExporter(graph).export_notebook(path)
    notebook = nbformat.read(path, as_version=4)
    assert any('treated_output = None' in cell.source for cell in notebook.cells)
    assert any('control_output = None' in cell.source for cell in notebook.cells)



def test_notebook_preserves_pytz_datetime_instants_and_ambiguous_offsets(tmp_path):
    import datetime
    import pytz

    def inspect_datetime(when):
        return (when.isoformat(), when.timestamp(), when.utcoffset().total_seconds(), when.fold)

    zone = pytz.timezone('America/New_York')
    values = [
        zone.localize(datetime.datetime(2024, 7, 1, 12, 13, 14, 123456)),
        zone.localize(datetime.datetime(2024, 1, 1, 12)),
        zone.localize(datetime.datetime(2024, 11, 3, 1, 30), is_dst=True),
        zone.localize(datetime.datetime(2024, 11, 3, 1, 30), is_dst=False),
    ]
    for index, value in enumerate(values):
        graph = ExecutableGraph().configure_runtime(run_context=RunContext({'when': value}))
        graph.create_node('result', inspect_datetime).execute_graph()
        path = tmp_path / f'pytz-{index}.ipynb'
        NotebookExporter(graph).export_notebook(path)
        _, namespace = _execute_export(path)
        assert namespace['result_output'] == graph.get('result').get_result_value()


def test_notebook_rejects_nonexistent_local_datetimes_instead_of_changing_them(tmp_path):
    import datetime
    import pytz
    import pytest
    from zoneinfo import ZoneInfo

    def identity(when):
        return when

    wall_time = datetime.datetime(2024, 3, 10, 2, 30)
    values = [
        pytz.timezone('America/New_York').localize(wall_time, is_dst=False),
        wall_time.replace(tzinfo=ZoneInfo('America/New_York')),
    ]
    for value in values:
        context = RunContext({'when': value})
        graph = ExecutableGraph().configure_runtime(run_context=context)
        graph.create_node('result', identity).execute_graph()
        with pytest.raises(ValueError, match='Nonexistent or inconsistent local datetime'):
            NotebookExporter(graph).export_notebook(tmp_path / 'nonexistent.ipynb')
        assert graph.run_context is context
