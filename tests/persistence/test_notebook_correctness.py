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
