"""Scalar NumPy contexts must survive execution, diagnostics and notebook replay."""
import json

import nbformat
import numpy as np
import pytest

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.run_context import RunContext
from pyautocausal.persistence.notebook_export import NotebookExporter


def execute_cells(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, str(path), 'exec'), namespace)
    return notebook, namespace


def test_scalar_array_context_is_saved_and_replayed_without_losing_its_shape(tmp_path):
    def double(scalar) -> float:
        assert scalar.ndim == 0
        return float(scalar * 2)

    graph = ExecutableGraph().configure_runtime(
        output_path=tmp_path, run_context=RunContext({'scalar': np.array(3.)}))
    graph.create_node('result', double, save_node=True).execute_graph()
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['run_context']['scalar'] == 3.
    assert json.loads((tmp_path / 'result.json').read_text()) == 6.

    path = tmp_path / 'scalar.ipynb'
    NotebookExporter(graph).export_notebook(path)
    notebook, namespace = execute_cells(path)
    assert namespace['result_output'] == graph.get('result').get_result_value() == 6.
    assert notebook.metadata.pyautocausal['run_context']['scalar'] == 3.


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), float('-inf')])
def test_unused_nonfinite_scalar_array_diagnostics_become_null(tmp_path, nonfinite):
    def answer() -> int:
        return 42

    context = RunContext({'diagnostics': {'value': np.array(nonfinite)}})
    graph = ExecutableGraph().configure_runtime(output_path=tmp_path, run_context=context)
    graph.create_node('result', answer, save_node=True).execute_graph()
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['run_context']['diagnostics']['value'] is None
    assert json.loads((tmp_path / 'result.json').read_text()) == 42

    path = tmp_path / 'diagnostics.ipynb'
    NotebookExporter(graph).export_notebook(path)
    notebook, namespace = execute_cells(path)
    assert namespace['result_output'] == 42
    assert notebook.metadata.pyautocausal['run_context']['diagnostics']['value'] is None
