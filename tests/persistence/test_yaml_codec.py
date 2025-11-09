from pathlib import Path
import os
import sys

import pytest

from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.persistence.notebook_export import NotebookExporter
import nbformat


def test_yaml_roundtrip(tmp_path):
    """Serialize a graph to YAML and load it back, validating integrity."""
    graph = simple_graph()

    yaml_path = tmp_path / "graph.yml"
    graph.to_yaml(yaml_path)

    loaded = ExecutableGraph.from_yaml(yaml_path)

    # Runtime configuration is not serialized; configure for the tmp dir
    loaded.configure_runtime(output_path=tmp_path / "loaded_output")

    # Structural checks
    assert len(graph.nodes()) == len(loaded.nodes())
    assert {n.name for n in graph.nodes()} == {n.name for n in loaded.nodes()}
    assert {
        (u.name, v.name) for u, v in graph.edges()
    } == {(u.name, v.name) for u, v in loaded.edges()}

    # Ensure the loaded graph can execute successfully
    df = generate_mock_data(n_units=100, n_periods=2, n_treated=20)
    loaded.fit(df=df)

    # All nodes should have completed or passed after execution
    for node in loaded.nodes():
        assert node.state.is_terminal()  # type: ignore[attr-defined] 


def test_yaml_roundtrip_notebook_export(tmp_path):
    """After YAML round-trip, ensure we can export a notebook successfully."""
    graph = simple_graph()

    # Save + load via YAML
    yaml_path = tmp_path / "graph.yml"
    graph.to_yaml(yaml_path)
    loaded = ExecutableGraph.from_yaml(yaml_path)
    loaded.configure_runtime(output_path=tmp_path / "loaded_output")

    # Execute so that nodes are completed (needed for exporter)
    df = generate_mock_data(n_units=100, n_periods=2, n_treated=20)
    loaded.fit(df=df)

    # Export notebook
    nb_path = tmp_path / "roundtrip.ipynb"
    exporter = NotebookExporter(loaded)
    exporter.export_notebook(str(nb_path))

    # Basic sanity check – notebook file exists and is parseable
    assert nb_path.exists()
    with nb_path.open() as fh:
        nb = nbformat.read(fh, as_version=4)
    # header markdown cell expected
    assert nb.cells and nb.cells[0].cell_type == "markdown"


def test_yaml_portability_for_notebook_export(tmp_path):
    """
    Simulates a "foreign" environment to ensure that a graph serialized
    on one machine can be deserialized and exported to a notebook on another
    where the original absolute file paths are not available.

    This directly tests for the "OSError: source code not available" bug.
    """
    # 1. Serialize the graph in the "local" environment
    local_graph_path = tmp_path / "local_graph"
    graph = simple_graph()
    yaml_path = tmp_path / "portable_graph.yml"
    graph.to_yaml(yaml_path)

    # Execute the graph so that notebook export has completed nodes
    df = generate_mock_data(n_units=10, n_periods=2, n_treated=2)
    graph.fit(df=df)

    # 2. Simulate the "foreign" environment
    original_cwd = Path.cwd()
    foreign_dir = tmp_path / "foreign_environment"
    foreign_dir.mkdir()
    os.chdir(foreign_dir)

    # Add project root to sys.path to simulate package installation
    project_root = Path(original_cwd).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        # 3. Deserialize and export in the "foreign" environment
        loaded_graph = ExecutableGraph.from_yaml(yaml_path)
        
        # This step would fail with an OSError if paths were not portable
        exporter = NotebookExporter(loaded_graph)
        nb_path = foreign_dir / "exported_notebook.ipynb"
        
        # The key assertion: does this raise an OSError?
        exporter.export_notebook(str(nb_path))
        
        # Verify notebook was created
        assert nb_path.exists()
        with nb_path.open() as fh:
            nb = nbformat.read(fh, as_version=4)
        assert len(nb.cells) > 0

    finally:
        # 4. Clean up the environment
        os.chdir(original_cwd)
        sys.path.pop(0) 