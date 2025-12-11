from pathlib import Path

from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.mock_data import generate_mock_data


def test_pickle_roundtrip(tmp_path):
    """Build a graph, dump to pickle, load, and verify basic structure."""
    graph = simple_graph()
    
    file_path = tmp_path / "graph.yml"
    graph.to_yaml(file_path)

    loaded = ExecutableGraph.from_yaml(file_path)
    
    # Configure runtime for the loaded graph
    loaded.configure_runtime(output_path=tmp_path / "loaded_output")

    # Basic structural checks
    assert len(graph.nodes()) == len(loaded.nodes())
    assert {n.name for n in graph.nodes()} == {n.name for n in loaded.nodes()}
    assert {
        (u.name, v.name) for u, v in graph.edges()
    } == {(u.name, v.name) for u, v in loaded.edges()}

    # Sanity-check that loaded graph can still execute
    df = generate_mock_data(n_units=100, n_periods=2, n_treated=20)
    loaded.fit(df=df) 