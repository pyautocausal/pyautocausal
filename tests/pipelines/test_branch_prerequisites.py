"""Branch builders fail before mutating a graph lacking shared inputs."""
import pytest

from pyautocausal import ExecutableGraph, create_panel_graph
from pyautocausal.pipelines.example_graph import branches


@pytest.mark.parametrize('builder', [
    branches.create_cross_sectional_branch,
    branches.create_synthetic_did_branch,
    branches.create_hainmueller_synth_branch,
    branches.create_did_branch,
    branches.create_event_study_branch,
    branches.create_staggered_did_branch,
    branches.create_uplift_branch,
])
def test_missing_prerequisites_leave_existing_graph_unchanged(builder):
    graph = ExecutableGraph().create_input_node('unrelated', int)
    before = list(graph.nodes), list(graph.edges)
    with pytest.raises(ValueError, match='missing prerequisite nodes'):
        builder(graph)
    assert (list(graph.nodes), list(graph.edges)) == before


def test_shared_panel_structure_supports_all_panel_branches(tmp_path):
    graph = create_panel_graph(tmp_path)
    for name in ('synthdid_spec', 'hainmueller_fit',
                 'did_spec', 'event_spec', 'stag_spec'):
        assert graph.get(name) is not None
