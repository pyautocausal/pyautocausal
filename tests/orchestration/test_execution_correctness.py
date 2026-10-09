import json
import pytest
from pyautocausal.orchestration.graph import ExecutableGraph, IncompleteExecutionError
from pyautocausal.orchestration.nodes import NodeExecutionError
from pyautocausal.orchestration.node_state import NodeState


def _branch_graph():
    graph = ExecutableGraph().create_input_node('value', int)
    graph.create_decision_node('positive', lambda value: value > 0, ['value'])
    graph.create_node('yes', lambda value: value * 2, ['positive'])
    graph.create_node('no', lambda value: -value, ['positive'])
    graph.when_true('positive', 'yes').when_false('positive', 'no')
    graph.create_node('result', lambda answer: answer + 1,
                      {'answer': ['yes', 'no']})
    return graph


@pytest.mark.parametrize('value, expected', [(4, 9), (-4, 5)])
def test_merged_decision_branches_preserve_renamed_members_and_source(value, expected):
    source = _branch_graph()
    original_nodes = list(source.nodes)
    original_edges = list(source.edges(data=True))
    original_branches = source.get('positive')._ewt_nodes.copy()
    target = ExecutableGraph().create_input_node('number', int)
    # Collide with multiple source names, including the branch target/input.
    for name in ['value', 'positive', 'yes']:
        target.create_node(name, lambda: 99)
    target.merge_with(source, target.get('number') >> source.get('value'))
    assert target.get('positive_1')._ewt_nodes == {target.get('yes_1')}
    target.fit(number=value)
    assert target.get('result').get_result_value() == expected
    assert list(source.nodes) == original_nodes
    assert list(source.edges(data=True)) == original_edges
    assert source.get('positive')._ewt_nodes == original_branches
    assert all(node.graph is source and node.state == NodeState.PENDING for node in source.nodes)
    source.fit(value=value)
    assert source.get('result').get_result_value() == expected


def test_join_becomes_ready_after_late_recursive_skip():
    calls = []
    graph = ExecutableGraph()
    graph.create_node('early', lambda: 10)
    graph.create_node('delay', lambda: 1)
    def decide(delay) -> bool:
        calls.append(delay)
        return False
    graph.create_decision_node('late_decision', decide, ['delay'])
    graph.create_node('unused', lambda: 100, ['late_decision'])
    graph.create_node('unused_child', lambda unused: unused, ['unused'])
    graph.when_true('late_decision', 'unused')
    graph.create_node('join', lambda early: early + 1, ['early', 'unused_child'])
    graph.execute_graph()
    assert graph.get('join').get_result_value() == 11
    assert graph.is_execution_finished()
    assert graph.get('unused_child').is_passed()
    assert calls == [1]


def test_alternative_predecessors_reject_ambiguity_even_when_values_equal():
    graph = ExecutableGraph().create_node('a', lambda: 1).create_node('b', lambda: 1)
    graph.create_node('join', lambda value: value, {'value': ['a', 'b']})
    with pytest.raises(NodeExecutionError, match="Ambiguous argument 'value'"):
        graph.execute_graph()


def test_failed_run_reports_error_and_can_be_reset(tmp_path):
    graph = ExecutableGraph().configure_runtime(output_path=tmp_path)
    graph.create_input_node('number', int)
    graph.create_node('divide', lambda number: 10 / number, ['number'])
    with pytest.raises(NodeExecutionError):
        graph.fit(number=0)
    report = json.loads((tmp_path / 'execution_report.json').read_text())
    assert report['nodes']['divide']['state'] == 'failed'
    assert report['nodes']['divide']['error']['type'] == 'ZeroDivisionError'
    with pytest.raises(IncompleteExecutionError, match='Failed nodes'):
        graph.execute_graph()
    graph.reset().fit(number=2)
    assert graph.get('divide').get_result_value() == 5
    assert graph.get('divide').execution_count == 1


def test_blocked_graph_raises_instead_of_returning_success():
    graph = ExecutableGraph().create_node('a', lambda: 1)
    graph.create_node('b', lambda a: a, ['a'])
    graph.edges[graph.get('a'), graph.get('b')]['traversable'] = False
    with pytest.raises(IncompleteExecutionError, match='blocked nodes.*b.*a'):
        graph.execute_graph()


def test_alternative_binding_survives_yaml(tmp_path):
    graph = _branch_graph()
    path = graph.to_yaml(tmp_path / 'graph.yaml')
    restored = ExecutableGraph.from_yaml(path)
    restored.fit(value=-7)
    assert restored.get('result').get_result_value() == 8


@pytest.mark.parametrize('with_decision', [False, True])
def test_nested_merges_compose_original_argument_names(tmp_path, with_decision):
    def add(x, z):
        return x + z
    inner = ExecutableGraph().create_input_node('x', int).create_node('z', lambda: 5)
    if with_decision:
        inner.create_decision_node('positive', lambda x: x > 0, ['x'])
        inner.create_node('sum', add, ['positive', 'z'])
        inner.when_true('positive', 'sum')
    else:
        inner.create_node('sum', add, ['x', 'z'])
    middle = ExecutableGraph().create_input_node('seed', int).create_node('x', lambda: 99)
    middle.merge_with(inner, middle.get('seed') >> inner.get('x'))
    outer = ExecutableGraph().create_node('origin', lambda: 3).create_node('x_1', lambda: 99)
    outer.merge_with(middle, outer.get('origin') >> middle.get('seed'))
    # Persist before execution to verify that the effective aliases survive.
    restored = ExecutableGraph.from_yaml(outer.to_yaml(tmp_path / 'nested.yaml'))
    for graph in [outer, restored]:
        graph.execute_graph()
        assert graph.get('sum').get_result_value() == 8
