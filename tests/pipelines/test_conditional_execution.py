import pytest
from pyautocausal.orchestration.nodes import Node, DecisionNode, NodeState, NodeExecutionError
from pyautocausal.orchestration.graph import ExecutableGraph
from typing import Any

# Test functions for decision conditions and branch actions
def always_true() -> bool:
    return True

def always_false() -> bool:
    return False

def number_is_positive(number: int) -> bool:
    return number > 0

def number_is_negative(number: int) -> bool:
    return number < 0

def true_branch() -> str:
    return "true_branch_executed"

def false_branch() -> str:
    return "false_branch_executed"

def identity(x: int) -> int:
    return x

def add_10(x: int) -> int:
    return x + 10

def subtract_10(x: int) -> int:
    return x - 10

def test_basic_decision_node_true_path():
    """
    Test that the decision node correctly handles the true path:
    1. Creates a source node that outputs a positive number
    2. Creates a decision node with a positive number condition
    3. Creates true and false branch nodes
    4. Specifies which nodes execute when true/false
    5. Verifies only the true branch executes
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: 5)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    graph.create_node("true_path", action_function=add_10, predecessors=["decision"])
    graph.create_node("false_path", action_function=subtract_10, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "true_path")
    graph.when_false("decision", "false_path")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify results
    assert graph.get("source").is_completed()
    assert graph.get("decision").is_completed()
    assert graph.get("true_path").is_completed()
    assert graph.get("true_path").output.result_dict == {'true_path': 15}  # 5 + 10
    assert graph.get("false_path").state == NodeState.PASSED  # Should be marked as PASSED now

def test_basic_decision_node_false_path():
    """
    Test that the decision node correctly handles the false path:
    1. Creates a source node that outputs a negative number
    2. Creates a decision node with a positive number condition
    3. Creates true and false branch nodes
    4. Specifies which nodes execute when true/false
    5. Verifies only the false branch executes
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: -5)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    graph.create_node("true_path", action_function=add_10, predecessors=["decision"])
    graph.create_node("false_path", action_function=subtract_10, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "true_path")
    graph.when_false("decision", "false_path")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify results
    assert graph.get("source").is_completed()
    assert graph.get("decision").is_completed() 
    assert graph.get("false_path").is_completed()
    assert graph.get("false_path").output.result_dict == {'false_path': -15}  # -5 - 10
    assert graph.get("true_path").state == NodeState.PASSED  # Should be marked as PASSED now

def test_multi_branch_execution():
    """
    Test that a decision node can have multiple branches for each condition:
    1. Creates a decision node that has multiple true branches and multiple false branches
    2. Verifies all true branches execute (or all false branches execute)
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: 5)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    
    # Create multiple true path nodes
    graph.create_node("true_path1", action_function=add_10, predecessors=["decision"])
    graph.create_node("true_path2", action_function=lambda source: source * 2, predecessors=["decision"])
    
    # Create multiple false path nodes
    graph.create_node("false_path1", action_function=subtract_10, predecessors=["decision"])
    graph.create_node("false_path2", action_function=lambda source: source * -1, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "true_path1")
    graph.when_true("decision", "true_path2")
    graph.when_false("decision", "false_path1")
    graph.when_false("decision", "false_path2")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify results - all true paths should execute, no false paths
    assert graph.get("true_path1").is_completed()
    assert graph.get("true_path1").output.result_dict == {'true_path1': 15}  # 5 + 10
    assert graph.get("true_path2").is_completed()
    assert graph.get("true_path2").output.result_dict == {'true_path2': 10}  # 5 * 2
    assert graph.get("false_path1").state == NodeState.PASSED
    assert graph.get("false_path2").state == NodeState.PASSED

def test_decision_node_passthrough():
    """
    Test that decision nodes correctly pass through their inputs:
    1. Verifies the decision node passes its inputs to downstream nodes
    2. Checks that when multiple inputs exist, they are passed as a dictionary
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("number", action_function=lambda: 42)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["number"]
    )
    
    # Node that simply returns the value it gets from the decision node
    graph.create_node(
        "passthrough", 
        action_function=lambda number: number,  # This should get the value 42
        predecessors=["decision"]
    )
    
    # Configure decision path
    graph.when_true("decision", "passthrough")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify the value was passed through correctly
    assert graph.get("passthrough").is_completed()
    assert graph.get("passthrough").output.result_dict == {'passthrough': 42}

def test_decision_node_validation():
    """
    Test that decision nodes validate that all successors are classified:
    1. Creates a decision node with a successor
    2. Does not classify the successor as true or false
    3. Verifies that validation fails when the graph executes
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: 5)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    graph.create_node("unclassified", action_function=identity, predecessors=["decision"])
    
    # Deliberately don't classify the successor
    # graph.when_true("decision", "unclassified")
    
    # Executing should raise an error during validation
    with pytest.raises(NodeExecutionError) as exc_info:
        graph.execute_graph()
    
    # Check that the original ValueError is wrapped and contains the expected message
    assert "not classified" in str(exc_info.value)
    assert isinstance(exc_info.value.original_exception, ValueError)
    assert "not classified" in str(exc_info.value.original_exception)

def test_chained_decisions():
    """
    Test that decision nodes can be chained:
    1. Creates a chain of decision nodes where the output of one feeds into another
    2. Verifies the correct execution path is taken through the chain
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: 5)
    
    # First decision: is number positive?
    graph.create_decision_node(
        "decision1", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    
    # Second decision: is number > 10?
    def greater_than_10(number):
        return number > 10
    
    graph.create_decision_node(
        "decision2",
        condition=greater_than_10,
        predecessors=["decision1"]
    )
    
    # Terminal nodes for each path
    graph.create_node("negative", action_function=lambda x: "negative", predecessors=["decision1"])
    graph.create_node("small_positive", action_function=lambda x: "small positive", predecessors=["decision2"])
    graph.create_node("large_positive", action_function=lambda x: "large positive", predecessors=["decision2"])
    
    # Configure decision paths
    graph.when_false("decision1", "negative")
    graph.when_true("decision1", "decision2")
    graph.when_false("decision2", "small_positive")
    graph.when_true("decision2", "large_positive")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify correct path was taken
    assert graph.get("source").is_completed()
    assert graph.get("decision1").is_completed()
    assert graph.get("decision2").is_completed()
    assert graph.get("negative").state == NodeState.PASSED
    assert graph.get("small_positive").is_completed()
    assert graph.get("small_positive").output.result_dict == {'small_positive': 'small positive'}
    assert graph.get("large_positive").state == NodeState.PASSED

def test_complex_decision_with_input_nodes():
    """
    Test decision nodes with external inputs:
    1. Creates a graph with input nodes feeding into a decision node
    2. Tests the graph with different inputs to verify both paths
    """
    graph = ExecutableGraph()
    
    # Create input node
    graph.create_input_node("number", input_dtype=int)
    
    # Create decision node
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["number"]
    )
    
    # Create branch nodes
    graph.create_node("positive_branch", action_function=add_10, predecessors=["decision"])
    graph.create_node("negative_branch", action_function=subtract_10, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "positive_branch")
    graph.when_false("decision", "negative_branch")
    
    # Test with positive input
    graph.fit(number=20)
    assert graph.get("positive_branch").is_completed()
    assert graph.get("positive_branch").output.result_dict == {'positive_branch': 30}
    assert graph.get("negative_branch").state == NodeState.PASSED
    


def test_complex_decision_with_input_nodes_negative():
    """
    Test decision nodes with external inputs:
    1. Creates a graph with input nodes feeding into a decision node
    2. Tests the graph with different inputs to verify both paths
    """
    graph = ExecutableGraph()
    
    # Create input node
    graph.create_input_node("number", input_dtype=int)
    
    # Create decision node
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["number"]
    )
    
    # Create branch nodes
    graph.create_node("positive_branch", action_function=add_10, predecessors=["decision"])
    graph.create_node("negative_branch", action_function=subtract_10, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "positive_branch")
    graph.when_false("decision", "negative_branch")
    
    # Test with positive input
    graph.fit(number=-20)
    assert graph.get("negative_branch").is_completed()
    assert graph.get("positive_branch").state == NodeState.PASSED
    assert graph.get("negative_branch").output.result_dict == {'negative_branch': -30}
    
