import pytest
from pyautocausal.orchestration.nodes import Node, DecisionNode, NodeState
from pyautocausal.orchestration.graph import ExecutableGraph
from typing import Any

# Test functions for actions
def create_number(value: int) -> int:
    return value

def number_is_positive(number: int) -> bool:
    return number > 0

def add_values(true_branch: int) -> int:
    # Only depends on the true_branch now, ignoring false_branch
    return true_branch

def add_values_with_default(true_3: int, false_3=None) -> int:
    # Uses an optional parameter for false_3
    return true_3 + (false_3 or 0)

def test_passed_node_state_basic():
    """
    Test the basic PASSED state functionality:
    1. Creates a graph with a decision node and two branches
    2. Verifies that nodes in the non-taken branch are marked as PASSED
    """
    graph = ExecutableGraph()
    
    # Create nodes
    graph.create_node("source", action_function=lambda: 5)
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    graph.create_node("true_branch", action_function=lambda x: x + 10, predecessors=["decision"])
    graph.create_node("false_branch", action_function=lambda x: x - 10, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "true_branch")
    graph.when_false("decision", "false_branch")
    
    # Execute graph
    graph.execute_graph()
    
    # Verify results
    assert graph.get("true_branch").is_completed()
    assert graph.get("false_branch").is_passed()  # Should be marked as PASSED instead of PENDING
    
    # Check the methods work correctly
    assert graph.get("false_branch").is_passed() == True
    assert graph.get("false_branch").is_completed() == False
    assert graph.get("false_branch").state == NodeState.PASSED

def test_node_with_passed_predecessor():
    """
    Test the case where a node has multiple predecessors and one is PASSED:
    1. Creates a graph with a decision node and a dependent node that has
       predecessors from both the true and false branch
    2. Verifies that the dependent node can execute when one predecessor is PASSED
    """
    graph = ExecutableGraph()
    
    # Create source nodes
    graph.create_node("source", action_function=lambda: 5)
    
    # Create decision node
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    
    # Create branch nodes
    graph.create_node("true_branch", action_function=create_number, predecessors=["decision"])
    graph.create_node("false_branch", action_function=create_number, predecessors=["decision"])
    
    # Configure decision paths
    graph.when_true("decision", "true_branch")
    graph.when_false("decision", "false_branch")
    
    # Create a node that depends on both branches - This simulates the problem
    # we were trying to solve, where a node needs outputs from mutually exclusive branches
    graph.create_node(
        "dependent_node", 
        action_function=add_values,  # Now only using true_branch as param
        predecessors=["true_branch", "false_branch"]
    )
    
    # Execute graph
    graph.execute_graph()
    
    # Verify results
    assert graph.get("source").is_completed()
    assert graph.get("decision").is_completed()
    
    # With positive source, true_branch should complete and false_branch should be passed
    assert graph.get("true_branch").is_completed()
    assert graph.get("false_branch").is_passed()
    
    # The dependent node should now execute successfully
    assert graph.get("dependent_node").is_completed()
    # It should correctly use the value from true_branch and ignore the passed node
    assert graph.get("dependent_node").output.result_dict["dependent_node"] == 5

def test_downstream_passed_propagation():
    """
    Test that the PASSED state correctly propagates to downstream nodes:
    1. Creates a chain of nodes where an early decision makes some paths unreachable
    2. Verifies that all nodes in the unreachable chain are marked as PASSED
    """
    graph = ExecutableGraph()
    
    # Create source
    graph.create_node("source", action_function=lambda: 5)
    
    # Create decision
    graph.create_decision_node(
        "decision", 
        condition=number_is_positive,
        predecessors=["source"]
    )
    
    # Create a chain of nodes in the false branch
    graph.create_node("false_1", action_function=lambda x: x, predecessors=["decision"])
    graph.create_node("false_2", action_function=lambda x: x, predecessors=["false_1"])
    graph.create_node("false_3", action_function=lambda x: x, predecessors=["false_2"])
    
    # Create a chain of nodes in the true branch
    graph.create_node("true_1", action_function=lambda x: x, predecessors=["decision"])
    graph.create_node("true_2", action_function=lambda x: x, predecessors=["true_1"])
    graph.create_node("true_3", action_function=lambda x: x, predecessors=["true_2"])
    
    # Configure decision paths
    graph.when_true("decision", "true_1")
    graph.when_false("decision", "false_1")
    
    # Create a final node that depends on both terminal nodes but with optional parameter
    graph.create_node(
        "final", 
        action_function=add_values_with_default,
        predecessors=["true_3", "false_3"]
    )
    
    # Execute graph
    graph.execute_graph()
    
    # Verify the true branch executed
    assert graph.get("true_1").is_completed()
    assert graph.get("true_2").is_completed()
    assert graph.get("true_3").is_completed()
    
    # Verify the entire false branch was marked as PASSED
    assert graph.get("false_1").is_passed()
    assert graph.get("false_2").is_passed()
    assert graph.get("false_3").is_passed()
    
    # The final node should execute successfully
    assert graph.get("final").is_completed()
    # It should use the true_3 value (5) and default for false_3
    assert graph.get("final").output.result_dict["final"] == 5 