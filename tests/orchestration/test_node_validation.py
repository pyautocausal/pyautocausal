import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import Node, InputNode
import inspect

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

class TestClass:
    def __init__(self, class_multiplier: int):
        self.class_multiplier = class_multiplier

    def instance_method(self, value: int) -> int:
        return value * self.class_multiplier
    
    @classmethod
    def class_method(cls, value: int) -> int:
        return value * 3
    
    @staticmethod
    def static_method(value: int) -> int:
        return value * 4

def test_duplicate_node_name():
    """Test that adding a node with a duplicate name raises an error"""
    graph = ExecutableGraph()
    
    # Add first node
    node1 = Node(name="process", action_function=process_dataframe)
    graph.add_node_to_graph(node1)
    
    # Try to add second node with same name
    node2 = Node(name="process", action_function=process_dataframe)
    with pytest.raises(ValueError) as exc_info:
        graph.add_node_to_graph(node2)
    assert "already exists in the graph" in str(exc_info.value)

def test_duplicate_input_node_name():
    """Test that adding an input node with a duplicate name raises an error"""
    graph = ExecutableGraph()
    
    # Add first input node
    input1 = InputNode(name="input", input_dtype=pd.DataFrame)
    graph.add_node_to_graph(input1)
    
    # Try to add second input node with same name
    input2 = InputNode(name="input", input_dtype=pd.DataFrame)
    with pytest.raises(ValueError) as exc_info:
        graph.add_node_to_graph(input2)
    assert "already exists in the graph" in str(exc_info.value)

def test_graph_builder_duplicate_node_name():
    """Test that ExecutableGraph prevents duplicate node names"""
    graph = (ExecutableGraph()
    .create_node(
        name="process",
        action_function=process_dataframe
    ))
    
    # Try to add second node with same name
    with pytest.raises(ValueError) as exc_info:
        graph.create_node(
            name="process",
            action_function=process_dataframe
        )
    assert "already exists in the graph" in str(exc_info.value) 

def test_validate_action_function_static():
    """Test that validate_action_function_static rejects instance and class methods"""
    # Create test class instance
    test_obj = TestClass(5)
    
    # Create graph
    graph = ExecutableGraph()
    
    # Test with a regular function (should work fine)
    graph.create_node(name="process", action_function=process_dataframe)
    
    # Test with a static method (should work fine)
    graph.create_node(name="static", action_function=TestClass.static_method)
    

    # Test with an instance method (should raise ValueError)
    with pytest.raises(ValueError) as exc_info:
        graph.create_node(name="instance", action_function=TestClass.instance_method)
    assert "action_function must be a static function" in str(exc_info.value)
    
    # Get direct reference to class method using __func__
    class_method = TestClass.class_method.__func__
    # Test with a class method (should raise ValueError)
    with pytest.raises(ValueError) as exc_info:
        graph.create_node(name="class", action_function=class_method)
    assert "action_function must be a static function" in str(exc_info.value) 