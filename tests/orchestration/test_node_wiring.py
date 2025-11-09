import pytest
from typing import Any
import pandas as pd
from pyautocausal.orchestration.nodes import Node, InputNode
from pyautocausal.orchestration.graph import ExecutableGraph


# Test functions with type hints
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df

def process_string(s: str) -> str:
    return s

def untyped_function(x):
    return x

# Test cases
def test_valid_type_connection():
    """Test connecting nodes with compatible types"""
    graph = ExecutableGraph()
    
    # Create nodes
    node = Node(
        name="process_df",
        action_function=process_dataframe,
        save_node=True
    )
    graph.add_node_to_graph(node)
    
    input_node = InputNode(name="input_df", input_dtype=pd.DataFrame)
    graph.add_node_to_graph(input_node)
    
    # Should not raise any errors
    node >> input_node

def test_incompatible_type_connection():
    """Test connecting nodes with incompatible types raises TypeError"""
    graph = ExecutableGraph()
    
    # Create nodes with incompatible types
    node = Node(
        name="process_string",
        action_function=process_string,
        save_node=True
    )
    graph.add_node_to_graph(node)
    
    input_node = InputNode(name="input_df", input_dtype=pd.DataFrame)
    graph.add_node_to_graph(input_node)
    
    # Should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        node >> input_node
    assert "Type mismatch" in str(exc_info.value)

def test_untyped_function_warning():
    """Test warning is logged when connecting untyped function"""
    graph = ExecutableGraph()
    
    # Create nodes
    node = Node(
        name="untyped",
        action_function=untyped_function,
        save_node=False
    )
    graph.add_node_to_graph(node)
    
    input_node = InputNode(name="input", input_dtype=Any)
    graph.add_node_to_graph(input_node)
    
    # Should log warning
    with pytest.warns(Warning) as warning_info:
        node >> input_node
        assert "Cannot validate connection" in str(warning_info[0].message)
        assert "lacks return type annotation" in str(warning_info[0].message)

def test_any_input_type_warning():
    """Test warning is logged when input node accepts Any type"""
    graph = ExecutableGraph()
    
    # Create nodes
    node = Node(
        name="process_df",
        action_function=process_dataframe,
        save_node=True
    )
    graph.add_node_to_graph(node)
    
    input_node = InputNode(name="input", input_dtype=Any)
    graph.add_node_to_graph(input_node)
    
    # Should log warning
    with pytest.warns(Warning) as warning_info:
        node >> input_node
        assert "Cannot validate connection" in str(warning_info[0].message)
        assert "accepts Any type" in str(warning_info[0].message)

def test_chaining_connections():
    """Test chaining multiple connections"""
    graph = ExecutableGraph()
    graph2 = ExecutableGraph()
    
    # Create nodes
    node1 = Node(
        name="process_df1",
        action_function=process_dataframe,
        save_node=False
    )
    graph.add_node_to_graph(node1)

    input_node = InputNode(name="input_df", input_dtype=pd.DataFrame)
    graph2.add_node_to_graph(input_node)
    
    input_node2 = InputNode(name="input_df2", input_dtype=pd.DataFrame)
    graph2.add_node_to_graph(input_node2)
    
    # Chain connections
    node1 >> input_node
    node1 >> input_node2
