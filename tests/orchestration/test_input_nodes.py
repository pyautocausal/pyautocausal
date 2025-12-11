import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import InputNode
from typing import Any

def test_input_node_type_validation():
    """Test that input nodes validate data types correctly"""
    # Create graph with typed input nodes
    graph = (ExecutableGraph()
        .create_input_node("df_input", input_dtype=pd.DataFrame)
        .create_input_node("int_input", input_dtype=int)
        .create_input_node("any_input")  # defaults to Any
        )
    
    # Test valid inputs
    graph.fit(
        df_input=pd.DataFrame({'a': [1, 2, 3]}),
        int_input=42,
        any_input="any type works"
    )
    
    # Verify nodes completed successfully
    for node in graph.nodes():
        assert node.is_completed()

def test_input_node_type_validation_errors():
    """Test that input nodes raise appropriate type errors"""
    # Create graph with typed input node
    graph = (ExecutableGraph()
        .create_input_node("df_input", input_dtype=pd.DataFrame)
        )
    
    # Test invalid input
    with pytest.raises(TypeError, match="must be of type DataFrame"):
        graph.fit(df_input=[1, 2, 3])

def test_graph_builder_with_typed_inputs():
    """Test that graph builder correctly sets up typed input nodes"""
    
    
    # Add typed input node
    graph = ExecutableGraph().create_input_node("data", input_dtype=pd.DataFrame)
    
    # Get the input node and verify its type
    input_node = graph.get("data")
    assert isinstance(input_node, InputNode)
    assert input_node.input_dtype == pd.DataFrame
    
    # Verify default Any type
    graph.create_input_node("any_data")
    assert graph.get("any_data").input_dtype == Any

def test_complex_graph_with_typed_inputs():
    """Test typed inputs in a more complex graph with processing nodes"""
    graph = (ExecutableGraph()
        .create_input_node("data", input_dtype=pd.DataFrame)
        .create_node(
            "process",
            lambda data: len(data),
            predecessors=["data"]
        )
        )
    
    # Test with valid input
    df = pd.DataFrame({'a': [1, 2, 3]})
    graph.fit(data=df)
    
    process_node = [n for n in graph.nodes() if n.name == "process"][0]
    assert process_node.get_result_value() == 3
    
def test_input_node_type_validation_errors():
    """Test that input nodes raise appropriate type errors"""
    # Create graph with typed input node
    graph = (ExecutableGraph()
        .create_input_node("data", input_dtype=pd.DataFrame)
        .create_node(
            "process",
            lambda data: len(data),
            predecessors=["data"]
        )
        )
    
    # Test with invalid input
    with pytest.raises(TypeError, match="must be of type DataFrame"):
        graph.fit(data=[1, 2, 3]) 