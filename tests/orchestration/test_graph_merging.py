import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.nodes import Node, InputNode
import copy
from inspect import Parameter, Signature

def process_dataframe(external_input: pd.DataFrame) -> pd.DataFrame:
    return external_input

def transform_dataframe(internal_input: pd.DataFrame) -> pd.DataFrame:
    return internal_input

def test_merge_linked_graphs():
    """Test merging graphs that are properly linked"""
    graph1Builder = ExecutableGraph()
    graph2Builder = ExecutableGraph()

    # Create and link nodes
    graph1Builder.create_node(name="process1", action_function=process_dataframe)
    graph2Builder.create_input_node(name="input2", input_dtype=pd.DataFrame)

    graph1 = graph1Builder
    graph2 = graph2Builder
    
    # Merge graphs with explicit wiring
    merged = graph1.merge_with(graph2, graph1.get("process1") >> graph2.get("input2"))
    
    # Verify structure
    new_input = next(n for n in merged.nodes() if n.name == "input2")
    assert new_input in merged.nodes()
    assert graph1.get("process1") in merged.nodes()
    assert new_input in graph1.get_node_successors(graph1.get("process1"))
    assert new_input.graph == merged

def test_fit_with_merged_graphs():
    """Test fit behavior with merged graphs"""
    graph1 = (ExecutableGraph()
    .create_input_node(name="external_input", input_dtype=pd.DataFrame)
    .create_node(name="process", action_function=process_dataframe, predecessors=["external_input"])
    )
    
    graph2 = (ExecutableGraph()
    .create_input_node(name="internal_input", input_dtype=pd.DataFrame)
    .create_node(name="transform", action_function=transform_dataframe, predecessors=["internal_input"])
    )
    
    # Merge graphs with explicit wiring
    graph1.merge_with(graph2, graph1.get("process") >> graph2.get("internal_input"))

    assert len(graph1.input_nodes) == 1
    
    # Should only need to provide external input
    test_df = pd.DataFrame({'a': [1, 2, 3]})
    

    graph1.fit(external_input=test_df)
    
    # Verify execution
    new_input = graph1.get("internal_input")
    new_transform = graph1.get("transform")
    assert new_input.is_completed()
    assert new_transform.is_completed()

def test_fit_with_multiple_external_inputs():
    """Builed two graphs. Graph 1 has one integer input, squares it and then has one output.
    and then has one output. Graph 2 takes two integer inputs and adds them together.
    Wire graph 2 to graph 1 and verify that the merge graph has two inputs and one output."""
    def square(input1: int) -> int:
        return input1**2
    
    def add(input2: int, input3: int) -> int:
        return input2 + input3
    
    graph1 = (ExecutableGraph()
    .create_input_node(name="input1", input_dtype=int)
    .create_node(name="square", action_function=square, predecessors=["input1"])
    )
    
    graph2 = (ExecutableGraph()
    .create_input_node(name="input2", input_dtype=int)
    .create_input_node(name="input3", input_dtype=int)
    .create_node(name="add", action_function=add, predecessors=["input2", "input3"])
    )
    
    graph1.merge_with(graph2, graph1.get("square") >> graph2.get("input2"))

    graph1.fit(input1=2, input3=3)
    assert graph1.get("add").is_completed()
    assert graph1.get("add").get_result_value() == 7
    
    assert len(graph1.input_nodes) == 2

def test_merge_with_non_pending_nodes():
    def square(input1: int) -> int:
        return input1**2
    
    def add(input2: int, input3: int) -> int:
        return input2 + input3
    
    graph1 = (ExecutableGraph()
    .create_input_node(name="input1", input_dtype=int)
    .create_node(name="square", action_function=square, predecessors=["input1"])
    )
    
    graph2 = (ExecutableGraph()
    .create_input_node(name="input2", input_dtype=int)
    .create_input_node(name="input3", input_dtype=int)
    .create_node(name="add", action_function=add, predecessors=["input2", "input3"])
    )

    graph2.fit(input2=2, input3=3)
    
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("square") >> graph2.get("input2"))
    assert "not in PENDING state" in str(exc_info.value)
    

def test_merge_with_duplicate_node_names():
    """Test that merging graphs with duplicate node names creates
    new nodes with unique names"""
    def add_one(input1: int) -> int:
        return input1 + 1
    
    def add_one_v2(input2: int) -> int:
        return input2 + 1
    
    graph1 = (ExecutableGraph()
    .create_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors=["input1"])
    )

    graph2 = (ExecutableGraph()
    .create_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one_v2, predecessors=["input2"])
    )

    graph1.merge_with(graph2, graph1.get("add_one") >> graph2.get("input2"))

    assert len(graph1.nodes()) == 4
    assert "add_one" in graph1._nodes_by_name
    assert "add_one_1" in graph1._nodes_by_name

    graph1.fit(input1=1)
    assert graph1.get("add_one_1").is_completed()
    assert graph1.get("add_one_1").get_result_value() == 3


def test_merge_with_non_input_target():
    """Test that merging fails when target node is not an InputNode"""
    def add_one(input1: int) -> int:
        return input1 + 1
    
    def add_one_v2(input2: int) -> int:
        return input2 + 1
    
    graph1 = (ExecutableGraph()
    .create_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors=["input1"])
    )

    graph2 = (ExecutableGraph()
    .create_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one_v2, predecessors=["input2"])
    )

    
    # Attempt to wire regular nodes should fail
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("add_one") >> graph2.get("add_one"))
    assert "Target node must be an input node" in str(exc_info.value)

def test_merge_with_wrong_graph_nodes():
    """Test that merging fails when nodes are from wrong graphs"""
    def add_one(input1: int) -> int:
        return input1 + 1
    
    def add_one_v2(input2: int) -> int:
        return input2 + 1
    
    def add_one_v3(input3: int) -> int:
        return input3 + 1
    
    graph1 = (ExecutableGraph()
    .create_input_node(name="input1", input_dtype=int)
    .create_node(name="add_one", action_function=add_one, predecessors=["input1"])
    )

    graph2 = (ExecutableGraph()
    .create_input_node(name="input2", input_dtype=int)
    .create_node(name="add_one", action_function=add_one_v2, predecessors=["input2"])
    )

    graph3 = (ExecutableGraph()
    .create_input_node(name="input3", input_dtype=int)
    .create_node(name="add_one", action_function=add_one_v3, predecessors=["input3"])
    )

    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2, graph1.get("add_one") >> graph3.get("input3"))
    assert str(exc_info.value).startswith("Invalid wiring")



def test_merge_preserves_right_hand_graph():
    """This test does nothing because I can't figure out how to copy a graph
    TODO: Test this
    """
    pass
    
def test_merge_with_no_wirings():
    """Test that merging without any wirings fails"""
    graph1 = ExecutableGraph()
    graph2 = ExecutableGraph()
    
    with pytest.raises(ValueError) as exc_info:
        graph1.merge_with(graph2)
    assert "At least one wiring" in str(exc_info.value)