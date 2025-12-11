import pytest
import pandas as pd
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.orchestration.run_context import RunContext
from pyautocausal.orchestration.nodes import NodeExecutionError

def process_data(
    data: pd.DataFrame,           # required param from predecessor
    n_jobs: int = 1,           # optional param with default
    verbose: bool = False,      # optional param with default
    model_type: str = 'basic'   # optional param with default
):
    return (
        f"Processed {data.shape[0]} rows with {n_jobs} jobs, "
        f"verbose={verbose}, "
        f"model_type={model_type}"
    )

@pytest.fixture
def basic_graph():
    """Graph with just data and process nodes using default parameters"""
    graph = ExecutableGraph()
    graph.configure_runtime()  # Configure with defaults
    
    # Create input node for data
    graph.create_input_node("data", input_dtype=pd.DataFrame)
    
    # Create process node
    graph.create_node(
        "process",
        action_function=process_data,
        predecessors=["data"]
    )
    
    return graph

@pytest.fixture
def graph_with_context():
    """Graph with run context overriding some parameters"""
    context = RunContext()
    context.n_jobs = 4
    context.model_type = 'advanced'
    
    graph = ExecutableGraph()
    graph.configure_runtime(run_context=context)
    
    # Create input node for data
    graph.create_input_node("data", input_dtype=pd.DataFrame)
    
    # Create process node
    graph.create_node(
        "process",
        action_function=process_data,
        predecessors=["data"]
    )
    
    return graph

@pytest.fixture
def graph_with_verbose_data():
    """Graph where data node provides verbose parameter"""
    def data_with_config():
        df = pd.DataFrame()
        return df

    context = RunContext()
    context.n_jobs = 4
    context.model_type = 'advanced'
    
    graph = ExecutableGraph()
    graph.configure_runtime(run_context=context)
    
    # Create input node for data
    graph.create_input_node("data", input_dtype=pd.DataFrame)
    
    # Create process node with custom argument mapping
    graph.create_node(
        "process", 
        action_function=process_data,
        action_condition_kwarg_map={"verbose": "data"},
        predecessors=["data"]
    )
    
    return graph

def test_default_parameters(basic_graph):
    """Test that default parameter values are used when not overridden"""
    graph = basic_graph
    result = graph.fit(data=pd.DataFrame())
    
    # Find the process node and check its result
    process_node = None
    for node in graph.nodes():
        if node.name == "process":
            process_node = node
            break
    
    assert process_node is not None
    assert process_node.get_result_value() == "Processed 0 rows with 1 jobs, verbose=False, model_type=basic"

def test_run_context_override(graph_with_context):
    """Test that run context values override default parameters"""
    graph = graph_with_context
    result = graph.fit(data=pd.DataFrame())
    
    # Find the process node and check its result
    process_node = None
    for node in graph.nodes():
        if node.name == "process":
            process_node = node
            break
    
    assert process_node is not None
    assert process_node.get_result_value() == "Processed 0 rows with 4 jobs, verbose=False, model_type=advanced"

def test_missing_required_argument():
    """Test that missing required parameters raise appropriate error"""
    def process_data(required_param: str, data: pd.DataFrame):
        return f"Processed {required_param}"
    
    graph = ExecutableGraph()
    graph.configure_runtime()  # Configure with defaults
    
    # Create input node for data
    graph.create_input_node("data", input_dtype=pd.DataFrame)
    
    # Create process node
    graph.create_node(
        "process",
        action_function=process_data,
        predecessors=["data"]
    )
    
    with pytest.raises(NodeExecutionError) as exc_info:
        graph.fit(data=pd.DataFrame())
    
    # Check that the original ValueError is wrapped and contains the expected message
    assert "Missing required parameters" in str(exc_info.value)
    assert "required_param" in str(exc_info.value)
    assert isinstance(exc_info.value.original_exception, ValueError)
    assert "Missing required parameters" in str(exc_info.value.original_exception)
    assert "required_param" in str(exc_info.value.original_exception) 