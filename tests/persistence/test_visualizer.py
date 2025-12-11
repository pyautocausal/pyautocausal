import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

# Define action functions for the graph nodes
def node1_action(input_data: pd.DataFrame) -> pd.DataFrame:
    """Simple processing function for node1"""
    if isinstance(input_data, pd.DataFrame):
        return input_data * 2
    raise ValueError(f"Expected pd.DataFrame, got {type(input_data)}")

def node2_action(node1_output: pd.DataFrame) -> pd.Series:
    """Simple processing function for node2"""
    if isinstance(node1_output, pd.DataFrame):
        return node1_output.mean()
    raise ValueError(f"Expected pd.DataFrame, got {type(node1_output)}")

@pytest.fixture
def simple_graph(tmp_path):
    """Create a simple test graph with a few nodes"""
    
    graph = (ExecutableGraph()
             .configure_runtime(output_path=tmp_path)
        .create_input_node("input")
        .create_node(
            "node1",
            node1_action,
            predecessors=["input"],
            save_node=True,
            output_config=OutputConfig(
                output_filename="node1_output",
                output_type=OutputType.PARQUET
            )
        )
        .create_node(
            "node2",
            node2_action,
            predecessors=["node1"],
            save_node=True,
            output_config=OutputConfig(
                output_filename="node2_output",
                output_type=OutputType.PARQUET
            )
        )
        )
    return graph

def test_visualize_graph_creates_file(simple_graph, tmp_path):
    """Test that visualize_graph creates an output file"""
    output_file = tmp_path / "test_graph.md"
    visualize_graph(simple_graph, save_path=str(output_file))
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_visualize_graph_node_labels(simple_graph, tmp_path):
    """Test that node labels are correctly set to node names"""
    output_file = tmp_path / "test_graph.md"
    
    # Get labels by accessing the internal state
    visualize_graph(simple_graph, save_path=str(output_file))
    
    # grab the output file
    with open(output_file, 'r') as f:
        content = f.read()
    
    expected_content = """# Graph Visualization

        ## Executable Graph

        ```mermaid
        graph TD
            node0[input]
            node1[node1]
            node2[node2]
            node0 --> node1
            node1 --> node2

            %% Node styling
            classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
            classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
            classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
            classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
            classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
            style node0 fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black
            style node1 fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black
            style node2 fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black
        ```

        ## Node Legend

        ### Node Types
        ```mermaid
        graph LR
            actionNode[Action Node] ~~~ decisionNode{Decision Node}
            style actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black
            style decisionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black
        ```

        ### Node States
        ```mermaid
        graph LR
            pendingNode[Pending]:::pendingNode ~~~ runningNode[Running]:::runningNode ~~~ completedNode[Completed]:::completedNode ~~~ failedNode[Failed]:::failedNode ~~~ passedNode[Passed]:::passedNode

            classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;
            classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;
            classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;
            classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;
            classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;
        ```

        Node state coloring indicates the execution status of each node in the graph."""
    
    # Remove all whitespace from both strings before comparing
    assert ''.join(content.split()) == ''.join(expected_content.split())