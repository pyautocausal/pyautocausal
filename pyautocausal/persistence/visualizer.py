import logging
from typing import Dict, Optional, Set
import os
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_node_style(node) -> str:
    """Get mermaid style for nodes based on their state"""
    # Base style format string
    style_format = "style {node_id} fill:{fill_color},stroke:{stroke_color},stroke-width:2px,color:black"
    
    # Default colors based on state
    fill_color = "lightblue"  # Default color (PENDING)
    stroke_color = "#3080cf"  # Default stroke color
    
    # Set colors based on state if available
    if hasattr(node, 'state'):
        state_name = node.state.name if isinstance(node.state, Enum) else str(node.state)
        if state_name == 'COMPLETED':
            fill_color = "lightgreen"
        elif state_name == 'FAILED':
            fill_color = "salmon"
        elif state_name == 'RUNNING':
            fill_color = "yellow"
        elif state_name == 'PASSED':
            fill_color = "#d8d8d8"  # Light gray
    
    return style_format.format(node_id="{node_id}", fill_color=fill_color, stroke_color=stroke_color)

def is_decision_node(node) -> bool:
    """Check if a node is a decision node"""
    # Check if the node is an instance of DecisionNode
    if hasattr(node, '__class__') and hasattr(node.__class__, '__name__'):
        if node.__class__.__name__ == 'DecisionNode':
            return True
    
    # Also check for duck typing - if it has the key attributes of a decision node
    if (hasattr(node, '_ewt_nodes') and 
        hasattr(node, '_ewf_nodes') and 
        hasattr(node, 'condition_satisfied')):
        return True
    
    return False

def visualize_graph(graph, save_path = None):
    """
    Visualize the provided graph as a mermaid flowchart within a markdown file.
    
    Parameters:
        graph (nx.DiGraph): The directed graph (e.g., an instance of ExecutableGraph) to visualize.
        save_path (str): The file path where the resulting markdown file will be saved.
                         Should end with .md extension.
        return_positions (bool): Not used with mermaid, kept for backward compatibility.
        return_labels (bool): Not used with mermaid, kept for backward compatibility.
    
    Returns:
        None
    """
    # Validate input
    if len(graph) == 0:
        raise ValueError("Graph is empty")
    

        
    # Start building the mermaid diagram
    markdown_lines = ["# Graph Visualization", "", "## Executable Graph"]
    mermaid_lines = ["```mermaid", "graph TD"]
    
    # Add node definitions
    node_styles = []
    node_ids = {}
    
    # Track if we have any decision nodes
    has_decision_nodes = False
    
    # Create unique, safe IDs for each node
    for i, node in enumerate(graph.nodes()):
        # Create a safe ID without special characters
        safe_id = f"node{i}"
        node_ids[node] = safe_id
        
        # Add node with label
        node_label = node.name if hasattr(node, 'name') else str(node)
        
        # Use diamond shape for decision nodes, rectangle for action nodes
        if is_decision_node(node):
            has_decision_nodes = True
            mermaid_lines.append(f"    {safe_id}{{{node_label}}}")  # Diamond shape for decision nodes
        else:
            mermaid_lines.append(f"    {safe_id}[{node_label}]")  # Rectangle for action nodes
        
        # Get node style based on state
        style_template = get_node_style(node)
        node_styles.append(style_template.format(node_id=safe_id))
    
    # Add edges with conditional labels for decision nodes
    for src, dst in graph.edges():
        edge_label = ""
        
        # Check if it's a decision node with execute-when-true/false edges
        if is_decision_node(src):
            if hasattr(src, '_ewt_nodes') and dst in src._ewt_nodes:
                edge_label = "|True|"
            elif hasattr(src, '_ewf_nodes') and dst in src._ewf_nodes:
                edge_label = "|False|"
        
        mermaid_lines.append(f"    {node_ids[src]} -->{edge_label} {node_ids[dst]}")
    
    # Add class definitions
    mermaid_lines.extend([
        "",
        "    %% Node styling",
        "    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;",
    ])
    
    # Add individual node styles
    for style in node_styles:
        mermaid_lines.append(f"    {style}")
    
    # End the mermaid diagram
    mermaid_lines.append("```")
    
    # Add legend as a separate section after the diagram
    legend_lines = [
        "",
        "## Node Legend",
        "",
        "### Node Types"
    ]
    
    # Use mermaid to display node shapes correctly
    legend_lines.extend([
        "```mermaid",
        "graph LR",
        "    actionNode[Action Node] ~~~ decisionNode{Decision Node}",
        "    style actionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black",
        "    style decisionNode fill:#d0e0ff,stroke:#3080cf,stroke-width:2px,color:black",
        "```"
    ])
    
    # Add node state legend
    legend_lines.extend([
        "",
        "### Node States",
        "```mermaid",
        "graph LR",
        "    pendingNode[Pending]:::pendingNode ~~~ runningNode[Running]:::runningNode ~~~ completedNode[Completed]:::completedNode ~~~ failedNode[Failed]:::failedNode ~~~ passedNode[Passed]:::passedNode",
        "",
        "    classDef pendingNode fill:lightblue,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef runningNode fill:yellow,stroke:#3080cf,stroke-width:2px,color:black;", 
        "    classDef completedNode fill:lightgreen,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef failedNode fill:salmon,stroke:#3080cf,stroke-width:2px,color:black;",
        "    classDef passedNode fill:#d8d8d8,stroke:#3080cf,stroke-width:2px,color:black;",
        "```",
        "",
        "Node state coloring indicates the execution status of each node in the graph.",
        ""
    ])
    
    # Combine markdown, mermaid, and legend content
    markdown_content = '\n'.join(markdown_lines + [''] + mermaid_lines + legend_lines)
    

    
    

    if save_path is None:
        return markdown_content

    else:
        # Ensure the file extension is .md
        file_path, ext = os.path.splitext(save_path)
        if not ext or ext.lower() != '.md':
            save_path = file_path + '.md'
            logger.info(f"Changed save path to {save_path} to ensure markdown format")
                # Write to markdown file
        with open(save_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Graph visualization saved as markdown to {save_path}")
    # For convenience, if someone runs this module directly

   