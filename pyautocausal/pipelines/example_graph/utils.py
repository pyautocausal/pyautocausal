"""Utility functions for the causal inference pipeline.

This module contains helper functions for:
- Directory setup
- Execution summary and reporting  
- Output export (visualizations, notebooks, HTML)
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import webbrowser
import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.visualizer import visualize_graph
from pyautocausal.persistence.notebook_export import NotebookExporter


def setup_output_directories(output_path: Path) -> Tuple[Path, Path]:
    """Create and return output subdirectories for plots and text.
    
    Args:
        output_path: Base output directory
        
    Returns:
        Tuple of (plots_dir, text_dir) as absolute paths
    """
    
    plots_dir = output_path / "plots"
    text_dir = output_path / "text"
    
    plots_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    
    return plots_dir.absolute(), text_dir.absolute()


def print_execution_summary(graph: ExecutableGraph) -> None:
    """Print a summary of graph execution results.
    
    Shows how many nodes were executed vs skipped due to decision branching.
    
    Args:
        graph: The ExecutableGraph that was executed
    """
    executed_nodes = sum(
        1 for node in graph.nodes() 
        if hasattr(node, 'state') and node.state.name == 'COMPLETED'
    )
    skipped_nodes = sum(
        1 for node in graph.nodes() 
        if hasattr(node, 'state') and node.state.name == 'PASSED'
    )
    total_nodes = len(list(graph.nodes()))
    
    print(f"Total nodes in graph: {total_nodes}")
    print(f"Executed nodes: {executed_nodes}")
    print(f"Skipped nodes (due to branching): {skipped_nodes}")


def export_outputs(
    graph: ExecutableGraph, 
    data_path: Optional[Union[Path, str]] = None,
    df: Optional[pd.DataFrame] = None,
    output_path: Path = None,
    data_filename: str = "data.csv"
) -> None:
    """Export graph visualization, notebook, and HTML report.
    
    This function creates:
    1. A markdown visualization of the graph structure
    2. A Jupyter notebook with the analysis code
    3. An HTML report with executed results (if possible)
    
    Args:
        graph: The ExecutableGraph that was executed
        data_path: Path to the data file (mutually exclusive with df)
        df: DataFrame to save and use (mutually exclusive with data_path)
        output_path: Base output directory
        data_filename: Name to use when saving DataFrame (default: "data.csv")
    
    Raises:
        ValueError: If neither or both data_path and df are provided
    """
    
    # Validate inputs
    if data_path is None and df is None:
        raise ValueError("Either data_path or df must be provided")
    if data_path is not None and df is not None:
        raise ValueError("Only one of data_path or df should be provided, not both")

    # Create output directories if they don't exist
    plots_dir, text_dir = setup_output_directories(output_path)

    # Handle DataFrame case - save it to output directory
    if df is not None:
        data_path = output_path / data_filename
        df.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")

    # Graph visualization
    md_visualization_path = text_dir / "pipeline_visualization.md"
    visualize_graph(graph, save_path=str(md_visualization_path))
    print(f"Graph visualization saved to {md_visualization_path}")
    
    # Notebook and HTML export - place directly in output_path
    notebook_path = output_path / "pipeline_execution.ipynb"
    html_path = output_path / "pipeline_execution.html"
    
    exporter = NotebookExporter(graph)
    
    # Export notebook - use just the filename since notebook and data are in same directory
    data_file_name = Path(data_path).name
    
    exporter.export_notebook(
        str(notebook_path),
        data_path=data_file_name,
        loading_function="pd.read_csv"
    )
    print(f"Notebook exported to {notebook_path}")
    
    # Export and run to HTML
    try:
        html_output_path = exporter.export_and_run_to_html(
            notebook_filepath=notebook_path,
            html_filepath=html_path,
            data_path=data_file_name,
            loading_function="pd.read_csv",
            timeout=300  # 5 minutes timeout
        )
        print(f"HTML report with executed results exported to {html_output_path}")
    except Exception as e:
        print(f"HTML export failed: {e}")
        print("Notebook is still available for manual inspection")


def print_data_characteristics(data) -> None:
    """Print summary of data characteristics relevant to causal analysis.
    
    Args:
        data: The input DataFrame
    """
    all_units = data['id_unit'].unique()
    ever_treated_units = data[data['treat'] == 1]['id_unit'].unique()
    never_treated_units = set(all_units) - set(ever_treated_units)
    never_treated_ratio = len(never_treated_units) / len(all_units)
    
    print(f"Total units: {len(all_units)}, Ever treated: {len(ever_treated_units)}")
    print(f"Never-treated units: {never_treated_ratio * 100:.1f}%")
    
    # Determine expected analysis path
    periods = data['t'].nunique()
    if periods == 1:
        print("Single period data → Cross-sectional analysis")
    elif len(ever_treated_units) == 1:
        print("Single treated unit → Synthetic DiD analysis")
    elif data[data['treat'] == 1].groupby('id_unit')['t'].min().nunique() > 1:
        print("Staggered treatment → Callaway & Sant'Anna methods")
    else:
        print("Panel data → Standard DiD or Event study") 