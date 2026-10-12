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
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    failed_nodes = sum(node.state.name == "FAILED" for node in graph.nodes())
    incomplete_nodes = len(graph.get_incomplete_nodes())
    print(f"Total nodes in graph: {total_nodes}")
    print(f"Executed nodes: {executed_nodes}")
    print(f"Skipped nodes (due to branching): {skipped_nodes}")
    print(f"Failed nodes: {failed_nodes}; incomplete nodes: {incomplete_nodes}")


def export_outputs(
    graph: ExecutableGraph, 
    data_path: Optional[Union[Path, str]] = None,
    df: Optional[pd.DataFrame] = None,
    output_path: Path = None,
    data_filename: str = "data.csv",
    *,
    compact: bool = True,
    title: Optional[str] = None,
) -> None:
    """Export graph visualization, notebook, and HTML report.
    
    This function creates:
    1. A markdown visualization of the graph structure
    2. A Jupyter notebook with the analysis code and executed results
    3. An HTML report with executed results
    
    Args:
        graph: The ExecutableGraph that was executed
        data_path: Path to the data file (mutually exclusive with df)
        df: DataFrame to save and use (mutually exclusive with data_path).
            A CSV copy is saved for inspection and a typed Parquet snapshot for replay.
        output_path: Base output directory
        data_filename: Name to use when saving DataFrame (default: "data.csv")
        compact: Collapse setup/implementation code and hide it in HTML (default: True)
        title: Analysis title shown in the notebook and HTML report
    
    Raises:
        ValueError: If neither or both data_path and df are provided
        Exception: Export failures propagate and are recorded in execution_report.json
    """
    
    # Validate inputs
    if data_path is None and df is None:
        raise ValueError("Either data_path or df must be provided")
    if data_path is not None and df is not None:
        raise ValueError("Only one of data_path or df should be provided, not both")

    if output_path is None:
        raise ValueError("output_path is required")
    output_path = Path(output_path)

    # Create output directories if they don't exist
    plots_dir, text_dir = setup_output_directories(output_path)

    from pyautocausal.persistence.local_output_handler import LocalOutputHandler
    handler = LocalOutputHandler(output_path)
    report = graph.execution_report()
    report['export'] = {'state': 'running', 'stage': 'data'}
    handler.save_json("execution_report", report)
    md_visualization_path = text_dir / "pipeline_visualization.md"
    notebook_path = output_path / "pipeline_execution.ipynb"
    html_path = output_path / "pipeline_execution.html"
    exporter = NotebookExporter(graph)

    try:
        loading_function = "pd.read_csv"
        if df is not None:
            data_path = output_path / data_filename
            df.to_csv(data_path, index=False)
            print(f"Data saved to {data_path}")
            # CSV inference can turn distinct labels ('001', '01', '1') into
            # one unit, and loses timestamps, categories and index values.
            # Keep an inspectable CSV, but replay a checked typed snapshot.
            data_path = Path(str(data_path) + '.parquet')
            df.to_parquet(data_path)
            try:
                pd.testing.assert_frame_equal(df, pd.read_parquet(data_path), check_exact=True)
            except AssertionError as error:
                raise ValueError("DataFrame cannot be replayed faithfully from Parquet; "
                                 "use NotebookExporter with an explicit typed loader") from error
            loading_function = "pd.read_parquet"
        else:
            # Refuse a successful-looking report if the convenient CSV loader
            # changes the actual input on which this executed path was based.
            inputs = [node.get_result_value() for node in graph.input_nodes.values()
                      if node.is_completed() and not list(graph.predecessors(node))]
            if len(inputs) == 1 and isinstance(inputs[0], pd.DataFrame):
                try:
                    pd.testing.assert_frame_equal(inputs[0], pd.read_csv(data_path), check_exact=True)
                except AssertionError as error:
                    raise ValueError("CSV replay would change the executed input data; "
                                     "pass df= with that DataFrame for a typed snapshot, "
                                     "or use NotebookExporter with an explicit loader") from error
        report['export']['data'] = {
            'path': str(Path(data_path).name), 'loader': loading_function,
        }

        report['export']['stage'] = 'visualization'
        visualize_graph(graph, save_path=str(md_visualization_path))
        print(f"Graph visualization saved to {md_visualization_path}")

        # Keep data references portable with the notebook's working directory.
        import os
        data_file_name = os.path.relpath(Path(data_path).resolve(), output_path.resolve())
        report['export']['stage'] = 'notebook'
        exporter.export_notebook(
            str(notebook_path), data_path=data_file_name,
            loading_function=loading_function, compact=compact, title=title,
        )
        print(f"Notebook exported to {notebook_path}")

        report['export']['stage'] = 'html'
        html_output_path = exporter.export_and_run_to_html(
            notebook_filepath=notebook_path,
            html_filepath=html_path,
            data_path=data_file_name,
            loading_function=loading_function,
            timeout=300,  # 5 minutes timeout
            compact=compact,
            title=title,
        )
        print(f"HTML report with executed results exported to {html_output_path}")
    except Exception as e:
        report['export'].update(state='failed', error={
            'type': type(e).__name__, 'message': str(e),
        })
        handler.save_json("execution_report", report)
        raise
    report['export'].update(state='completed', artifacts={
        'notebook': notebook_path.name, 'html': html_path.name,
    })
    handler.save_json("execution_report", report)


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
