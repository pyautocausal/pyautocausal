import pytest
import pandas as pd
import numpy as np
import nbformat
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.notebook_decorators import expose_in_notebook
import os
import tempfile

def simple_func(x: pd.DataFrame) -> pd.DataFrame:
    """A simple function that adds a column"""
    x['new_col'] = x['value'] * 2
    return x

def process_func(df: pd.DataFrame) -> str:
    """Process the data and return a string"""
    return f"Processed {len(df)} rows"

# Define a complex function that we want to expose in the notebook
def complex_statistical_function(df: pd.DataFrame, alpha: float = 0.05):
    """
    Perform a complex statistical analysis on the dataframe.
    
    Args:
        df: The input DataFrame to analyze
        alpha: Significance level
        
    Returns:
        Summary statistics
    """
    # Calculate mean and standard deviation for each column
    means = df.mean()
    stds = df.std()
    
    # Calculate confidence intervals
    z_value = 1.96  # Approx. 95% confidence
    ci_lower = means - z_value * stds / np.sqrt(len(df))
    ci_upper = means + z_value * stds / np.sqrt(len(df))
    
    return pd.DataFrame({
        'mean': means,
        'std': stds,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })




@pytest.fixture
def one_node_graph_with_static_method():
    """Create a simple graph with one node"""

    class SimpleClass:

        @staticmethod
        def simple_static_method():
            return "Hello, world!"

    graph = ExecutableGraph()
    graph.create_node("data", SimpleClass.simple_static_method)
    graph.execute_graph()
    return graph


@pytest.fixture
def sample_graph():
    """Create a simple graph for testing"""

    def transform_func(data: pd.DataFrame) -> pd.DataFrame:
        return data
    
    def process_func_action(transform: pd.DataFrame) -> str:
        return f"Processed {len(transform)} rows"
    
    graph = (ExecutableGraph()
        .create_input_node("data")
        .create_node(
            "transform",
            transform_func,
            predecessors=["data"],
            node_description = "Transform the data"
        )
        .create_node(
            "process",
            process_func_action,
            predecessors=["transform"]
        )
        )
    
    # Set input data
    input_data = pd.DataFrame({'value': [1, 2, 3]})
    graph.fit(data=input_data)
    
    return graph

@pytest.fixture
def output_path(tmp_path):
    """Create temporary directory for notebook output"""
    return tmp_path / "test_notebook.ipynb"

def test_notebook_creation(sample_graph, output_path):
    """Test basic notebook creation"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    assert output_path.exists()
    
    # Load and verify notebook
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check basic structure
    assert len(nb.cells) > 0
    assert nb.cells[0].cell_type == "markdown"  # Header

def test_node_cell_creation(sample_graph, output_path):
    """Test that cells are created for each node"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Count node-related cells (each node should have at least 2 cells)
    node_cells = [
        cell for cell in nb.cells 
        if any(node.name in cell.source for node in sample_graph.nodes())
    ]
    
    # We should have at least 2 cells per node (markdown + execution)
    assert len(node_cells) >= len(sample_graph.nodes()) * 2

def test_topological_order(sample_graph, output_path):
    """Test that nodes appear in topological order"""
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(output_path))
    
    with open(output_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Extract node names in order of appearance
    node_order = []
    for cell in nb.cells:
        for node in sample_graph.nodes():
            if node.name in cell.source and node.name not in node_order:
                node_order.append(node.name)
    
    # Verify order (data should come before transform, which comes before process)
    assert node_order.index('data') < node_order.index('transform')
    assert node_order.index('transform') < node_order.index('process')






# Create a wrapper function that adapts parameter names
@expose_in_notebook(
    target_function=complex_statistical_function,
    arg_mapping={'data': 'df', 'significance': 'alpha'}
)
def stats_wrapper(data, significance=0.05):
    """Wrapper for complex_statistical_function that adapts parameter names."""
    return complex_statistical_function(df=data, alpha=significance)

# Test that the wrapper function works and maintains its behavior
def test_wrapper_execution():
    """Test that the wrapper function executes correctly."""
    # Create test data
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    # Call the wrapper function
    result = stats_wrapper(test_data, significance=0.01)
    
    # Check that we got expected output
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns
    assert 'std' in result.columns
    assert 'ci_lower' in result.columns
    assert 'ci_upper' in result.columns


# Test the notebook export functionality with the decorated function
@pytest.fixture
def sample_graph_with_wrapper(tmp_path):
    """Create a sample graph with a wrapper function."""
    # Create a test DataFrame
    def create_data() -> pd.DataFrame:
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    # Build a simple graph
    graph = (ExecutableGraph()
             .configure_runtime(output_path=tmp_path)
             .create_node(
                 "data",
                 create_data,
             )
             .create_node(
                 "stats_processor",
                 stats_wrapper,
                 predecessors=["data"],
             )
    )
    
    # Execute the graph
    graph.execute_graph()
    return graph
    

def test_notebook_export_with_wrapper(sample_graph_with_wrapper, tmp_path):
    """Test that the notebook exporter correctly handles decorated wrapper functions."""
    # Export the notebook
    notebook_path = os.path.join(tmp_path, "test_notebook.ipynb")
    exporter = NotebookExporter(sample_graph_with_wrapper)
    exporter.export_notebook(notebook_path)
    
    # Verify the notebook exists
    assert os.path.exists(notebook_path)
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Look for key indicators that our wrapper was properly processed
    found_target_function = False
    unnecessary_import_statement = False
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'complex_statistical_function' in cell.source:
                found_target_function = True
                if "from test_notebook_export" in cell.source:
                    unnecessary_import_statement = True
    
    # Verify we found all the expected elements
    assert found_target_function, "Target function not found in notebook"
    assert not unnecessary_import_statement, "Unnecessary import statement found in notebook"


def test_notebook_export_with_static_method(one_node_graph_with_static_method, tmp_path):
    """Test that the notebook exporter correctly handles static methods."""
    # Export the notebook
    notebook_path = os.path.join(tmp_path, "test_notebook.ipynb")
    exporter = NotebookExporter(one_node_graph_with_static_method)
    exporter.export_notebook(notebook_path)

    # Verify the notebook exists
    assert os.path.exists(notebook_path)

    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)

    # Look for key indicators that our static method was properly processed
    found_static_method = False
    unnecessary_import_statement = False
    new_line_before_static_method = False
    annotation_before_static_method = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'simple_static_method' in cell.source:
                found_static_method = True
                if "from test_notebook_decorators" in cell.source:
                    unnecessary_import_statement = True
                # check that the substring "def simple_static_method()" is immediately preceded by a new line
                if cell.source.split("def simple_static_method()")[0].endswith("\n"):
                    new_line_before_static_method = True

                # check that the cell begins with an annotation
                if cell.source.split("def simple_static_method()")[0].startswith("@"):
                    annotation_before_static_method = True
        

    # Verify we found all the expected elements
    assert found_static_method, "Static method not found in notebook"
    assert not unnecessary_import_statement, "Unnecessary import statement found in notebook"
    assert new_line_before_static_method, "New line not found before static method in notebook"
    assert annotation_before_static_method, "Annotation not found before static method in notebook"


# New tests for data loading functionality

@pytest.fixture
def incomplete_graph(tmp_path):
    """Create a graph with both completed and incomplete nodes."""
    def transform_func(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()
    
    def process_func_action(transform: pd.DataFrame) -> str:
        return f"Processed {len(transform)} rows"
    
    graph = (ExecutableGraph()
             .configure_runtime(output_path=tmp_path)
             .create_input_node("data", input_dtype=pd.DataFrame)
        .create_node(
            "transform",
            transform_func,
            predecessors=["data"],
            node_description="Transform the data"
        )
        .create_node(
            "process",
            process_func_action,
            predecessors=["transform"],
            node_description="Process the transformed data"
        )
    )
    
    # Set input data and execute only part of the graph
    input_data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    
    # Execute input and transform nodes manually to simulate partial execution
    data_node = graph.get("data")
    transform_node = graph.get("transform")
    
    data_node.set_input(input_data)
    data_node.execute()
    transform_node.execute()
    
    # Note: process node remains unexecuted, so it should be incomplete
    
    return graph


@pytest.fixture 
def test_data_files(tmp_path):
    """Create test data files in different formats."""
    # Create test DataFrame
    data = pd.DataFrame({
        'id': range(10),
        'value': [i * 2 for i in range(10)],
        'category': ['A', 'B'] * 5
    })
    
    # Save in different formats
    csv_path = tmp_path / "test_data.csv"
    pickle_path = tmp_path / "test_data.pkl"
    
    data.to_csv(csv_path, index=False)
    data.to_pickle(pickle_path)
    
    return {
        'data': data,
        'csv_path': str(csv_path),
        'pickle_path': str(pickle_path)
    }


def test_export_completed_nodes_only(incomplete_graph, tmp_path):
    """Test that only completed nodes are exported (default behavior)."""
    notebook_path = tmp_path / "completed_only.ipynb"
    exporter = NotebookExporter(incomplete_graph)
    exporter.export_notebook(str(notebook_path))
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check that only completed nodes are included
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Should include data and transform (completed)
    assert "data_output" in source_text
    assert "transform_output" in source_text
    
    # Should NOT include process (incomplete)
    assert "process_output" not in source_text


def test_data_loading_csv(sample_graph, test_data_files, tmp_path):
    """Test notebook export with CSV data loading."""
    notebook_path = tmp_path / "with_csv_loading.ipynb"
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['csv_path'],
        loading_function='pd.read_csv'
    )
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Check for data loading cell
    assert "input_data = pd.read_csv(" in source_text
    assert test_data_files['csv_path'] in source_text
    
    # Check for pandas import
    assert "import pandas as pd" in source_text
    
    # Check that input nodes use loaded data
    assert "data_output = input_data" in source_text


def test_data_loading_pickle(sample_graph, test_data_files, tmp_path):
    """Test notebook export with pickle data loading."""
    notebook_path = tmp_path / "with_pickle_loading.ipynb"
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['pickle_path'],
        loading_function='pd.read_pickle'
    )
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Check for data loading cell
    assert "input_data = pd.read_pickle(" in source_text
    assert test_data_files['pickle_path'] in source_text
    
    # Check for pandas import
    assert "import pandas as pd" in source_text


def test_custom_loading_function(sample_graph, test_data_files, tmp_path):
    """Test notebook export with custom loading function."""
    notebook_path = tmp_path / "with_custom_loading.ipynb"
    exporter = NotebookExporter(sample_graph)
    
    # Test with a custom loading function that uses numpy
    custom_loading = "np.loadtxt"
    
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['csv_path'], 
        loading_function=custom_loading
    )
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Check for data loading cell with custom function
    assert f"input_data = {custom_loading}(" in source_text
    
    # Check for numpy import
    assert "import numpy as np" in source_text


def test_data_loading_only_completed_nodes(sample_graph, test_data_files, tmp_path):
    """Test that data loading works with only completed nodes being exported."""
    notebook_path = tmp_path / "data_loading_completed_only.ipynb"
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['csv_path'],
        loading_function='pd.read_csv'
    )
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Should still have data loading
    assert "input_data = pd.read_csv(" in source_text
    assert "import pandas as pd" in source_text


def test_multiple_import_detection(sample_graph, test_data_files, tmp_path):
    """Test that multiple imports are correctly detected and added."""
    notebook_path = tmp_path / "multiple_imports.ipynb"
    exporter = NotebookExporter(sample_graph)
    
    # Custom loading function that would require multiple imports
    custom_loading = "pickle.load(open(path, 'rb')) if path.endswith('.pkl') else pd.read_csv(path)"
    
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['csv_path'],
        loading_function=custom_loading
    )
    
    # Read the notebook  
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Should detect both pandas and pickle imports
    assert "import pandas as pd" in source_text
    assert "import pickle" in source_text


def test_no_data_loading_parameters(sample_graph, tmp_path):
    """Test that notebook works normally when no data loading parameters are provided."""
    notebook_path = tmp_path / "no_data_loading.ipynb"
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(str(notebook_path))
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    source_text = "\n".join(cell.source for cell in nb.cells)
    
    # Should have TODO comments for input nodes
    assert "# TODO: Load your input data" in source_text
    assert "input_data =" not in source_text


def test_data_loading_cell_placement(sample_graph, test_data_files, tmp_path):
    """Test that data loading cell is placed correctly in the notebook."""
    notebook_path = tmp_path / "cell_placement.ipynb"
    exporter = NotebookExporter(sample_graph)
    exporter.export_notebook(
        str(notebook_path),
        data_path=test_data_files['csv_path'],
        loading_function='pd.read_csv'
    )
    
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Find the positions of key cells
    data_loading_pos = None
    imports_pos = None
    first_node_pos = None
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if 'input_data = pd.read_csv(' in cell.source:
                data_loading_pos = i
            elif 'import pandas as pd' in cell.source:
                imports_pos = i
            elif 'data_output' in cell.source and first_node_pos is None:
                first_node_pos = i
    
    # Data loading should come before node processing
    assert data_loading_pos is not None, "Data loading cell not found"
    assert imports_pos is not None, "Imports cell not found"
    assert first_node_pos is not None, "First node cell not found"
    
    # Imports should come before data loading, which should come before nodes
    assert imports_pos < data_loading_pos < first_node_pos