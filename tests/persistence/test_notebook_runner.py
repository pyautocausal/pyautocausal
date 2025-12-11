import pytest
import pandas as pd
import tempfile
from pathlib import Path
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.persistence.notebook_runner import (
    run_notebook_and_create_html, 
    convert_notebook_to_html
)
from pyautocausal.orchestration.graph import ExecutableGraph
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


@pytest.fixture
def sample_notebook(tmp_path):
    """Create a simple test notebook."""
    nb = new_notebook()
    
    # Add some cells
    nb.cells.append(new_markdown_cell("# Test Notebook"))
    nb.cells.append(new_code_cell("import pandas as pd\nimport numpy as np"))
    nb.cells.append(new_code_cell("df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})\nprint(df)"))
    nb.cells.append(new_code_cell("result = df['x'].sum()\nprint(f'Sum of x: {result}')"))
    
    # Save notebook
    notebook_path = tmp_path / "test_notebook.ipynb"
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    return notebook_path


@pytest.fixture
def sample_graph_with_data(tmp_path):
    """Create a sample graph that can be exported and run."""
    # Create test data
    test_data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50]
    })
    
    # Save test data
    data_path = tmp_path / "test_data.csv"
    test_data.to_csv(data_path, index=False)
    
    # Create a simple graph
    def simple_transform(data):
        data['z'] = data['x'] + data['y']
        return data
    
    graph = ExecutableGraph()
    graph.configure_runtime(output_path=tmp_path)
    graph.create_input_node("data", input_dtype=pd.DataFrame)
    graph.create_node("transform", simple_transform, predecessors=["data"])
    
    # Execute the graph
    graph.fit(data=test_data)
    
    return graph, data_path


def test_convert_notebook_to_html(sample_notebook):
    """Test converting an existing notebook to HTML."""
    html_path = convert_notebook_to_html(sample_notebook)
    
    assert html_path.exists()
    assert html_path.suffix == '.html'
    
    # Check that HTML file has content
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    assert '<html' in html_content
    assert 'Test Notebook' in html_content


def test_convert_notebook_to_html_custom_path(sample_notebook, tmp_path):
    """Test converting notebook to HTML with custom output path."""
    custom_html_path = tmp_path / "custom_output.html"
    
    result_path = convert_notebook_to_html(sample_notebook, custom_html_path)
    
    assert result_path == custom_html_path
    assert custom_html_path.exists()


def test_export_and_run_to_html_integration(sample_graph_with_data, tmp_path):
    """Test the full integration of exporting a graph and running it to HTML."""
    graph, data_path = sample_graph_with_data
    
    exporter = NotebookExporter(graph)
    
    # Test the export_and_run_to_html method
    notebook_path = tmp_path / "integration_test.ipynb"
    
    try:
        html_path = exporter.export_and_run_to_html(
            notebook_filepath=notebook_path,
            data_path=str(data_path),
            loading_function="pd.read_csv"
        )
        
        # Verify both notebook and HTML were created
        assert notebook_path.exists()
        assert html_path.exists()
        assert html_path.suffix == '.html'
        
        # Check HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        assert '<html' in html_content
        assert 'Causal Analysis Pipeline' in html_content
        
    except ImportError as e:
        if "Jupyter" in str(e):
            pytest.skip("Jupyter not installed, skipping integration test")
        else:
            raise


def test_run_existing_notebook_to_html(sample_graph_with_data, tmp_path):
    """Test running an existing exported notebook to HTML."""
    graph, data_path = sample_graph_with_data
    
    exporter = NotebookExporter(graph)
    
    # First export a notebook
    notebook_path = tmp_path / "existing_notebook.ipynb"
    exporter.export_notebook(
        str(notebook_path),
        data_path=str(data_path),
        loading_function="pd.read_csv"
    )
    
    assert notebook_path.exists()
    
    # Now run it to HTML
    try:
        html_path = exporter.run_existing_notebook_to_html(notebook_path)
        
        assert html_path.exists()
        assert html_path.suffix == '.html'
        
        # Check HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        assert '<html' in html_content
        
    except ImportError as e:
        if "Jupyter" in str(e):
            pytest.skip("Jupyter not installed, skipping test")
        else:
            raise


def test_notebook_runner_error_handling():
    """Test error handling for non-existent files."""
    non_existent_path = Path("non_existent_notebook.ipynb")
    
    with pytest.raises(FileNotFoundError):
        convert_notebook_to_html(non_existent_path)
    
    with pytest.raises(FileNotFoundError):
        run_notebook_and_create_html(non_existent_path)


if __name__ == "__main__":
    # Simple test runner
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create and test sample notebook
        sample_nb = tmp_path / "sample.ipynb"
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("# Test"))
        nb.cells.append(new_code_cell("print('Hello World')"))
        
        with open(sample_nb, 'w') as f:
            nbformat.write(nb, f)
        
        # Test conversion
        try:
            html_path = convert_notebook_to_html(sample_nb)
            print(f"Successfully created HTML: {html_path}")
        except Exception as e:
            print(f"Error: {e}") 