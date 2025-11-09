import pytest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

def create_sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25, 30, 35]
    })

def compute_average(df: pd.DataFrame) -> pd.Series:
    """Compute average values by category"""
    return df.groupby('category')['value'].mean()

def create_plot(avg_data: pd.Series) -> Figure:
    """Create a plot visualization of the averages"""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    avg_data.plot(kind='bar', ax=ax)
    plt.xlabel('Category')
    plt.ylabel('Average Value')
    return fig

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing"""
    return create_sample_data()

@pytest.fixture
def pipeline_graph(tmp_path):
    """Create a configured pipeline graph with all nodes"""
    
    def compute_average_action(create_data: pd.DataFrame) -> pd.Series:
        return compute_average(create_data)
    
    def create_plot_action(compute_average: pd.Series) -> Figure:
        return create_plot(compute_average)

    graph = (ExecutableGraph()
             .configure_runtime(output_path=tmp_path / 'outputs')
        .create_node(
            "create_data", 
            create_sample_data,
            save_node=True,
            output_config=OutputConfig(
                output_filename="create_data",
                output_type=OutputType.PARQUET
            )
        )
        .create_node(
            "compute_average",
            compute_average_action,
            predecessors=["create_data"],  # Connect to data node
            save_node=True,
            output_config=OutputConfig(
                output_filename="compute_average",
                output_type=OutputType.CSV
            )
        )
        .create_node(
            "create_plot",
            create_plot_action,
            predecessors=["compute_average"],  # Connect to average node
            save_node=True,
            output_config=OutputConfig(
                output_filename="create_plot",
                output_type=OutputType.PNG
            )
        )
        )
    
    return graph

@pytest.fixture
def executed_pipeline(pipeline_graph):
    """Execute the pipeline and return the graph"""
    pipeline_graph.execute_graph()
    return pipeline_graph

def test_pipeline_execution(executed_pipeline):
    """Test that all nodes complete execution"""
    graph = executed_pipeline
    
    # Get nodes by name
    data_node = [n for n in graph.nodes() if n.name == "create_data"][0]
    average_node = [n for n in graph.nodes() if n.name == "compute_average"][0]
    plot_node = [n for n in graph.nodes() if n.name == "create_plot"][0]
    
    assert data_node.is_completed()
    assert average_node.is_completed()
    assert plot_node.is_completed()

def test_data_node_output(executed_pipeline):
    """Test the output of the data node"""
    graph = executed_pipeline
    
    # Get data node
    data_node = [n for n in graph.nodes() if n.name == "create_data"][0]
    
    df = data_node.get_result_value()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['category', 'value']
    assert len(df) == 6

def test_average_node_output(executed_pipeline):
    """Test the output of the average computation node"""
    graph = executed_pipeline
    
    # Get average node
    average_node = [n for n in graph.nodes() if n.name == "compute_average"][0]
    
    averages = average_node.get_result_value()
    assert isinstance(averages, pd.Series)
    assert len(averages) == 2
    assert all(averages.index == ['A', 'B'])
    assert all(averages > 0)

def test_plot_node_output(executed_pipeline):
    """Test the output of the plot creation node"""
    graph = executed_pipeline
    
    # Get plot node
    plot_node = [n for n in graph.nodes() if n.name == "create_plot"][0]
    
    plot = plot_node.get_result_value()
    assert isinstance(plot, Figure)
    assert isinstance(plot, Figure)

def test_output_files_creation(executed_pipeline, tmp_path):
    """Test that all output files are created correctly"""
    output_dir = tmp_path / 'outputs'
    assert output_dir.exists()
    
    expected_files = [
        'create_data.parquet',
        'compute_average.csv',
        'create_plot.png'
    ]
    
    for filename in expected_files:
        file_path = output_dir / filename
        assert file_path.exists(), f"Expected output file {filename} not found"
        assert file_path.stat().st_size > 0, f"Output file {filename} is empty"

def test_output_files_content(executed_pipeline, tmp_path, sample_data):
    """Test the content of output files"""
    output_dir = tmp_path / 'outputs'
    
    # Test parquet file
    df = pd.read_parquet(output_dir / 'create_data.parquet')
    pd.testing.assert_frame_equal(df, sample_data)
    
    # Test CSV file
    averages = pd.read_csv(output_dir / 'compute_average.csv', index_col=0)
    assert len(averages) == 2
    assert all(averages.index == ['A', 'B'])
    
    # Test PNG file
    with open(output_dir / 'create_plot.png', 'rb') as f:
        plot_data = f.read()
    assert len(plot_data) > 0

