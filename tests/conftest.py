import pytest
import pandas as pd
from pathlib import Path
from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
from pyautocausal.pipelines.library.estimators import fit_double_lasso, fit_ols
from pyautocausal.pipelines.library.output import write_linear_models_to_summary


@pytest.fixture
def lalonde_data():
    """Returns the lalonde dataset for integration tests."""
    # Get the directory where this conftest.py file is located
    test_dir = Path(__file__).parent
    fixture_path = test_dir / "fixtures" / "lalonde.csv"
    return pd.read_csv(fixture_path)

@pytest.fixture
def minimum_wage_data():
    """Returns the minimum wage dataset for integration tests."""
    # Get the directory where this conftest.py file is located
    test_dir = Path(__file__).parent
    fixture_path = test_dir / "fixtures" / "mpdta.csv"
    return pd.read_csv(fixture_path)


@pytest.fixture
def output_dir(tmp_path):
    """Creates a temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture
def causal_graph(tmp_path):
    """
    Fixture that provides a sample causal graph for testing.
    
    Args:
        tmp_path: Built-in pytest fixture providing a temporary directory
        
    Returns:
        ExecutableGraph: A configured causal graph ready for testing
    """
    output_dir = tmp_path / "causal_output"
    output_dir.mkdir(exist_ok=True)
    
    graph = (ExecutableGraph()
             .configure_runtime(output_path=str(output_dir))
        .create_input_node("data", input_dtype=pd.DataFrame)
        .create_node(
            "spec",
            action_function=create_cross_sectional_specification.get_function(),
            predecessors=["data"],
        )
        .create_decision_node(
            "doubleml_condition",
            condition=lambda x: len(x.data) > 100,
            predecessors=["spec"],
        )
        .create_node(
            "doubleml",
            fit_double_lasso.get_function(),
            predecessors=["doubleml_condition"],
        )
        .when_true("doubleml_condition", "doubleml")
        .create_node(
            "ols",  
            fit_ols.get_function(),
            predecessors=["doubleml_condition"],
        ) 
        .when_false("doubleml_condition", "ols")
        .create_node(
            "doubleml_summary",
            write_linear_models_to_summary.transform(arg_mapping={"doubleml": "res"}),
            predecessors=["doubleml"],
            save_node=True,
        )
        .create_node(
            "ols_summary",
            write_linear_models_to_summary.transform(arg_mapping={"ols": "res"}),
            predecessors=["ols"],
            save_node=True,
        )
    )
    return graph 