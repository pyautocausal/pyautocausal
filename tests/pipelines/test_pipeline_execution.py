"""Execution tests for the refactored example graph.

This test module verifies that the refactored example graph:
1. Actually executes successfully end-to-end
2. Creates the expected output files
3. Routes data through different analysis branches correctly
4. Produces valid results
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
import os

from pyautocausal.pipelines.example_graph import create_cross_sectional_graph, create_panel_graph
from pyautocausal.pipelines.mock_data import generate_mock_data


class TestExampleGraphExecution:
    """Test suite for actual execution of the refactored example graph."""

    def test_cross_sectional_graph_execution(self):
        """Test the cross-sectional graph with single period data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate single period data
            data = generate_mock_data(n_units=100, n_periods=1, n_treated=50, staggered_treatment=False)
            
            # Create and execute cross-sectional graph
            graph = create_cross_sectional_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for the cross-sectional path
            expected_nodes = {'cross_sectional_cleaned_data', 'basic_cleaning', 'df', 'ols_stand', 'ols_stand_output'}
            assert expected_nodes.issubset(completed_nodes)
            
            print(f"✓ Cross-sectional graph executed successfully: {len(completed_nodes)} nodes completed")

    def test_panel_graph_execution(self):
        """Test the panel graph with multi-period data and no staggered treatment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate panel data
            data = generate_mock_data(n_units=50, n_periods=3, n_treated=25, staggered_treatment=False)
            
            # Create and execute panel graph
            graph = create_panel_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for a simple panel data path
            expected_nodes = {'save_event_output', 'stag_treat', 'ols_event', 'panel_cleaned_data', 'multi_post_periods', 'single_treated_unit', 'df', 'basic_cleaning', 'event_spec', 'event_plot'}
            assert expected_nodes.issubset(completed_nodes)

            # check that the results were saved in the new directory structure
            assert set(os.listdir(output_path)) == {'plots', 'text'}
            assert (output_path / "text" / "save_event_output.txt").exists()
            assert (output_path / "plots" / "event_study_plot.png").exists()
            
            print(f"✓ Panel graph executed successfully: {len(completed_nodes)} nodes completed")
    
    
    def test_staggered_treatment_in_panel_graph(self, minimum_wage_data):
        """Test the staggered treatment path within the panel graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Generate staggered treatment data
            raw_data = minimum_wage_data
            data = raw_data.rename(columns={
                "countyreal": "id_unit",  # county identifier
                "year": "t",              # time variable
                "lemp": "y"               # log employment as outcome
            })

            # Create proper treatment indicator based on first.treat timing
            def reconstruct_treatment(row):
                """Reconstruct treatment: 0 before first.treat, 1 from first.treat onwards"""
                if pd.isna(row['first.treat']) or row['first.treat'] == 0:
                    # Never treated units
                    return 0
                elif row['t'] >= row['first.treat']:
                    # Treated in current period (treatment started)
                    return 1
                else:
                    # Not yet treated (before treatment start)
                    return 0

            data['treat'] = data.apply(reconstruct_treatment, axis=1)

            # Keep additional covariates
            data = data[["id_unit", "t", "treat", "y", "lpop"]]  # Include lpop as covariate
            
            # Create and execute panel graph
            graph = create_panel_graph(output_path)
            graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
            
            assert not failed_nodes, f"Failed nodes: {failed_nodes}"
            
            # Expected nodes for the staggered treatment path (updated based on actual execution)
            expected_nodes = {'stag_spec', 'panel_cleaned_data', 'stag_spec_balance_plot', 'basic_cleaning', 'single_treated_unit', 'stag_spec_balance', 'ols_stag', 'has_never_treated', 'cs_never_treated', 'cs_never_treated_plot', 'cs_never_treated_group_plot', 'stag_event_plot', 'save_stag_output', 'df', 'stag_spec_balance_table', 'stag_treat', 'multi_post_periods'}
            assert expected_nodes.issubset(completed_nodes)
            
            # export graph to notebook
            from pyautocausal.persistence.notebook_export import NotebookExporter
            exporter = NotebookExporter(graph)
            exporter.export_notebook(str(output_path  / "panel_execution.ipynb"))
            
            assert (output_path / "panel_execution.ipynb").exists()
            
            print(f"✓ Staggered treatment path in panel graph executed successfully: {len(completed_nodes)} nodes completed")


    def test_synthetic_did_branch(self):
        """Test Synthetic DiD branch (multiple periods, single treated unit)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
    
        # Create data with single treated unit (triggers Synthetic DiD branch)
        data = generate_mock_data(n_units=50, n_periods=5, n_treated=1)
        
        graph = create_panel_graph(output_path)
        graph.fit(df=data)
        
        
        
        # Verify nodes completed
        completed_nodes = [node.name for node in graph.nodes() 
                        if hasattr(node, 'state') and node.state.name == 'COMPLETED']
        failed_nodes = [node.name for node in graph.nodes() 
                    if hasattr(node, 'state') and node.state.name == 'FAILED']
        
        # Ensure no nodes failed
        assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"
        
        # Check for the new directory structure
        assert set(os.listdir(output_path)) == {'plots', 'text'}
        assert (output_path / "plots" / "synthdid_plot.png").exists()


    def test_node_states_in_cross_sectional_graph(self):
        """Test that node states are consistent in the cross-sectional graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            data = generate_mock_data(n_units=50, n_periods=1, n_treated=25, staggered_treatment=False)
            
            graph = create_cross_sectional_graph(output_path)
            result = graph.fit(df=data)
            
            completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
            passed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'PASSED'}
            failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}

            assert not failed_nodes
            assert len(completed_nodes) > 5
            # In a single-path graph, we might not have 'PASSED' nodes unless there are internal decisions
            # For the cross-sectional graph, we expect most nodes to complete.
            
            print(f"✓ Cross-sectional graph node states consistent: {len(completed_nodes)} completed, {len(passed_nodes)} skipped.")

    def test_standard_did_branch(self):
        """Test standard DiD branch (multiple periods, multiple treated units, insufficient post periods)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
        
        # Create data with 2 periods only (insufficient for staggered treatment)
        data = generate_mock_data(n_units=100, n_periods=2, n_treated=50)
        
        graph = create_panel_graph(output_path)
        graph.fit(df=data)
        
        # Check that standard DiD files are generated in the new directory structure
        assert set(os.listdir(output_path)) == {'plots', 'text'}
        assert (output_path / "text" / "save_ols_did.txt").exists()
        
        # Verify nodes completed
        completed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'COMPLETED'}
        failed_nodes = {node.name for node in graph.nodes() if hasattr(node, 'state') and node.state.name == 'FAILED'}
        
        # Expected nodes for standard DiD branch (including balance test nodes)
        expected_nodes = {'did_spec', 'did_spec_balance', 'did_spec_balance_table', 
                        'did_spec_balance_plot', 'ols_did', 'save_ols_did', 'single_treated_unit', 
                        'basic_cleaning', 'multi_post_periods', 'panel_cleaned_data', 'df'}
        assert completed_nodes == expected_nodes

        # Ensure no nodes failed
        assert len(failed_nodes) == 0, f"The following nodes failed: {failed_nodes}"