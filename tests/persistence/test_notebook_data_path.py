#!/usr/bin/env python3
"""
Test script to verify that the generated notebook can load data correctly
when run from its own directory with relative paths.
"""

import os
import sys
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from pyautocausal.persistence.notebook_export import NotebookExporter
from pyautocausal.orchestration.graph import ExecutableGraph


def test_notebook_relative_data_path():
    """Test that the notebook uses relative paths correctly for data loading."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create a simple graph
        graph = ExecutableGraph()
        graph.create_input_node("data", input_dtype=pd.DataFrame)
        
        # Create test data
        test_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        
        # Save test data to the same directory where notebook will be saved
        data_file = tmp_path / "test_data.csv"
        test_data.to_csv(data_file, index=False)
        
        # Export notebook with data loading
        notebook_file = tmp_path / "test_notebook.ipynb"
        exporter = NotebookExporter(graph)
        exporter.export_notebook(
            str(notebook_file),
            data_path="test_data.csv",  # Just the filename, not full path
            loading_function='pd.read_csv'
        )
        
        # Verify the notebook file was created
        assert notebook_file.exists()
        
        # Read the notebook and check the data loading cell
        with open(notebook_file, 'r') as f:
            notebook_content = f.read()
        
        # Should contain relative path, not absolute path
        assert "input_data = pd.read_csv('test_data.csv')" in notebook_content
        assert str(data_file) not in notebook_content  # Should not contain absolute path
        
        # Test that the data can actually be loaded from the notebook's perspective
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # This simulates what happens when the notebook is run from its directory
            loaded_data = pd.read_csv("test_data.csv")
            assert loaded_data.shape == test_data.shape
            assert list(loaded_data.columns) == list(test_data.columns)
            
        finally:
            os.chdir(original_cwd)


def test_notebook_absolute_vs_relative_path():
    """Test that absolute paths are converted to relative when appropriate."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create a simple graph
        graph = ExecutableGraph()
        graph.create_input_node("data", input_dtype=pd.DataFrame)
        
        # Create test data
        test_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        data_file = tmp_path / "data.csv"
        test_data.to_csv(data_file, index=False)
        
        # Test with just filename (should stay as is)
        notebook_file = tmp_path / "notebook.ipynb"
        exporter = NotebookExporter(graph)
        exporter.export_notebook(
            str(notebook_file),
            data_path="data.csv",  # Just filename
            loading_function='pd.read_csv'
        )
        
        with open(notebook_file, 'r') as f:
            content = f.read()
        
        # Should use the simple filename
        assert "input_data = pd.read_csv('data.csv')" in content


if __name__ == "__main__":
    test_notebook_relative_data_path()
    test_notebook_absolute_vs_relative_path()
    print("✅ All tests passed!") 