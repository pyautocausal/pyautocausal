import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from io import BytesIO
from pyautocausal.persistence.local_output_handler import LocalOutputHandler
from pyautocausal.persistence.output_types import OutputType
from pyautocausal.persistence.output_handler import UnsupportedOutputTypeError

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return tmp_path / "test_outputs"

@pytest.fixture
def handler(output_dir):
    """Create a LocalOutputHandler instance"""
    return LocalOutputHandler(output_dir)

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame"""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })

@pytest.fixture
def sample_plot_bytes():
    """Create a sample plot and return its bytes"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig

def test_save_csv(handler, sample_dataframe, output_dir):
    handler.save("test_data", sample_dataframe, OutputType.CSV)
    output_file = output_dir / "test_data.csv"
    
    assert output_file.exists()
    loaded_df = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_parquet(handler, sample_dataframe, output_dir):
    handler.save("test_data", sample_dataframe, OutputType.PARQUET)
    output_file = output_dir / "test_data.parquet"
    
    assert output_file.exists()
    loaded_df = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_json_dataframe(handler, sample_dataframe, output_dir):
    handler.save("test_data", sample_dataframe, OutputType.JSON)
    output_file = output_dir / "test_data.json"
    
    assert output_file.exists()
    loaded_df = pd.read_json(output_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_json_dict(handler, output_dir):
    test_dict = {"name": "test", "value": 42}
    handler.save("test_data", test_dict, OutputType.JSON)
    output_file = output_dir / "test_data.json"
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        loaded_dict = json.load(f)
    assert loaded_dict == test_dict

def test_save_pickle(handler, sample_dataframe, output_dir):
    handler.save("test_data", sample_dataframe, OutputType.PICKLE)
    output_file = output_dir / "test_data.pkl"
    
    assert output_file.exists()
    loaded_df = pd.read_pickle(output_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_save_text(handler, output_dir):
    test_text = "Hello, world!"
    handler.save("test_data", test_text, OutputType.TEXT)
    output_file = output_dir / "test_data.txt"
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        loaded_text = f.read()
    assert loaded_text == test_text

def test_save_png(handler, sample_plot_bytes, output_dir):
    handler.save("test_plot", sample_plot_bytes, OutputType.PNG)
    output_file = output_dir / "test_plot.png"
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_save_bytes(handler, output_dir):
    test_bytes = b"Hello, world!"
    handler.save("test_data", test_bytes, OutputType.BINARY)
    output_file = output_dir / "test_data.bytes"
    
    assert output_file.exists()
    with open(output_file, 'rb') as f:
        loaded_bytes = f.read()
    assert loaded_bytes == test_bytes

def test_invalid_output_type(handler):
    with pytest.raises(UnsupportedOutputTypeError):
        handler.save("test_data", "some data", "invalid_type")

def test_type_validation_csv(handler):
    with pytest.raises(TypeError):
        handler.save("test_data", "not a dataframe", OutputType.CSV)

def test_type_validation_parquet(handler):
    with pytest.raises(TypeError):
        handler.save("test_data", "not a dataframe", OutputType.PARQUET)

def test_type_validation_text(handler):
    with pytest.raises(TypeError):
        handler.save("test_data", 123, OutputType.TEXT)

def test_type_validation_png(handler):
    with pytest.raises(TypeError):
        handler.save("test_data", "not bytes", OutputType.PNG)

def test_nested_output_directory(output_dir):
    nested_dir = output_dir / "nested" / "path"
    handler = LocalOutputHandler(nested_dir)
    
    test_text = "Hello, world!"
    handler.save("test_data", test_text, OutputType.TEXT)
    output_file = nested_dir / "test_data.txt"
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        loaded_text = f.read()
    assert loaded_text == test_text
