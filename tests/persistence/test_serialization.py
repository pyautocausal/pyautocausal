from pyautocausal.persistence.parameter_mapper import make_transformable
import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pyautocausal.persistence.serialization import jsonify, prepare_output_for_saving
from pyautocausal.persistence.output_types import OutputType

@dataclass
class SampleClass:
    name: str
    value: int

@pytest.fixture
def sample_data():
    return {
        'dict': {'a': 1, 'b': 2},
        'list': [1, 2, 3],
        'dataframe': pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
        'custom_obj': SampleClass(name='test', value=42)
    }

def test_jsonify_dict(sample_data):
    result = jsonify(sample_data['dict'])
    assert result == {'a': 1, 'b': 2}

def test_jsonify_list(sample_data):
    result = jsonify(sample_data['list'])
    assert result == [1, 2, 3]

def test_jsonify_dataframe(sample_data):
    result = jsonify(sample_data['dataframe'])
    assert isinstance(result, dict)
    assert 'x' in result
    assert 'y' in result

def test_jsonify_custom_object(sample_data):
    result = jsonify(sample_data['custom_obj'])
    assert result == {'name': 'test', 'value': 42}

def test_jsonify_nested_structure():
    nested = {
        'outer': {
            'inner': [1, 2, {'key': 'value'}]
        }
    }
    result = jsonify(nested)
    assert result == nested

def test_prepare_output_json():
    data = {'test': 'value'}
    result = prepare_output_for_saving(data, OutputType.JSON)
    assert result == data

def test_prepare_output_other_types():
    data = "test string"
    result = prepare_output_for_saving(data, OutputType.TEXT)
    assert result == data 
