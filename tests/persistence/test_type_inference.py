import pytest
from typing import List, Dict, Any, Union
import pandas as pd
from pyautocausal.persistence.type_inference import infer_output_type
from pyautocausal.persistence.output_types import OutputType

def test_infer_str_type():
    assert infer_output_type(str) == OutputType.TEXT

def test_infer_bytes_type():
    assert infer_output_type(bytes) == OutputType.BINARY

def test_infer_dataframe_type():
    assert infer_output_type(pd.DataFrame) == OutputType.PARQUET

def test_infer_dict_type():
    assert infer_output_type(dict) == OutputType.JSON
    assert infer_output_type(Dict[str, Any]) == OutputType.JSON

def test_infer_list_type():
    assert infer_output_type(list) == OutputType.JSON
    assert infer_output_type(List[int]) == OutputType.JSON

def test_infer_union_type():
    assert infer_output_type(Union[str, int]) == OutputType.TEXT
    assert infer_output_type(Union[dict, list]) == OutputType.JSON

def test_infer_none_type():
    with pytest.raises(ValueError, match="Cannot infer output type for None"):
        infer_output_type(None)
    with pytest.raises(ValueError, match="Cannot infer output type for None"):
        infer_output_type(type(None))

def test_infer_unknown_type_strict_mode():
    class CustomClass:
        pass
        
    with pytest.raises(ValueError, match="Cannot infer output type"):
        infer_output_type(CustomClass)

@pytest.fixture
def custom_class():
    class CustomClass:
        pass
    return CustomClass

def test_infer_unknown_type_non_strict_with_dict(custom_class):
        
    assert infer_output_type(custom_class, strict=False) == OutputType.JSON

def test_infer_unknown_type_non_strict_without_dict():
    class NonDictClass:
        __slots__ = ['x']  # Class without __dict__
        
    with pytest.raises(ValueError, match="Cannot infer output type"):
        infer_output_type(NonDictClass(), strict=False)

def test_infer_numeric_types():
    assert infer_output_type(int) == OutputType.JSON
    assert infer_output_type(float) == OutputType.JSON