from typing import Any, Union, Dict, List, _GenericAlias
import pandas as pd
import matplotlib.pyplot as plt
from .output_types import OutputType

def infer_output_type(return_type: Any, strict: bool = True) -> OutputType:
    """
    Infer the output type from a function's return type annotation
    
    Args:
        return_type: The type to infer from
        strict: If True, raise error for unknown types. If False, attempt to use JSON
               for objects with __dict__ attribute, but still error on other unknown types.
    
    Raises:
        ValueError: If type cannot be inferred, or if in non-strict mode and type lacks __dict__
    """
    # Error on None type
    if return_type is None or return_type is type(None):
        raise ValueError("Cannot infer output type for None. If you need to persist None values, please specify an output type explicitly.")
        
    type_mapping = {
        pd.DataFrame: OutputType.PARQUET,
        pd.Series: OutputType.CSV,
        str: OutputType.TEXT,
        int: OutputType.JSON,
        float: OutputType.JSON,
        bytes: OutputType.BINARY,
        plt.Figure: OutputType.PNG,
        dict: OutputType.JSON,
        list: OutputType.JSON,
    }
    
    # Handle Union types
    if hasattr(return_type, "__origin__") and return_type.__origin__ is Union:
        return_type = return_type.__args__[0]
    
    # Handle generic types (Dict, List)
    if isinstance(return_type, _GenericAlias):
        if return_type.__origin__ in (dict, Dict):
            return OutputType.JSON
        if return_type.__origin__ in (list, List):
            return OutputType.JSON
    
    # Check if it's a dictionary type
    if (isinstance(return_type, type) and 
        issubclass(return_type, dict)) or return_type is dict:
        return OutputType.JSON
    
    # Check if it's a list type
    if (isinstance(return_type, type) and 
        issubclass(return_type, list)) or return_type is list:
        return OutputType.JSON
    
    # Check if it's an sklearn-like estimator
    if hasattr(return_type, 'fit') and hasattr(return_type, 'predict'):
        return OutputType.PICKLE
    
    result = type_mapping.get(return_type)
    if result is not None:
        return result
        
    if not strict:
        if isinstance(return_type, type) and hasattr(return_type, '__dict__'):
            return OutputType.JSON
    
    raise ValueError(f"Cannot infer output type for {return_type}. Please specify an output type explicitly.") 