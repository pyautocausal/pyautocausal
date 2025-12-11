from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Set
import pandas as pd
from .output_types import OutputType
from matplotlib.figure import Figure
from ..orchestration.result import Result
class UnsupportedOutputTypeError(Exception):
    """Raised when attempting to save an unsupported output type"""
    pass

class OutputHandler(ABC):
    """Abstract base class for output handlers"""
    
    # Class variable mapping output types to required save methods
    SAVE_METHOD_MAP: ClassVar[dict[OutputType, str]] = {
        OutputType.CSV: 'save_csv',
        OutputType.PARQUET: 'save_parquet',
        OutputType.JSON: 'save_json',
        OutputType.PICKLE: 'save_pickle',
        OutputType.TEXT: 'save_text',
        OutputType.PNG: 'save_png',
        OutputType.BINARY: 'save_bytes'
    }
    
    def __init__(self):
        # Verify subclass implements all required save methods
        self._verify_save_methods()
    
    def _verify_save_methods(self):
        """Verify that all required save methods are implemented"""
        missing_methods = []
        for method_name in self.SAVE_METHOD_MAP.values():
            if not hasattr(self, method_name) or not callable(getattr(self, method_name)):
                missing_methods.append(method_name)
        
        if missing_methods:
            raise NotImplementedError(
                f"Output handler must implement the following methods: {', '.join(missing_methods)}"
            )
    
    def save(self, name: str, data: Any, output_type: OutputType):
        """
        Save data with the given name and type.
        
        Args:
            name: Base filename without extension
            data: The data to save
            output_type: The OutputType enum specifying the format
            
        Raises:
            UnsupportedOutputTypeError: If the output type is not supported
            TypeError: If the data is not of the correct type for the specified format
            RuntimeError: If there's an error during the save operation
        """             

        if not isinstance(output_type, OutputType):
            raise UnsupportedOutputTypeError(
                f"Output type {output_type} is not supported. "
                f"Supported types are: {', '.join(OutputType.get_supported_extensions())}"
            )
        
        # Get the appropriate save method
        save_method = getattr(self, self.SAVE_METHOD_MAP[output_type])
        
        # Validate data type before saving
        self._validate_data_type(data, output_type)
        
        # Call the appropriate save method
        try:
            save_method(name, data)
        except Exception as e:
            raise RuntimeError(f"Failed to save {name}: {str(e)}")
    
    def _validate_data_type(self, data: Any, output_type: OutputType):
        """Validate that the data is of the correct type for the output format"""
        if output_type == OutputType.CSV:
            if not isinstance(data, (pd.DataFrame, pd.Series)):
                raise TypeError("Data must be DataFrame or Series for CSV output")
        elif output_type == OutputType.PARQUET:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be DataFrame for Parquet output")
        elif output_type == OutputType.TEXT:
            if not isinstance(data, str):
                raise TypeError("Data must be string for text output")
        elif output_type == OutputType.BINARY:
            if not isinstance(data, bytes):
                raise TypeError("Data must be in bytes format")
        elif output_type == OutputType.PNG:
            if not isinstance(data, Figure):
                raise TypeError("Data must be a matplotlib Figure for PNG output")
        elif output_type == OutputType.JSON:
            if not isinstance(data, dict) and not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be dict or DataFrame for JSON output")
    
    @abstractmethod
    def save_json(self, name: str, data: Any):
        """Save data as JSON"""
        pass
    
    @abstractmethod
    def save_csv(self, name: str, data: Any):
        """Save data as CSV"""
        pass
    
    @abstractmethod
    def save_parquet(self, name: str, data: Any):
        """Save data as Parquet"""
        pass
    
    @abstractmethod
    def save_json(self, name: str, data: Any):
        """Save data as JSON"""
        pass
    
    @abstractmethod
    def save_pickle(self, name: str, data: Any):
        """Save data as pickle"""
        pass
    
    @abstractmethod
    def save_text(self, name: str, data: Any):
        """Save data as text"""
        pass
    
    @abstractmethod
    def save_png(self, name: str, data: Any):
        """Save data as PNG"""
        pass
    
    @abstractmethod
    def save_bytes(self, name: str, data: Any):
        """Save raw bytes"""
        pass