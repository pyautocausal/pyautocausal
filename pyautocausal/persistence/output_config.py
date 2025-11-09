from dataclasses import dataclass
from typing import Optional
from .output_types import OutputType

@dataclass
class OutputConfig:
    """Configuration for node output persistence"""
    output_type: OutputType  # Must be specified
    output_filename: Optional[str] = None  # If None and save_output is True, use node name
    
    def __post_init__(self):
        """Validate the output configuration after initialization"""
        if self.output_type is None:
            raise ValueError(
                "output_type must be specified when creating an OutputConfig. "
                "Choose from: TEXT, CSV, PARQUET, PNG, BINARY, JSON, PICKLE"
            )