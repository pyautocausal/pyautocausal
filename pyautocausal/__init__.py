"""PyAutoCausal: Automated Causal Inference Pipelines

PyAutoCausal automates the complex decision tree of modern causal inference methods.
Instead of manually implementing and choosing between dozens of estimators, PyAutoCausal:

1. Analyzes your data structure to understand treatment timing, units, and available controls
2. Selects appropriate methods based on your data characteristics  
3. Validates assumptions and warns about potential violations
4. Executes analysis with proper statistical inference
5. Exports results in formats ready for stakeholder communication

Quick Start:
    >>> from pyautocausal import create_panel_graph
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> 
    >>> # For panel data (multiple time periods)
    >>> pipeline = create_panel_graph(Path("./output"))
    >>> results = pipeline.fit(df=your_panel_data)
    >>> 
    >>> # For cross-sectional data (single time period)
    >>> from pyautocausal import create_cross_sectional_graph
    >>> pipeline = create_cross_sectional_graph(Path("./output"))
    >>> results = pipeline.fit(df=your_cross_sectional_data)

Main Components:
    - create_panel_graph: Creates pipelines for panel data (multiple time periods)
    - create_cross_sectional_graph: Creates pipelines for cross-sectional data
    - ExecutableGraph: Core orchestration framework for building custom pipelines
    - AutoCleaner: Automated data cleaning and validation
"""

__version__ = "0.1.0"
__author__ = "Nicholas Topousis, Yogam Tchokni"
__email__ = "nicholas.topousis@example.com"

# Core pipeline functionality - main user entry points
from .pipelines.example_graph import (
    create_panel_graph,
    create_cross_sectional_graph, 
    simple_graph,
    export_outputs
)

# Core orchestration framework for advanced users
from .orchestration.graph import ExecutableGraph

# Data cleaning and validation
from .data_cleaner_interface.autocleaner import AutoCleaner

# Output and persistence utilities
from .persistence.output_config import OutputConfig, OutputType
from .persistence.local_output_handler import LocalOutputHandler

__all__ = [
    # Main pipeline entry points
    "create_panel_graph",
    "create_cross_sectional_graph",
    "simple_graph",
    "export_outputs",
    
    # Core framework
    "ExecutableGraph", 
    
    # Data utilities
    "AutoCleaner",
    
    # Output utilities
    "OutputConfig",
    "OutputType", 
    "LocalOutputHandler",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]
