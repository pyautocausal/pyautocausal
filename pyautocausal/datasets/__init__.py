"""PyAutoCausal Example Datasets

This module provides easy access to example datasets for causal inference demonstrations.

Usage:
    from examples.data import minimum_wage, california_prop99, lalonde
    
    df = minimum_wage
    graph.fit(df=df)
"""
import pandas as pd
from pathlib import Path

# Get the directory where this file is located
_data_dir = Path(__file__).parent

# Load all available datasets
minimum_wage = pd.read_csv(_data_dir / "minimum_wage.csv")
california_prop99 = pd.read_csv(_data_dir / "california_prop99.csv")
lalonde = pd.read_csv(_data_dir / "lalonde.csv")
mpdta = pd.read_csv(_data_dir / "mpdta.csv")

__all__ = [
    "minimum_wage",
    "california_prop99", 
    "lalonde",
    "mpdta",
    "PENN"
]

