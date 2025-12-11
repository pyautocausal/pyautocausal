"""
Base definitions for estimation functionality in pyautocausal.
"""
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper


@runtime_checkable
class ModelResult(Protocol):
    """Protocol defining the interface for model estimation results."""
    
    def summary(self) -> Any:
        """Return a summary of the estimation results."""
        ...


def format_statsmodels_result(model_result: RegressionResultsWrapper) -> str:
    """
    Format statsmodels regression results as a string.
    
    Args:
        model_result: Fitted statsmodels model result
    
    Returns:
        Formatted string representation of the model results
    """
    return model_result.summary().as_text()