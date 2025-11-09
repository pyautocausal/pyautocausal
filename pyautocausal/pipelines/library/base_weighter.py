import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from abc import ABC, abstractmethod

class BaseWeighter(ABC):
    """
    Base class for all weighting methods in the pyautocausal package.
    
    This abstract base class defines the common interface that all weighting
    methods should implement, including methods for weight computation and
    diagnostics output.
    """
    
    @classmethod
    @abstractmethod
    def compute_weights(cls, df: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute weights for balancing treatment and control groups.
        
        Args:
            df: DataFrame containing the data for weight computation
            **kwargs: Additional method-specific parameters
                
        Returns:
            Tuple containing:
            - Array of weights (one per observation)
            - Dictionary with diagnostic information
        """
        pass
    
    @classmethod
    @abstractmethod
    def output(cls, diagnostics: Dict) -> str:
        """
        Format the weighting diagnostics into a readable string.
        
        Args:
            diagnostics: Dictionary with diagnostic information from compute_weights
                
        Returns:
            Formatted string with weighting diagnostics
        """
        pass
    
    @classmethod
    def action(cls, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply the weighting method and return the weighted data and diagnostics.
        
        This default implementation calls compute_weights and returns the data
        with weighted observations only (those with non-zero weights).
        
        Args:
            df: DataFrame containing the data to be weighted
            **kwargs: Additional method-specific parameters
                
        Returns:
            Tuple containing:
            - DataFrame with only weighted observations
            - Dictionary with diagnostic information
        """
        weights, diagnostics = cls.compute_weights(df, **kwargs)
        
        # Keep only observations with non-zero weights
        weighted_df = df.copy()[weights > 0]
        
        # Add weights column
        weighted_df['weight'] = weights[weights > 0]
        
        return weighted_df, diagnostics 