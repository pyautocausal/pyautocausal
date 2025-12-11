"""Data cleaner that executes cleaning plans."""

from typing import Tuple
import pandas as pd
from .base import CleaningPlan, CleaningMetadata


class DataCleaner:
    """Executes cleaning plans on DataFrames.
    
    This is a simple wrapper that makes it easy to use cleaning
    in a PyAutoCausal graph node.
    """
    
    def __init__(self, plan: CleaningPlan):
        """Initialize with a cleaning plan.
        
        Args:
            plan: The cleaning plan to execute
        """
        self.plan = plan
    
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningMetadata]:
        """Execute the cleaning plan on a DataFrame.
        
        Args:
            df: The DataFrame to clean
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning metadata)
        """
        return self.plan(df)
    
    def __call__(self, df: pd.DataFrame, plan: CleaningPlan) -> Tuple[pd.DataFrame, CleaningMetadata]:
        """Make the cleaner callable for use as a node action function.
        
        This allows using the cleaner directly in a PyAutoCausal graph,
        taking both the DataFrame and plan as inputs.
        
        Args:
            df: The DataFrame to clean
            plan: The cleaning plan to execute
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning metadata)
        """
        self.plan = plan
        return self.clean(df) 