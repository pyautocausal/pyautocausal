"""Data cleaning planner that creates cleaning plans from validation results."""

from typing import List, Optional
from ..data_validation.validator_base import AggregatedValidationResult
from .base import CleaningPlan, CleaningOperation
from .operations import get_all_operations


class DataCleaningPlanner:
    """Creates cleaning plans based on validation results.
    
    The planner:
    1. Collects all cleaning hints from validation results
    2. Matches hints to available cleaning operations
    3. Orders operations by priority
    4. Returns a callable CleaningPlan
    """
    
    def __init__(self, 
                 validation_results: AggregatedValidationResult,
                 operations: Optional[List[CleaningOperation]] = None):
        """Initialize the planner.
        
        Args:
            validation_results: Results from data validation
            operations: List of available cleaning operations. If None, uses all built-in operations.
        """
        self.validation_results = validation_results
        self.operations = operations or get_all_operations()
    
    def create_plan(self) -> CleaningPlan:
        """Create a cleaning plan from validation results.
        
        Returns:
            A CleaningPlan that can be called on a DataFrame
        """
        plan = CleaningPlan(validation_results=self.validation_results)
        
        # Collect all cleaning hints from validation results
        all_hints = []
        for result in self.validation_results.individual_results:
            all_hints.extend(result.cleaning_hints)
        
        # Match hints to operations
        for hint in all_hints:
            for operation in self.operations:
                if operation.can_apply(hint):
                    plan.add_operation(operation, hint)
                    break  # Only apply first matching operation per hint
        
        # Sort operations by priority
        plan.sort_operations()
        
        return plan
    
    def __call__(self, validation_results: AggregatedValidationResult) -> CleaningPlan:
        """Make the planner callable for use as a node action function.
        
        This allows using the planner directly in a PyAutoCausal graph.
        """
        self.validation_results = validation_results
        return self.create_plan() 