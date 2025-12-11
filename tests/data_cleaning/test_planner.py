"""Tests for the data cleaning planner."""

import pytest
import pandas as pd
from datetime import datetime

from pyautocausal.data_validation.base import (
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)
from pyautocausal.data_cleaning.hints import (
    CleaningHint,
    UpdateColumnTypesHint,
    InferCategoricalHint,
    DropMissingRowsHint,
    EncodeMissingAsCategoryHint,
)
from dataclasses import dataclass
from pyautocausal.data_validation.validator_base import AggregatedValidationResult
from pyautocausal.data_cleaning.planner import DataCleaningPlanner
from pyautocausal.data_cleaning.base import CleaningOperation, TransformationRecord
from pyautocausal.data_cleaning.operations import (
    UpdateColumnTypesOperation,
    ConvertToCategoricalOperation,
    DropMissingRowsOperation,
    EncodeMissingAsCategoryOperation,
)


@dataclass
class MockHint(CleaningHint):
    """Mock hint for testing."""
    hint_type: str
    test_priority: int
    
    @property
    def priority(self) -> int:
        return self.test_priority


class MockOperation(CleaningOperation):
    """Mock operation for testing."""
    
    def __init__(self, name: str, priority: int, can_apply_to: str):
        self._name = name
        self._priority = priority
        self._can_apply_to = can_apply_to
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def priority(self) -> int:
        return self._priority
    
    def can_apply(self, hint: CleaningHint) -> bool:
        return isinstance(hint, MockHint) and hint.hint_type == self._can_apply_to
    
    def apply(self, df: pd.DataFrame, hint: CleaningHint):
        return df, TransformationRecord(
            operation_name=self.name,
            timestamp=datetime.now(),
            details={"mock": True}
        )


class TestDataCleaningPlanner:
    """Test the DataCleaningPlanner."""
    
    def test_empty_validation_results(self):
        """Test planner with no validation results."""
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 0
        assert "No cleaning operations planned" in plan.describe()
    
    def test_single_cleaning_hint(self):
        """Test planner with a single cleaning hint."""
        hint = UpdateColumnTypesHint(
            type_mapping={"status": "category", "category": "category"}
        )
        
        validation_result = DataValidationResult(
            check_name="column_types",
            passed=True,
            cleaning_hints=[hint]
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 1
        operation, matched_hint = plan.operations[0]
        assert isinstance(operation, UpdateColumnTypesOperation)
        assert matched_hint == hint
    
    def test_multiple_hints_different_operations(self):
        """Test planner with multiple hints for different operations."""
        hints = [
            InferCategoricalHint(
                target_columns=["col1"],
                threshold=10,
                unique_counts={"col1": 5}
            ),
            DropMissingRowsHint(
                target_columns=["col2"]
            ),
            EncodeMissingAsCategoryHint(
                target_columns=["col3"]
            )
        ]
        
        validation_result = DataValidationResult(
            check_name="test_check",
            passed=True,
            cleaning_hints=hints
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 3
        
        # Check operations are in priority order
        operation_names = [op[0].name for op in plan.operations]
        assert operation_names[0] == "encode_missing_as_category"  # priority 85
        assert operation_names[1] == "convert_to_categorical"  # priority 80
        assert operation_names[2] == "drop_missing_rows"  # priority 20
    
    def test_priority_ordering(self):
        """Test that operations are ordered by priority."""
        operations = [
            MockOperation("low_priority", 10, "type_a"),
            MockOperation("high_priority", 100, "type_b"),
            MockOperation("medium_priority", 50, "type_c"),
        ]
        
        hints = [
            MockHint(hint_type="type_a", test_priority=10),
            MockHint(hint_type="type_b", test_priority=100),
            MockHint(hint_type="type_c", test_priority=50),
        ]
        
        validation_result = DataValidationResult(
            check_name="test",
            passed=True,
            cleaning_hints=hints
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results, operations=operations)
        plan = planner.create_plan()
        
        operation_names = [op[0].name for op in plan.operations]
        assert operation_names == ["high_priority", "medium_priority", "low_priority"]
    
    def test_unmatched_hints_ignored(self):
        """Test that hints without matching operations are ignored."""
        hints = [
            InferCategoricalHint(target_columns=["col1"], threshold=10, unique_counts={}),
            MockHint(hint_type="unknown_operation", test_priority=50),  # No matching operation
            DropMissingRowsHint(target_columns=["col3"]),
        ]
        
        validation_result = DataValidationResult(
            check_name="test",
            passed=True,
            cleaning_hints=hints
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        # Should only have 2 operations (unknown_operation is ignored)
        assert len(plan.operations) == 2
        operation_types = [op[0].name for op in plan.operations]
        assert "convert_to_categorical" in operation_types
        assert "drop_missing_rows" in operation_types
    
    def test_hints_from_multiple_validation_results(self):
        """Test collecting hints from multiple validation results."""
        result1 = DataValidationResult(
            check_name="check1",
            passed=True,
            cleaning_hints=[
                InferCategoricalHint(target_columns=["col1"], threshold=10, unique_counts={})
            ]
        )
        
        result2 = DataValidationResult(
            check_name="check2",
            passed=True,  # Both validations pass
            issues=[ValidationIssue(
                severity=ValidationSeverity.INFO,  # Only INFO level
                message="Some info"
            )],
            cleaning_hints=[
                DropMissingRowsHint(target_columns=["col2"])
            ]
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,  # Overall passed
            individual_results=[result1, result2]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 2
    
    def test_plan_preserves_validation_results(self):
        """Test that plan maintains reference to validation results."""
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert plan.validation_results is validation_results
    
    def test_planner_as_callable(self):
        """Test using planner as a callable (for graph nodes)."""
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[
                DataValidationResult(
                    check_name="test",
                    passed=True,
                    cleaning_hints=[
                        InferCategoricalHint(target_columns=["col1"], threshold=10, unique_counts={})
                    ]
                )
            ]
        )
        
        planner = DataCleaningPlanner(validation_results)
        
        # Test callable interface
        plan = planner(validation_results)
        
        assert len(plan.operations) == 1
        assert isinstance(plan.operations[0][0], ConvertToCategoricalOperation)
    
    def test_plan_description(self):
        """Test that plan description is informative."""
        hints = [
            UpdateColumnTypesHint(
                type_mapping={"status": "category", "category": "category"}
            ),
            InferCategoricalHint(
                target_columns=["status", "category"],
                threshold=10,
                unique_counts={}
            ),
            DropMissingRowsHint(
                target_columns=["value"],
                how="any"
            )
        ]
        
        validation_result = DataValidationResult(
            check_name="test",
            passed=True,
            cleaning_hints=hints
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        description = plan.describe()
        assert "Cleaning Plan:" in description
        assert "convert_to_categorical" in description
        assert "status, category" in description
        assert "drop_missing_rows" in description
        assert "value" in description
        assert "threshold=10" in description
        assert "how=any" in description
    
    def test_custom_operations_list(self):
        """Test providing custom operations to planner."""
        custom_ops = [
            MockOperation("custom_op", 50, "custom_type")
        ]
        
        hint = MockHint(hint_type="custom_type", test_priority=50)
        
        validation_result = DataValidationResult(
            check_name="test",
            passed=True,
            cleaning_hints=[hint]
        )
        
        validation_results = AggregatedValidationResult(
            passed=True,
            individual_results=[validation_result]
        )
        
        planner = DataCleaningPlanner(validation_results, operations=custom_ops)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 1
        assert plan.operations[0][0].name == "custom_op"
    
    def test_validation_with_errors_note(self):
        """Note: In practice, planner should never receive validation results with errors.
        
        The graph should enforce that planning/cleaning nodes are not executed
        when validation has errors. This test documents what would happen if
        the planner did receive such results (it would still create a plan from
        available hints).
        """
        validation_result = DataValidationResult(
            check_name="test",
            passed=False,
            issues=[ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Critical error"
            )],
            cleaning_hints=[
                InferCategoricalHint(target_columns=["col1"], threshold=10, unique_counts={})
            ]
        )
        
        validation_results = AggregatedValidationResult(
            passed=False,
            individual_results=[validation_result]
        )
        
        # Planner would still work, but this scenario should be prevented by the graph
        planner = DataCleaningPlanner(validation_results)
        plan = planner.create_plan()
        
        assert len(plan.operations) == 1  # Would still create plan
        assert plan.validation_results.passed is False  # Preserves failed status 