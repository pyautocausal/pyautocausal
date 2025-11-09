import pandas as pd
import pytest
from unittest.mock import patch
from pyautocausal.data_cleaning.hints import UpdateColumnTypesHint
from pyautocausal.data_validation.base import DataValidationResult, DataValidationCheck, ValidationIssue, ValidationSeverity, DataValidationConfig
from pyautocausal.data_validation.validator_base import DataValidator, AggregatedValidationResult, DataValidationError


class MockCheck(DataValidationCheck):
    def __init__(self, is_valid, name="mock_check", errors=None, hints=None):
        super().__init__(config=DataValidationConfig())
        self._is_valid = is_valid
        self._name = name
        self._errors = errors or []
        self._hints = hints or []

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def get_default_config(cls) -> DataValidationConfig:
        return DataValidationConfig()

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = [ValidationIssue(severity=ValidationSeverity.ERROR, message=e) for e in self._errors]
        return self._create_result(passed=self._is_valid, issues=issues, cleaning_hints=self._hints)


def test_data_validator_node_all_pass():
    """Tests that the validator node passes when all checks pass."""
    df = pd.DataFrame()
    checks = [
        MockCheck(is_valid=True, hints=[UpdateColumnTypesHint(type_mapping={"A": "category"})]),
        MockCheck(is_valid=True, hints=[UpdateColumnTypesHint(type_mapping={"B": "category"})]),
    ]
    node = DataValidator(checks=checks)
    result = node.validate(df)

    assert result.passed
    assert not result.get_all_issues()
    
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert len(all_hints) == 2
    assert isinstance(all_hints[0], UpdateColumnTypesHint)


def test_data_validator_node_one_fails():
    """Tests that the validator node raises an exception if any check fails with errors."""
    df = pd.DataFrame()
    checks = [
        MockCheck(is_valid=True, hints=[UpdateColumnTypesHint(type_mapping={"A": "category"})]),
        MockCheck(is_valid=False, errors=["Column C is bad"]),
    ]
    node = DataValidator(checks=checks)
    
    with pytest.raises(DataValidationError) as exc_info:
        node.validate(df)
    
    # Check the exception details
    exception = exc_info.value
    assert "Data validation failed with 1 error(s)" in str(exception)
    assert "mock_check" in str(exception)
    
    # Check the validation result attached to the exception
    result = exception.validation_result
    assert not result.passed
    errors = result.get_all_issues()
    assert len(errors) == 1
    assert "Column C is bad" in errors[0].message
    
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert len(all_hints) == 1  # Should still collect hints from passing checks


def test_data_validator_node_no_checks():
    """Tests that the validator node passes if no checks are provided."""
    df = pd.DataFrame()
    node = DataValidator(checks=[])
    result = node.validate(df)

    assert result.passed
    assert not result.get_all_issues()
    all_hints = [h for res in result.individual_results for h in res.cleaning_hints]
    assert not all_hints


class TrackingMockCheck(DataValidationCheck):
    """Mock check that tracks when its validate method is called."""
    
    execution_log = []  # Class variable to track execution order
    
    def __init__(self, is_valid, name="tracking_check", errors=None, severity=ValidationSeverity.ERROR):
        super().__init__(config=DataValidationConfig())
        self._is_valid = is_valid
        self._name = name
        self._errors = errors or []
        self._severity = severity

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def get_default_config(cls) -> DataValidationConfig:
        return DataValidationConfig()

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        # Track that this check was executed
        TrackingMockCheck.execution_log.append(self._name)
        
        issues = [ValidationIssue(severity=self._severity, message=e) for e in self._errors]
        return self._create_result(passed=self._is_valid, issues=issues)

    @classmethod
    def reset_log(cls):
        """Reset the execution log for a fresh test."""
        cls.execution_log = []


def test_data_validator_fail_fast_on_error():
    """Tests that the validator stops execution when a check encounters ERROR-level issues and raises an exception."""
    TrackingMockCheck.reset_log()
    
    df = pd.DataFrame()
    checks = [
        TrackingMockCheck(is_valid=False, name="check_1", errors=["Check 1 failed"], severity=ValidationSeverity.ERROR),
        TrackingMockCheck(is_valid=True, name="check_2"),  # This should not be executed
        TrackingMockCheck(is_valid=True, name="check_3"),  # This should not be executed
    ]
    
    with patch('logging.getLogger') as mock_logger:
        mock_log_instance = mock_logger.return_value
        
        node = DataValidator(checks=checks)
        
        with pytest.raises(DataValidationError) as exc_info:
            node.validate(df)

        # Verify that only the first check was executed
        assert TrackingMockCheck.execution_log == ["check_1"]
        
        # Verify the exception and its attached result
        exception = exc_info.value
        result = exception.validation_result
        assert not result.passed
        assert len(result.individual_results) == 1  # Only first check should have results
        assert result.individual_results[0].check_name == "check_1"
        assert result.individual_results[0].has_errors
        
        # Verify the exception message
        assert "Data validation failed with 1 error(s) in 1 check(s): check_1" in str(exception)
        
        # Verify the warning was logged
        mock_log_instance.warning.assert_called_once()
        warning_call = mock_log_instance.warning.call_args[0][0]
        assert "Stopping validation early due to ERROR in check 'check_1'" in warning_call
        assert "Skipping 2 remaining checks" in warning_call


def test_data_validator_continues_on_warning():
    """Tests that the validator continues execution when a check has only WARNING-level issues and doesn't raise exception."""
    TrackingMockCheck.reset_log()
    
    df = pd.DataFrame()
    checks = [
        TrackingMockCheck(is_valid=False, name="check_1", errors=["Check 1 warning"], severity=ValidationSeverity.WARNING),
        TrackingMockCheck(is_valid=True, name="check_2"),  # This should be executed
        TrackingMockCheck(is_valid=True, name="check_3"),  # This should be executed
    ]
    
    node = DataValidator(checks=checks)
    result = node.validate(df)  # Should not raise exception

    # Verify that all checks were executed (no fail-fast for warnings)
    assert TrackingMockCheck.execution_log == ["check_1", "check_2", "check_3"]
    
    # Verify the result (should be returned normally, not as exception)
    assert len(result.individual_results) == 3  # All checks should have results
    assert result.individual_results[0].check_name == "check_1"
    assert result.individual_results[0].has_warnings
    assert not result.individual_results[0].has_errors
    
    # Verify no errors in summary (should not raise exception)
    assert result.summary['total_errors'] == 0
    assert result.summary['total_warnings'] == 1


def test_data_validation_error_message_contains_details():
    """Tests that DataValidationError includes detailed validation results in its string representation."""
    TrackingMockCheck.reset_log()
    
    df = pd.DataFrame()
    checks = [
        TrackingMockCheck(is_valid=False, name="required_columns", errors=["Missing column 'treatment'"], severity=ValidationSeverity.ERROR),
        TrackingMockCheck(is_valid=False, name="column_types", errors=["Invalid type for column 'y'"], severity=ValidationSeverity.ERROR),
    ]
    
    node = DataValidator(checks=checks)
    
    with pytest.raises(DataValidationError) as exc_info:
        node.validate(df)
    
    exception = exc_info.value
    exception_str = str(exception)
    
    # Due to fail-fast behavior, only the first check runs
    # Check that the exception message contains summary information
    assert "Data validation failed with 1 error(s) in 1 check(s)" in exception_str
    assert "required_columns" in exception_str
    
    # Check that the detailed validation results are included
    assert "Data Validation Summary" in exception_str
    assert "FAILED" in exception_str
    assert "required_columns:" in exception_str
    assert "Missing column 'treatment'" in exception_str
    
    # The second check should NOT have run due to fail-fast
    assert "column_types:" not in exception_str
    assert "Invalid type for column 'y'" not in exception_str
    
    # Verify only the first check was executed
    assert TrackingMockCheck.execution_log == ["required_columns"] 