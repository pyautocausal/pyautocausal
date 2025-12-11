"""Tests for causal-specific validation checks."""

import pytest
import pandas as pd
import numpy as np

from pyautocausal.data_validation.base import ValidationSeverity
from pyautocausal.data_validation.checks.causal_checks import (
    TreatmentPersistenceCheck,
    TreatmentPersistenceConfig,
    OutcomeVariableCheck,
    OutcomeVariableConfig,
    CausalMethodRequirementsCheck,
    CausalMethodRequirementsConfig,
    TreatmentTimingPatternsCheck,
    TreatmentTimingPatternsConfig,
)


class TestTreatmentPersistenceCheck:
    """Test TreatmentPersistenceCheck with various scenarios."""
    
    def test_valid_persistent_treatment(self):
        """Test data with valid persistent treatment (no reversals)."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "treatment": [0, 0, 1, 1, 0, 1, 1, 1],  # Valid: once treated, stays treated
            "outcome": [10, 12, 15, 18, 8, 14, 16, 20]
        })
        
        check = TreatmentPersistenceCheck()
        result = check.validate(df)
        
        assert result.passed
        assert len(result.issues) == 0
        assert result.metadata["violations_found"] == 0
    
    def test_treatment_reversals_detected(self):
        """Test detection of treatment reversals."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "treatment": [0, 1, 0, 1, 1, 0, 1, 1],  # Invalid: treatment reversals
            "outcome": [10, 12, 15, 18, 8, 14, 16, 20]
        })
        
        check = TreatmentPersistenceCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "Treatment reversals detected" in result.issues[0].message
        assert result.metadata["violations_found"] == 2  # Both units have reversals
    
    def test_treatment_reversals_allowed(self):
        """Test allowing treatment reversals with configuration."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1],
            "time": [1, 2, 3, 4],
            "treatment": [0, 1, 0, 1],  # Reversals allowed
            "outcome": [10, 12, 15, 18]
        })
        
        config = TreatmentPersistenceConfig(allow_treatment_reversals=True)
        check = TreatmentPersistenceCheck(config)
        result = check.validate(df)
        
        assert result.passed
        assert len(result.issues) == 0
    
    def test_missing_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            "unit_id": [1, 2, 3],
            "outcome": [10, 20, 30]
            # Missing treatment and time columns
        })
        
        check = TreatmentPersistenceCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "Missing required columns" in result.issues[0].message


class TestOutcomeVariableCheck:
    """Test OutcomeVariableCheck with various scenarios."""
    
    def test_valid_numeric_outcome(self):
        """Test valid numeric outcome variable."""
        df = pd.DataFrame({
            "outcome": [10.5, 12.3, 15.7, 18.2, 8.9, 14.1, 16.8, 20.4],
            "treatment": [0, 0, 1, 1, 0, 1, 1, 1]
        })
        
        check = OutcomeVariableCheck()
        result = check.validate(df)
        
        assert result.passed
        assert result.metadata["is_numeric"]
        assert result.metadata["variance"] > 0
    
    def test_non_numeric_outcome_error(self):
        """Test error when outcome is not numeric."""
        df = pd.DataFrame({
            "outcome": ["low", "medium", "high", "medium", "low", "high", "high", "medium"],
            "treatment": [0, 0, 1, 1, 0, 1, 1, 1]
        })
        
        check = OutcomeVariableCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "not numeric" in result.issues[0].message
    
    def test_insufficient_variation(self):
        """Test detection of insufficient outcome variation."""
        df = pd.DataFrame({
            "outcome": [10.0] * 8,  # No variation
            "treatment": [0, 0, 1, 1, 0, 1, 1, 1]
        })
        
        check = OutcomeVariableCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "insufficient variation" in result.issues[0].message
    
    def test_extreme_outliers_detection(self):
        """Test detection of extreme outliers."""
        df = pd.DataFrame({
            "outcome": [1, 1, 1, 1, 1, 100000, 1, 1],  # 100000 is an extreme outlier
            "treatment": [0, 0, 1, 1, 0, 1, 1, 1]
        })
        
        # Use lower outlier threshold to detect the outlier
        config = OutcomeVariableConfig(outlier_threshold=2.0)  # 2 standard deviations
        check = OutcomeVariableCheck(config)
        result = check.validate(df)
        
        # Should pass but generate warning about outliers
        outlier_issues = [i for i in result.issues if "outliers" in i.message]
        assert len(outlier_issues) > 0
    
    def test_missing_outcome_column(self):
        """Test error when outcome column doesn't exist."""
        df = pd.DataFrame({
            "treatment": [0, 1, 0, 1],
            "other_col": [1, 2, 3, 4]
        })
        
        check = OutcomeVariableCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert "not found" in result.issues[0].message


class TestCausalMethodRequirementsCheck:
    """Test CausalMethodRequirementsCheck with various scenarios."""
    
    def test_sufficient_pre_post_periods(self):
        """Test data with sufficient pre/post treatment periods."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "time": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "treatment": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Treatment starts at t=4
            "outcome": range(12)
        })
        
        check = CausalMethodRequirementsCheck()
        result = check.validate(df)
        
        # Should pass without warnings about insufficient periods
        period_issues = [i for i in result.issues if "periods" in i.message]
        assert len([i for i in period_issues if i.severity == ValidationSeverity.ERROR]) == 0
    
    def test_insufficient_pre_periods(self):
        """Test warning for insufficient pre-treatment periods."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 0, 0],  # Only 1 pre-period (t=1)
            "outcome": range(6)
        })
        
        check = CausalMethodRequirementsCheck()
        result = check.validate(df)
        
        pre_period_issues = [i for i in result.issues if "pre-treatment periods" in i.message]
        assert len(pre_period_issues) > 0
        assert pre_period_issues[0].severity == ValidationSeverity.WARNING
    
    def test_synthetic_control_requirements(self):
        """Test requirements check for synthetic control (one treated unit)."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Only unit 1 treated, 2 control units
            "outcome": range(9)
        })
        
        config = CausalMethodRequirementsConfig(min_control_units=5)
        check = CausalMethodRequirementsCheck(config)
        result = check.validate(df)
        
        # Should warn about insufficient control units for synthetic control
        synth_issues = [i for i in result.issues if "Synthetic control method" in i.message]
        assert len(synth_issues) > 0
    
    def test_staggered_did_requirements(self):
        """Test requirements check for staggered DiD."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2 treated, 2 never-treated
            "outcome": range(12)
        })
        
        config = CausalMethodRequirementsConfig(min_never_treated_fraction=0.8)  # Require 80% never-treated
        check = CausalMethodRequirementsCheck(config)
        result = check.validate(df)
        
        # Should warn about insufficient never-treated units
        stag_issues = [i for i in result.issues if "Staggered DiD" in i.message]
        assert len(stag_issues) > 0
    
    def test_simultaneous_vs_staggered_detection(self):
        """Test detection of simultaneous vs staggered treatment adoption."""
        # Simultaneous adoption
        df_simul = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 1, 1, 0, 0, 0],  # Units 1&2 both treated at t=2
            "outcome": range(9)
        })
        
        check = CausalMethodRequirementsCheck()
        result = check.validate(df_simul)
        
        simul_issues = [i for i in result.issues if "simultaneously" in i.message]
        assert len(simul_issues) > 0
        assert result.metadata["design_type"] == "simultaneous"
        
        # Staggered adoption
        df_stag = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],  # Unit 1 treated at t=2, unit 2 at t=3
            "outcome": range(9)
        })
        
        result_stag = check.validate(df_stag)
        stag_issues = [i for i in result_stag.issues if "staggered" in i.message]
        assert len(stag_issues) > 0
        assert result_stag.metadata["design_type"] == "staggered"


class TestTreatmentTimingPatternsCheck:
    """Test TreatmentTimingPatternsCheck with various scenarios."""
    
    def test_valid_treatment_timing(self):
        """Test data with valid treatment timing patterns."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "time": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "treatment": [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],  # Treatment at t=3 and t=4
            "outcome": range(10)
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        # Should pass without major issues
        error_issues = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(error_issues) == 0
    
    def test_treatment_starts_first_period(self):
        """Test warning when treatment starts in first period."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "treatment": [1, 1, 1, 0, 0, 0],  # Treatment starts at t=1 (first period)
            "outcome": range(6)
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        first_period_issues = [i for i in result.issues if "first time period" in i.message]
        assert len(first_period_issues) > 0
        assert first_period_issues[0].severity == ValidationSeverity.WARNING
    
    def test_treatment_continues_last_period(self):
        """Test warning when treatment continues through last period."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 0, 0],  # Treatment continues through t=3 (last period)
            "outcome": range(6)
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        last_period_issues = [i for i in result.issues if "last time period" in i.message]
        assert len(last_period_issues) > 0
        assert last_period_issues[0].severity == ValidationSeverity.WARNING
    
    def test_no_treatment_found(self):
        """Test handling when no treatment is found."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "treatment": [0, 0, 0, 0, 0, 0],  # No treatment
            "outcome": range(6)
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        assert result.passed  # Not an error, just warning
        no_treatment_issues = [i for i in result.issues if "No treated observations" in i.message]
        assert len(no_treatment_issues) > 0
    
    def test_simultaneous_adoption_detection(self):
        """Test detection of high simultaneous adoption rates."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],  # 3 out of 3 treated units adopt at t=2
            "outcome": range(12)
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        # Should detect high simultaneous adoption
        adoption_issues = [i for i in result.issues if "adopt treatment at time" in i.message]
        assert len(adoption_issues) > 0
        assert result.metadata["adoption_pattern"] == "simultaneous"
    
    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            "unit_id": [1, 2, 3],
            "outcome": [10, 20, 30]
            # Missing treatment and time columns
        })
        
        check = TreatmentTimingPatternsCheck()
        result = check.validate(df)
        
        assert not result.passed
        assert len(result.issues) == 1
        assert "Missing required columns" in result.issues[0].message


class TestIntegrationScenarios:
    """Test integration scenarios with multiple checks."""
    
    def test_comprehensive_causal_validation(self):
        """Test running multiple causal checks on the same dataset."""
        # Create comprehensive test dataset
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            "time": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "treatment": [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "outcome": [10, 12, 15, 18, 20, 8, 10, 12, 16, 18, 9, 11, 13, 15, 17]
        })
        
        # Run all checks
        persistence_check = TreatmentPersistenceCheck()
        outcome_check = OutcomeVariableCheck()
        requirements_check = CausalMethodRequirementsCheck()
        timing_check = TreatmentTimingPatternsCheck()
        
        persistence_result = persistence_check.validate(df)
        outcome_result = outcome_check.validate(df)
        requirements_result = requirements_check.validate(df)
        timing_result = timing_check.validate(df)
        
        # All should pass basic validation
        assert persistence_result.passed
        assert outcome_result.passed
        # requirements and timing may have warnings but shouldn't error
        assert not any(i.severity == ValidationSeverity.ERROR for i in requirements_result.issues)
        assert not any(i.severity == ValidationSeverity.ERROR for i in timing_result.issues)
    
    def test_problematic_dataset(self):
        """Test dataset with multiple causal inference problems."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "treatment": [1, 0, 1, 0, 0, 0],  # Treatment reversal + starts in first period
            "outcome": ["low", "high", "medium", "low", "medium", "high"]  # Non-numeric outcome
        })
        
        # Run all checks - should detect multiple problems
        persistence_check = TreatmentPersistenceCheck()
        outcome_check = OutcomeVariableCheck()
        timing_check = TreatmentTimingPatternsCheck()
        
        persistence_result = persistence_check.validate(df)
        outcome_result = outcome_check.validate(df)
        timing_result = timing_check.validate(df)
        
        # Should detect treatment reversal
        assert not persistence_result.passed
        
        # Should detect non-numeric outcome
        assert not outcome_result.passed
        
        # Should detect problematic timing patterns
        timing_warnings = [i for i in timing_result.issues if i.severity == ValidationSeverity.WARNING]
        assert len(timing_warnings) > 0 