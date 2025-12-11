"""Causal inference specific validation checks for pandas DataFrames."""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union, Dict, Any
import pandas as pd
import numpy as np

from ..base import (
    DataValidationCheck,
    DataValidationConfig,
    DataValidationResult,
    ValidationIssue,
    ValidationSeverity
)


@dataclass
class BinaryTreatmentConfig(DataValidationConfig):
    """Configuration for BinaryTreatmentCheck."""
    treatment_column: str = "treatment"
    allow_missing: bool = False
    valid_values: Set[Union[int, float]] = field(default_factory=lambda: {0, 1})


class BinaryTreatmentCheck(DataValidationCheck[BinaryTreatmentConfig]):
    """Check that treatment variable is binary (0/1)."""
    
    @property
    def name(self) -> str:
        return "binary_treatment"
    
    @classmethod
    def get_default_config(cls) -> BinaryTreatmentConfig:
        return BinaryTreatmentConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if treatment column exists
        if self.config.treatment_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Treatment column '{self.config.treatment_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        treatment = df[self.config.treatment_column]
        
        # Check for missing values
        missing_count = treatment.isna().sum()
        if missing_count > 0 and not self.config.allow_missing:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment column has {missing_count} missing values",
                affected_columns=[self.config.treatment_column],
                details={"missing_count": int(missing_count)}
            ))
        
        # Check values are in valid set
        unique_values = set(treatment.dropna().unique())
        invalid_values = unique_values - self.config.valid_values
        
        if invalid_values:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment column contains invalid values: {invalid_values}. Expected: {self.config.valid_values}",
                affected_columns=[self.config.treatment_column],
                details={
                    "invalid_values": list(invalid_values),
                    "valid_values": list(self.config.valid_values),
                    "unique_values": list(unique_values)
                }
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        # Calculate treatment statistics
        value_counts = treatment.value_counts().to_dict()
        
        # Calculate treatment fraction only if treatment column is numeric and has valid values
        treatment_fraction = None
        if len(treatment.dropna()) > 0:
            try:
                # Only calculate fraction if treatment contains numeric values
                numeric_treatment = pd.to_numeric(treatment.dropna(), errors='coerce')
                if not numeric_treatment.isna().all():
                    treatment_fraction = float(numeric_treatment.sum() / len(numeric_treatment.dropna()))
            except (TypeError, ValueError):
                # If conversion fails, leave treatment_fraction as None
                pass
        
        metadata = {
            "treatment_column": self.config.treatment_column,
            "unique_values": list(unique_values),
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
            "missing_count": int(missing_count),
            "treatment_fraction": treatment_fraction
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TreatmentVariationConfig(DataValidationConfig):
    """Configuration for TreatmentVariationCheck."""
    treatment_column: str = "treatment"
    min_treated_fraction: float = 0.05  # Minimum fraction of treated units
    max_treated_fraction: float = 0.95  # Maximum fraction of treated units
    min_treated_count: int = 10  # Minimum number of treated units
    min_control_count: int = 10  # Minimum number of control units


class TreatmentVariationCheck(DataValidationCheck[TreatmentVariationConfig]):
    """Check that there is sufficient variation in treatment assignment."""
    
    @property
    def name(self) -> str:
        return "treatment_variation"
    
    @classmethod
    def get_default_config(cls) -> TreatmentVariationConfig:
        return TreatmentVariationConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if treatment column exists
        if self.config.treatment_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Treatment column '{self.config.treatment_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        treatment = df[self.config.treatment_column].dropna()
        
        # Count treated and control units
        treated_count = (treatment == 1).sum()
        control_count = (treatment == 0).sum()
        total_count = len(treatment)
        
        if total_count == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No non-missing treatment values found",
                affected_columns=[self.config.treatment_column]
            ))
            return self._create_result(False, issues)
        
        treated_fraction = treated_count / total_count
        
        # Check minimum counts
        if treated_count < self.config.min_treated_count:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Only {treated_count} treated units, minimum required is {self.config.min_treated_count}",
                affected_columns=[self.config.treatment_column],
                details={"treated_count": int(treated_count), "required": self.config.min_treated_count}
            ))
        
        if control_count < self.config.min_control_count:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Only {control_count} control units, minimum required is {self.config.min_control_count}",
                affected_columns=[self.config.treatment_column],
                details={"control_count": int(control_count), "required": self.config.min_control_count}
            ))
        
        # Check treatment fraction
        if treated_fraction < self.config.min_treated_fraction:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment fraction {treated_fraction:.1%} is below minimum {self.config.min_treated_fraction:.1%}",
                affected_columns=[self.config.treatment_column],
                details={
                    "treated_fraction": float(treated_fraction),
                    "min_fraction": self.config.min_treated_fraction
                }
            ))
        
        if treated_fraction > self.config.max_treated_fraction:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment fraction {treated_fraction:.1%} is above maximum {self.config.max_treated_fraction:.1%}",
                affected_columns=[self.config.treatment_column],
                details={
                    "treated_fraction": float(treated_fraction),
                    "max_fraction": self.config.max_treated_fraction
                }
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "treated_count": int(treated_count),
            "control_count": int(control_count),
            "total_count": int(total_count),
            "treated_fraction": float(treated_fraction)
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class PanelStructureConfig(DataValidationConfig):
    """Configuration for PanelStructureCheck."""
    unit_column: str = "unit_id"
    time_column: str = "time"
    treatment_column: Optional[str] = "treatment"  # Treatment column for monotonicity check
    require_balanced: bool = True  # Whether to require balanced panel


class PanelStructureCheck(DataValidationCheck[PanelStructureConfig]):
    """Check panel data structure for causal analysis."""
    
    @property
    def name(self) -> str:
        return "panel_structure"
    
    @classmethod
    def get_default_config(cls) -> PanelStructureConfig:
        return PanelStructureConfig()
    
    def _check_required_columns(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], bool]:
        """Check that required columns exist in the dataframe."""
        issues = []
        
        # Check required columns exist
        for col, col_name in [(self.config.unit_column, "unit"), (self.config.time_column, "time")]:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"{col_name.capitalize()} column '{col}' not found",
                    details={"available_columns": list(df.columns)}
                ))
        
        # Check treatment column exists if specified
        treatment_col_exists = True
        if self.config.treatment_column and self.config.treatment_column not in df.columns:
            treatment_col_exists = False
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Treatment column '{self.config.treatment_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
        
        return issues, treatment_col_exists
    
    def _check_panel_structure(self, df: pd.DataFrame, units: np.ndarray, times: np.ndarray) -> list[ValidationIssue]:
        """Check panel structure for duplicates and balance."""
        issues = []
        n_units = len(units)
        n_times = len(times)
        
        # Check for duplicate unit-time observations
        duplicates = df.duplicated(subset=[self.config.unit_column, self.config.time_column])
        if duplicates.any():
            dup_count = duplicates.sum()
            sample_dups = df[duplicates].head(5)
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Found {dup_count} duplicate unit-time observations",
                affected_rows=sample_dups.index.tolist(),
                details={
                    "duplicate_count": int(dup_count),
                    "sample_duplicates": sample_dups[[self.config.unit_column, self.config.time_column]].to_dict('records')
                }
            ))
        
        # Check if panel is balanced
        expected_obs = n_units * n_times
        actual_obs = len(df.drop_duplicates(subset=[self.config.unit_column, self.config.time_column]))
        is_balanced = expected_obs == actual_obs
        
        if self.config.require_balanced and not is_balanced:
            # Find missing observations
            all_combinations = pd.MultiIndex.from_product([units, times], 
                                                        names=[self.config.unit_column, self.config.time_column])
            existing_combinations = pd.MultiIndex.from_frame(
                df[[self.config.unit_column, self.config.time_column]].drop_duplicates()
            )
            missing_combinations = all_combinations.difference(existing_combinations)
            
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Panel is unbalanced: {len(missing_combinations)} unit-time combinations missing",
                details={
                    "expected_observations": expected_obs,
                    "actual_observations": actual_obs,
                    "missing_count": len(missing_combinations),
                    "sample_missing": [f"{u}-{t}" for u, t in missing_combinations[:5]]
                }
            ))
        
        return issues
    
    def _check_treatment_monotonicity(self, df: pd.DataFrame, units: np.ndarray, treatment_col_exists: bool) -> list[ValidationIssue]:
        """Check treatment monotonicity for each unit."""
        issues = []
        
        if not (self.config.treatment_column and treatment_col_exists):
            return issues
        
        df_sorted = df.sort_values([self.config.unit_column, self.config.time_column])
        violations = []
        
        for unit in units:
            unit_data = df_sorted[df_sorted[self.config.unit_column] == unit]
            treatment_sequence = unit_data[self.config.treatment_column].values
            
            # Check monotonicity: once treated (1), should remain treated
            treated_once = False
            for i, treatment in enumerate(treatment_sequence):
                if treatment == 1:
                    treated_once = True
                elif treated_once and treatment == 0:
                    # Found violation: unit was treated but then reverted to control
                    violations.append({
                        "unit": unit,
                        "violation_at_time": unit_data.iloc[i][self.config.time_column],
                        "treatment_sequence": treatment_sequence.tolist()
                    })
                    break  # Only report first violation per unit
        
        if violations:
            sample_violations = violations[:5]  # Show up to 5 examples
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment monotonicity violated for {len(violations)} units",
                details={
                    "violation_count": len(violations),
                    "sample_violations": sample_violations,
                    "treatment_column": self.config.treatment_column
                }
            ))
        
        return issues
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check required columns exist
        missing_cols_issues, treatment_col_exists = self._check_required_columns(df)
        issues.extend(missing_cols_issues)
        
        if issues:
            return self._create_result(False, issues)
        
        # Get unique units and time periods
        units = df[self.config.unit_column].unique()
        times = df[self.config.time_column].unique()
        n_units = len(units)
        n_times = len(times)
        
        # Check panel structure (duplicates and balance)
        panel_issues = self._check_panel_structure(df, units, times)
        issues.extend(panel_issues)
        
        # Check treatment monotonicity if treatment column is available
        monotonicity_issues = self._check_treatment_monotonicity(df, units, treatment_col_exists)
        issues.extend(monotonicity_issues)
        
        # Calculate balance for metadata
        expected_obs = n_units * n_times
        actual_obs = len(df.drop_duplicates(subset=[self.config.unit_column, self.config.time_column]))
        is_balanced = expected_obs == actual_obs
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "n_units": n_units,
            "n_times": n_times,
            "n_observations": len(df),
            "is_balanced": is_balanced,
            "balance_ratio": actual_obs / expected_obs if expected_obs > 0 else 0
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TimeColumnConfig(DataValidationConfig):
    """Configuration for TimeColumnCheck."""
    time_column: str = "time"
    require_sequential: bool = True  # Whether time periods should be sequential
    require_numeric: bool = False  # Whether time should be numeric
    date_format: Optional[str] = None  # Expected date format if time is string


class TimeColumnCheck(DataValidationCheck[TimeColumnConfig]):
    """Check time column properties for causal analysis."""
    
    @property
    def name(self) -> str:
        return "time_column"
    
    @classmethod
    def get_default_config(cls) -> TimeColumnConfig:
        return TimeColumnConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if time column exists
        if self.config.time_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Time column '{self.config.time_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        time_col = df[self.config.time_column]
        
        # Check for missing values
        if time_col.isna().any():
            missing_count = time_col.isna().sum()
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Time column has {missing_count} missing values",
                affected_columns=[self.config.time_column],
                details={"missing_count": int(missing_count)}
            ))
        
        # Get unique time periods
        unique_times = sorted(time_col.dropna().unique())
        n_periods = len(unique_times)
        
        # Check data type
        is_numeric = pd.api.types.is_numeric_dtype(time_col)
        is_datetime = pd.api.types.is_datetime64_any_dtype(time_col)
        
        if self.config.require_numeric and not is_numeric:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Time column is not numeric (type: {time_col.dtype})",
                affected_columns=[self.config.time_column],
                details={"actual_type": str(time_col.dtype)}
            ))
        
        # Check sequential periods if required
        if self.config.require_sequential and n_periods > 1:
            if is_numeric:
                # Check for gaps in numeric sequence
                diffs = np.diff(unique_times)
                if not np.all(diffs == diffs[0]):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Time periods are not evenly spaced",
                        affected_columns=[self.config.time_column],
                        details={
                            "period_gaps": [f"{unique_times[i]}-{unique_times[i+1]}: {diffs[i]}" 
                                          for i in range(min(5, len(diffs)))]
                        }
                    ))
            elif is_datetime:
                # Check for regular intervals in datetime
                time_diffs = pd.Series(unique_times).diff()[1:]
                if not time_diffs.nunique() == 1:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Time periods have irregular intervals",
                        affected_columns=[self.config.time_column],
                        details={"unique_intervals": time_diffs.value_counts().to_dict()}
                    ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        metadata = {
            "n_periods": n_periods,
            "time_range": [str(unique_times[0]), str(unique_times[-1])] if n_periods > 0 else None,
            "is_numeric": is_numeric,
            "is_datetime": is_datetime,
            "unique_periods": [str(t) for t in unique_times[:10]]  # First 10 periods
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TreatmentPersistenceConfig(DataValidationConfig):
    """Configuration for TreatmentPersistenceCheck."""
    treatment_column: str = "treatment"
    unit_column: str = "unit_id"
    time_column: str = "time"
    allow_treatment_reversals: bool = False  # Whether to allow treatment reversals


class TreatmentPersistenceCheck(DataValidationCheck[TreatmentPersistenceConfig]):
    """Check that treatment is persistent over time (weakly increasing)."""
    
    @property
    def name(self) -> str:
        return "treatment_persistence"
    
    @classmethod
    def get_default_config(cls) -> TreatmentPersistenceConfig:
        return TreatmentPersistenceConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check required columns exist
        required_cols = [self.config.treatment_column, self.config.unit_column, self.config.time_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required columns for treatment persistence check: {missing_cols}",
                details={"missing_columns": missing_cols, "available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        # Sort by unit and time for checking persistence
        df_sorted = df.sort_values([self.config.unit_column, self.config.time_column])
        
        # Check treatment persistence for each unit
        violation_units = []
        violation_details = []
        
        for unit_id in df_sorted[self.config.unit_column].unique():
            unit_data = df_sorted[df_sorted[self.config.unit_column] == unit_id]
            treatment_values = unit_data[self.config.treatment_column].dropna()
            
            if len(treatment_values) <= 1:
                continue  # Skip units with only one observation
            
            # Check for treatment reversals (1 -> 0)
            reversals = []
            prev_treatment = None
            
            for idx, (time, treatment) in enumerate(zip(unit_data[self.config.time_column], treatment_values)):
                if prev_treatment is not None:
                    if prev_treatment == 1 and treatment == 0:
                        reversals.append((time, treatment))
                prev_treatment = treatment
            
            if reversals and not self.config.allow_treatment_reversals:
                violation_units.append(unit_id)
                violation_details.append({
                    "unit": unit_id,
                    "reversals": reversals[:3]  # First 3 reversals
                })
        
        if violation_units:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Treatment reversals detected for {len(violation_units)} units. Treatment should be persistent (weakly increasing).",
                affected_columns=[self.config.treatment_column, self.config.unit_column],
                details={
                    "violating_units": violation_units[:10],  # First 10 units
                    "violation_details": violation_details[:5],  # First 5 details
                    "total_violations": len(violation_units)
                }
            ))
        
        passed = len(issues) == 0
        metadata = {
            "units_checked": df_sorted[self.config.unit_column].nunique(),
            "violations_found": len(violation_units),
            "allow_reversals": self.config.allow_treatment_reversals
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class OutcomeVariableConfig(DataValidationConfig):
    """Configuration for OutcomeVariableCheck."""
    outcome_column: str = "outcome"
    min_variation_threshold: float = 1e-10  # Minimum variance required
    outlier_threshold: float = 5.0  # Standard deviations for outlier detection
    require_numeric: bool = True  # Whether outcome must be numeric


class OutcomeVariableCheck(DataValidationCheck[OutcomeVariableConfig]):
    """Check that outcome variable is suitable for causal analysis."""
    
    @property
    def name(self) -> str:
        return "outcome_variable"
    
    @classmethod
    def get_default_config(cls) -> OutcomeVariableConfig:
        return OutcomeVariableConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if outcome column exists
        if self.config.outcome_column not in df.columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Outcome column '{self.config.outcome_column}' not found",
                details={"available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        outcome = df[self.config.outcome_column].dropna()
        
        if len(outcome) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Outcome column has no non-missing values",
                affected_columns=[self.config.outcome_column]
            ))
            return self._create_result(False, issues)
        
        # Check if outcome is numeric
        is_numeric = pd.api.types.is_numeric_dtype(outcome)
        if self.config.require_numeric and not is_numeric:
            issues.append(ValidationIssue(
                severity=self.config.severity_on_fail,
                message=f"Outcome column is not numeric (type: {outcome.dtype})",
                affected_columns=[self.config.outcome_column],
                details={"actual_type": str(outcome.dtype)}
            ))
        
        if is_numeric:
            # Check for sufficient variation
            outcome_var = outcome.var()
            if outcome_var < self.config.min_variation_threshold:
                issues.append(ValidationIssue(
                    severity=self.config.severity_on_fail,
                    message=f"Outcome has insufficient variation (variance: {outcome_var:.2e})",
                    affected_columns=[self.config.outcome_column],
                    details={"variance": float(outcome_var), "threshold": self.config.min_variation_threshold}
                ))
            
            # Check for extreme outliers
            if len(outcome) > 3:  # Need at least 3 values for outlier detection
                mean_val = outcome.mean()
                std_val = outcome.std()
                
                if std_val > 0:  # Avoid division by zero
                    z_scores = np.abs((outcome - mean_val) / std_val)
                    outliers = z_scores > self.config.outlier_threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        outlier_fraction = outlier_count / len(outcome)
                        severity = ValidationSeverity.WARNING if outlier_fraction < 0.05 else self.config.severity_on_fail
                        
                        issues.append(ValidationIssue(
                            severity=severity,
                            message=f"Outcome has {outlier_count} extreme outliers ({outlier_fraction:.1%} of data)",
                            affected_columns=[self.config.outcome_column],
                            details={
                                "outlier_count": int(outlier_count),
                                "outlier_fraction": float(outlier_fraction),
                                "threshold_std_devs": self.config.outlier_threshold
                            }
                        ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        # Calculate metadata
        metadata = {
            "outcome_column": self.config.outcome_column,
            "is_numeric": is_numeric,
            "non_missing_count": len(outcome),
            "unique_values": outcome.nunique() if len(outcome) > 0 else 0
        }
        
        if is_numeric and len(outcome) > 0:
            metadata.update({
                "mean": float(outcome.mean()),
                "std": float(outcome.std()),
                "variance": float(outcome.var()),
                "min": float(outcome.min()),
                "max": float(outcome.max())
            })
        
        return self._create_result(passed, issues, metadata)


@dataclass
class CausalMethodRequirementsConfig(DataValidationConfig):
    """Configuration for CausalMethodRequirementsCheck."""
    treatment_column: str = "treatment"
    unit_column: str = "unit_id"
    time_column: str = "time"
    min_pre_periods: int = 3  # Minimum pre-treatment periods
    min_post_periods: int = 1  # Minimum post-treatment periods
    min_control_units: int = 5  # Minimum control units for synthetic control
    min_never_treated_fraction: float = 0.2  # Minimum fraction of never-treated units


class CausalMethodRequirementsCheck(DataValidationCheck[CausalMethodRequirementsConfig]):
    """Check data requirements for different causal inference methods."""
    
    @property
    def name(self) -> str:
        return "causal_method_requirements"
    
    @classmethod
    def get_default_config(cls) -> CausalMethodRequirementsConfig:
        return CausalMethodRequirementsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check required columns exist
        required_cols = [self.config.treatment_column, self.config.unit_column, self.config.time_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            ))
            return self._create_result(False, issues)
        
        # Analyze treatment patterns
        units = df[self.config.unit_column].unique()
        times = sorted(df[self.config.time_column].unique())
        
        # Find treatment timing for each unit
        treatment_start_times = {}
        never_treated_units = set()
        
        for unit in units:
            unit_data = df[df[self.config.unit_column] == unit].sort_values(self.config.time_column)
            treatment_values = unit_data[self.config.treatment_column].dropna()
            
            first_treatment = treatment_values[treatment_values == 1]
            if len(first_treatment) > 0:
                treatment_start_times[unit] = unit_data[unit_data[self.config.treatment_column] == 1][self.config.time_column].min()
            else:
                never_treated_units.add(unit)
        
        treated_units = set(treatment_start_times.keys())
        
        # Check for sufficient pre/post periods
        if treatment_start_times:
            min_treatment_time = min(treatment_start_times.values())
            max_treatment_time = max(treatment_start_times.values())
            
            pre_periods = [t for t in times if t < min_treatment_time]
            post_periods = [t for t in times if t > max_treatment_time]
            
            if len(pre_periods) < self.config.min_pre_periods:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Only {len(pre_periods)} pre-treatment periods available, {self.config.min_pre_periods} recommended for robust causal inference",
                    details={
                        "pre_periods_available": len(pre_periods),
                        "pre_periods_recommended": self.config.min_pre_periods
                    }
                ))
            
            if len(post_periods) < self.config.min_post_periods:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Only {len(post_periods)} post-treatment periods available, {self.config.min_post_periods} minimum for causal analysis",
                    details={
                        "post_periods_available": len(post_periods),
                        "post_periods_required": self.config.min_post_periods
                    }
                ))
        
        # Check for synthetic control feasibility (exactly one treated unit)
        if len(treated_units) == 1:
            if len(never_treated_units) < self.config.min_control_units:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Synthetic control method: only {len(never_treated_units)} control units available, {self.config.min_control_units} recommended",
                    details={
                        "control_units_available": len(never_treated_units),
                        "control_units_recommended": self.config.min_control_units,
                        "method": "synthetic_control"
                    }
                ))
        
        # Check for staggered DiD feasibility
        if len(treated_units) > 1:
            never_treated_fraction = len(never_treated_units) / len(units)
            if never_treated_fraction < self.config.min_never_treated_fraction:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Staggered DiD: only {never_treated_fraction:.1%} never-treated units, {self.config.min_never_treated_fraction:.1%} recommended for robust identification",
                    details={
                        "never_treated_fraction": float(never_treated_fraction),
                        "never_treated_recommended": self.config.min_never_treated_fraction,
                        "method": "staggered_did"
                    }
                ))
        
        # Check for treatment timing variation (all units treated at same time)
        if len(set(treatment_start_times.values())) == 1 and len(treated_units) > 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="All treated units adopt treatment simultaneously (standard DiD design)",
                details={"design_type": "simultaneous_adoption"}
            ))
        elif len(set(treatment_start_times.values())) > 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Treatment adoption is staggered across units (staggered DiD design)",
                details={"design_type": "staggered_adoption"}
            ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        metadata = {
            "total_units": len(units),
            "treated_units": len(treated_units),
            "never_treated_units": len(never_treated_units),
            "never_treated_fraction": len(never_treated_units) / len(units) if len(units) > 0 else 0,
            "time_periods": len(times),
            "treatment_times": list(set(treatment_start_times.values())) if treatment_start_times else [],
            "design_type": "staggered" if len(set(treatment_start_times.values())) > 1 else "simultaneous"
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TreatmentTimingPatternsConfig(DataValidationConfig):
    """Configuration for TreatmentTimingPatternsCheck."""
    treatment_column: str = "treatment"
    unit_column: str = "unit_id" 
    time_column: str = "time"
    check_simultaneous_adoption: bool = True  # Check if treatment adoption follows expected patterns


class TreatmentTimingPatternsCheck(DataValidationCheck[TreatmentTimingPatternsConfig]):
    """Check treatment timing patterns for causal inference validity."""
    
    @property
    def name(self) -> str:
        return "treatment_timing_patterns"
    
    @classmethod
    def get_default_config(cls) -> TreatmentTimingPatternsConfig:
        return TreatmentTimingPatternsConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check required columns exist
        required_cols = [self.config.treatment_column, self.config.unit_column, self.config.time_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            ))
            return self._create_result(False, issues)
        
        # Analyze treatment timing patterns
        df_sorted = df.sort_values([self.config.unit_column, self.config.time_column])
        
        # Find first and last treatment periods
        treatment_data = df_sorted[df_sorted[self.config.treatment_column] == 1]
        if len(treatment_data) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No treated observations found in data",
                affected_columns=[self.config.treatment_column]
            ))
            return self._create_result(True, issues)  # Not necessarily an error
        
        first_treatment_period = treatment_data[self.config.time_column].min()
        last_treatment_period = treatment_data[self.config.time_column].max()
        all_periods = sorted(df[self.config.time_column].unique())
        
        # Check if treatment starts/ends in extreme periods
        if first_treatment_period == min(all_periods):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Treatment begins in the first time period, limiting pre-treatment analysis",
                details={"first_treatment_period": first_treatment_period, "first_available_period": min(all_periods)}
            ))
        
        if last_treatment_period == max(all_periods):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Treatment continues through the last time period, limiting post-treatment analysis",
                details={"last_treatment_period": last_treatment_period, "last_available_period": max(all_periods)}
            ))
        
        # Analyze treatment adoption patterns
        treatment_adoptions = {}
        for unit in df[self.config.unit_column].unique():
            unit_data = df_sorted[df_sorted[self.config.unit_column] == unit]
            first_treated = unit_data[unit_data[self.config.treatment_column] == 1]
            
            if len(first_treated) > 0:
                adoption_time = first_treated[self.config.time_column].iloc[0]
                treatment_adoptions[unit] = adoption_time
        
        # Check for problematic adoption patterns
        if len(treatment_adoptions) > 0:
            adoption_times = list(treatment_adoptions.values())
            unique_adoption_times = list(set(adoption_times))
            
            # Check if too many units adopt simultaneously
            for adoption_time in unique_adoption_times:
                units_adopting = [unit for unit, time in treatment_adoptions.items() if time == adoption_time]
                adoption_fraction = len(units_adopting) / len(treatment_adoptions)
                
                if adoption_fraction > 0.5:  # More than 50% adopt at same time
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"{len(units_adopting)} units ({adoption_fraction:.1%}) adopt treatment at time {adoption_time}",
                        details={
                            "adoption_time": adoption_time,
                            "units_adopting": len(units_adopting),
                            "adoption_fraction": float(adoption_fraction)
                        }
                    ))
        
        passed = not any(issue.severity == self.config.severity_on_fail for issue in issues)
        
        metadata = {
            "treatment_periods": len(treatment_data[self.config.time_column].unique()),
            "first_treatment_period": first_treatment_period,
            "last_treatment_period": last_treatment_period,
            "units_ever_treated": len(treatment_adoptions),
            "unique_adoption_times": len(set(treatment_adoptions.values())) if treatment_adoptions else 0,
            "adoption_pattern": "staggered" if len(set(treatment_adoptions.values())) > 1 else "simultaneous"
        }
        
        return self._create_result(passed, issues, metadata)


@dataclass
class TimePeriodStandardizationConfig(DataValidationConfig):
    """Configuration for TimePeriodStandardizationCheck."""
    treatment_column: str = "treatment"
    time_column: str = "time"


class TimePeriodStandardizationCheck(DataValidationCheck[TimePeriodStandardizationConfig]):
    """Check and prepare standardization of time periods relative to treatment start.
    
    This check:
    1. Finds the minimum time period where treatment==1 occurs
    2. Creates a standardized mapping where that period becomes index 0
    3. Earlier periods get negative indices (-1, -2, etc.)
    4. Later periods get positive indices (1, 2, etc.)
    5. Returns a cleaning hint with the value mapping
    
    Always requires treatment data to exist - fails with ERROR if no treatment==1 found.
    """
    
    @property
    def name(self) -> str:
        return "time_period_standardization"
    
    @classmethod
    def get_default_config(cls) -> TimePeriodStandardizationConfig:
        return TimePeriodStandardizationConfig()
    
    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        issues = []
        
        # Check if required columns exist
        missing_cols = []
        for col in [self.config.treatment_column, self.config.time_column]:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Required columns not found: {missing_cols}",
                details={"missing_columns": missing_cols, "available_columns": list(df.columns)}
            ))
            return self._create_result(False, issues)
        
        # Check if treatment data exists (treatment==1)
        treatment_data = df[df[self.config.treatment_column] == 1]
        if len(treatment_data) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"No treatment data found (no rows where {self.config.treatment_column}==1). "
                       "Time period standardization requires treatment data to define the reference point.",
                affected_columns=[self.config.treatment_column],
                details={"unique_treatment_values": df[self.config.treatment_column].unique().tolist()}
            ))
            return self._create_result(False, issues)
        
        # Get all unique time periods and find the treatment start period
        unique_times = df[self.config.time_column].dropna().unique()
        treatment_times = treatment_data[self.config.time_column].dropna().unique()
        
        if len(treatment_times) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"No valid time periods found for treatment data (all time values are NaN where {self.config.treatment_column}==1)",
                affected_columns=[self.config.time_column, self.config.treatment_column]
            ))
            return self._create_result(False, issues)
        
        # Check for mixed data types in time column
        mixed_types_issue = self._check_for_mixed_types(unique_times)
        if mixed_types_issue:
            issues.append(mixed_types_issue)
            return self._create_result(False, issues)
        
        try:
            # Try to parse and sort time periods
            parsed_times = self._parse_time_periods(unique_times)
            parsed_treatment_times = self._parse_time_periods(treatment_times)
            
            # Find the minimum treatment time period (this becomes index 0)
            treatment_start_parsed = min(parsed_treatment_times)
            
            # Create mapping from original values to standardized indices
            sorted_times = sorted(parsed_times)
            
            value_mapping = {}
            for original_val, parsed_val in zip(unique_times, parsed_times):
                value_mapping[original_val] = sorted_times.index(parsed_val) + 1
                if parsed_val == treatment_start_parsed:
                    treatment_start_original = original_val
            

            # Create cleaning hint
            from ...data_cleaning.hints import StandardizeTimePeriodHint
            cleaning_hint = StandardizeTimePeriodHint(
                time_column=self.config.time_column,
                value_mapping=value_mapping,
                metadata={
                    "total_periods": len(unique_times),
                    "pre_treatment_periods": sum(1 for v in value_mapping.values() if v < value_mapping[treatment_start_original]),
                    "post_treatment_periods": sum(1 for v in value_mapping.values() if v >= value_mapping[treatment_start_original]),
                    "original_treatment_start": str(treatment_start_original)
                }
            )
            
            metadata = {
                "time_column": self.config.time_column,
                "treatment_column": self.config.treatment_column,
                "total_periods": len(unique_times),
                "standardized_range": [min(value_mapping.values()), max(value_mapping.values())],
                "value_mapping_sample": dict(list(value_mapping.items())[:5])  # First 5 for display
            }
            
            return self._create_result(True, issues, metadata, [cleaning_hint])
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Failed to parse time periods in column '{self.config.time_column}': {str(e)}",
                affected_columns=[self.config.time_column],
                details={"error_type": type(e).__name__, "sample_values": unique_times[:10].tolist()}
            ))
            return self._create_result(False, issues)
    
    def _parse_time_periods(self, time_values):
        """Parse time periods into comparable values.
        
        Handles multiple formats:
        - Datetime strings (parsed with pd.to_datetime)
        - Datetime objects
        - Numeric values (treated as periods)
        """
        parsed = []
        
        for val in time_values:
            if pd.isna(val):
                continue
                
            # Try numeric first (integers, floats)
            try:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    parsed.append(float(val))
                    continue
            except (ValueError, TypeError):
                pass
            
            # Try datetime parsing
            try:
                if isinstance(val, str) or hasattr(val, 'strftime'):
                    dt = pd.to_datetime(val)
                    # Convert to ordinal for comparison (days since year 1)
                    parsed.append(dt.toordinal())
                    continue
            except (ValueError, TypeError):
                pass
            
            # If nothing else works, try to convert to string and then float
            try:
                parsed.append(float(str(val)))
            except (ValueError, TypeError):
                raise ValueError(f"Unable to parse time period value: {val} (type: {type(val)})")
        
        return parsed
    
    def _check_for_mixed_types(self, time_values):
        """Check if time column contains mixed data types.
        
        Returns ValidationIssue if mixed types are detected, None otherwise.
        """
        if len(time_values) == 0:
            return None
        
        # Categorize types
        numeric_types = set()
        string_types = set()
        datetime_types = set()
        other_types = set()
        
        for val in time_values:
            if pd.isna(val):
                continue
                
            val_type = type(val)
            
            # Categorize by type family
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_types.add(val_type.__name__)
            elif isinstance(val, str):
                # Check if string could be a date
                try:
                    pd.to_datetime(val)
                    string_types.add("date_string")
                except (ValueError, TypeError):
                    string_types.add("string")
            elif hasattr(val, 'strftime'):  # datetime-like objects
                datetime_types.add(val_type.__name__)
            else:
                other_types.add(val_type.__name__)
        
        # Count how many different type categories we have
        type_families = []
        if numeric_types:
            type_families.append(f"numeric ({', '.join(sorted(numeric_types))})")
        if string_types:
            type_families.append(f"string ({', '.join(sorted(string_types))})")
        if datetime_types:
            type_families.append(f"datetime ({', '.join(sorted(datetime_types))})")
        if other_types:
            type_families.append(f"other ({', '.join(sorted(other_types))})")
        
        # Error if we have mixed type families
        if len(type_families) > 1:
            sample_values = [f"{val} ({type(val).__name__})" for val in time_values[:5]]
            
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Time column '{self.config.time_column}' contains mixed data types: {', '.join(type_families)}. "
                       "All time values should be of the same type family (all numeric, all date strings, or all datetime objects).",
                affected_columns=[self.config.time_column],
                details={
                    "mixed_type_families": type_families,
                    "sample_values": sample_values,
                    "total_unique_values": len(time_values)
                }
            )
        
        return None 