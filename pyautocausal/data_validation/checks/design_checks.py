"""Validation of the analysis sample after cleaning."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..base import DataValidationCheck, DataValidationConfig, ValidationIssue, ValidationSeverity


@dataclass
class CausalDesignConfig(DataValidationConfig):
    treatment_column: str = "treat"
    outcome_column: str = "y"
    unit_column: Optional[str] = None
    time_column: Optional[str] = None

    def __post_init__(self):
        if (self.unit_column is None) != (self.time_column is None):
            raise ValueError("unit_column and time_column must be supplied together")


class CausalDesignCheck(DataValidationCheck[CausalDesignConfig]):
    """Reject samples that cannot identify the pipeline's advertised contrasts.

    Unit identifiers remain labels, including strings and pandas string dtypes.
    Unbalanced panels are allowed; individual estimators may impose stricter
    requirements. Panel treatment must be an absorbing exposure indicator.
    """

    @property
    def name(self):
        return "causal_design"

    @classmethod
    def get_default_config(cls):
        return CausalDesignConfig()

    def validate(self, df):
        config = self.config
        issues = []
        metadata = {"rows": len(df), "panel": config.unit_column is not None}

        def error(message, **details):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR, message=message, details=details
            ))

        columns = [config.treatment_column, config.outcome_column]
        if config.unit_column is not None:
            columns += [config.unit_column, config.time_column]
        missing = [column for column in columns if column not in df.columns]
        if missing:
            error(f"Missing required analysis columns: {missing}")
            return self._create_result(False, issues, metadata)
        if df.empty:
            error("Cleaning left no observations for causal estimation")
            return self._create_result(False, issues, metadata)
        incomplete = [column for column in columns if df[column].isna().any()]
        if incomplete:
            error(f"Analysis columns contain missing values after cleaning: {incomplete}")
            return self._create_result(False, issues, metadata)

        treatment = df[config.treatment_column]
        if set(treatment.unique()) != {0, 1}:
            error("Causal estimation requires both treatment groups coded 0 and 1 after cleaning",
                  observed_values=treatment.unique().tolist())
        outcome = pd.to_numeric(df[config.outcome_column], errors="coerce")
        if not np.isfinite(outcome.to_numpy(dtype=float)).all():
            error("Outcome must contain finite numeric values after cleaning")
        metadata["treatment_counts"] = {str(value): int(count) for value, count in treatment.value_counts().items()}
        if config.unit_column is None or issues:
            return self._create_result(not issues, issues, metadata)

        unit, time = config.unit_column, config.time_column
        duplicates = df.duplicated([unit, time], keep=False)
        if duplicates.any():
            error("Duplicate unit-time observations are ambiguous; resolve them explicitly before estimation",
                  rows=int(duplicates.sum()), sample=df.loc[duplicates, [unit, time]].head().to_dict("records"))
        if df[unit].nunique() < 2 or df[time].nunique() < 2:
            error("Panel causal estimation requires at least two units and two periods")
        metadata.update(units=int(df[unit].nunique()), periods=int(df[time].nunique()))
        if issues:
            return self._create_result(False, issues, metadata)

        # Sort within each unit by time only so string labels are never coerced.
        ordered = df.sort_values(time)
        adoption_times = []
        no_pre_period = []
        reversals = []
        treated_units = []
        for label, observations in ordered.groupby(unit, sort=False, observed=True):
            sequence = observations[config.treatment_column]
            # Boolean diff is XOR, and unsigned diff can wrap 0 - 1 to 255.
            # Coding has already been validated, so signed differences are safe.
            if (sequence.astype('int64').diff() < 0).any():
                reversals.append(label)
            treated = observations.loc[sequence == 1, time]
            if len(treated):
                treated_units.append(label)
                adoption = treated.min()
                adoption_times.append(adoption)
                if not ((sequence == 0) & (observations[time] < adoption)).any():
                    no_pre_period.append(label)
        if reversals:
            error("Panel treatment reversals are unsupported by the automatic pipeline", units=reversals)
        if no_pre_period:
            error("Treated units require observed pre-treatment periods after cleaning", units=no_pre_period)
        # No never-treated requirement: staggered designs may use not-yet-treated units.
        comparison_times = df.loc[treatment == 0, time]
        if not any((comparison_times == adoption).any() for adoption in adoption_times):
            error("No untreated comparison observations remain at any treatment adoption period")
        metadata.update(
            treated_units=len(treated_units),
            never_treated_units=int(df[unit].nunique()) - len(treated_units),
            balanced=len(df) == df[unit].nunique() * df[time].nunique(),
        )
        return self._create_result(not issues, issues, metadata)
