"""Core decision logic for the causal inference pipeline.

This module contains the essential decision structure that determines which 
causal analysis methods are applied based on data characteristics.
"""

from pathlib import Path

import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.pipelines.library.conditions import (
    has_multiple_periods, 
    has_staggered_treatment, 
    has_minimum_post_periods, 
    has_sufficient_never_treated_units, 
    has_single_treated_unit
)
from pyautocausal.data_cleaner_interface.autocleaner import AutoCleaner
from pyautocausal.persistence.parameter_mapper import make_transformable


def create_basic_cleaner(
    df: pd.DataFrame, *, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1
) -> pd.DataFrame:
    """Clean required fields and validate both treatment groups.

    Missingness policy is explicit: ``drop_rows`` drops incomplete observations
    up to the per-column threshold; ``reject`` refuses any missing values.
    Unit identifiers are preserved as supplied, including string labels.
    """
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    unit_column = "id_unit" if "id_unit" in df.columns else None
    # Apply the missingness policy before attempting integer treatment casts.
    complete = (
        AutoCleaner(unit_column=unit_column)
        .check_required_columns(required_columns=["treat", "y"])
        .check_binary_treatment(treatment_column="treat", allow_missing=True)
        .check_for_missing_data(strategy=missing_strategy, check_columns=["treat", "y"],
                                max_missing_fraction=max_missing_fraction)
        .clean(df)
    )
    return (
        AutoCleaner(unit_column=unit_column)
        .check_column_types(expected_types={"treat": int, "y": float})
        .check_binary_treatment(treatment_column="treat")
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
        .check_causal_design()
        .clean(complete)
    )


def _validate_missing_policy(missing_strategy: str, max_missing_fraction: float) -> None:
    if missing_strategy not in {"drop_rows", "reject"}:
        raise ValueError("missing_strategy must be 'drop_rows' or 'reject'")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")


def create_panel_cleaner(
    df: pd.DataFrame, *, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1
) -> pd.DataFrame:
    """Clean panel data, preserving unit labels and recording time mappings.

    Revalidate unique unit/time pairs, absorbing treatment, pre-treatment data,
    and comparison support after dropping incomplete rows. Identical panel
    records are ambiguous and require explicit resolution by the caller.
    """
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    complete = (
        AutoCleaner(unit_column="id_unit")
        .check_required_columns(required_columns=["treat", "y", "t", "id_unit"])
        .check_for_missing_data(strategy=missing_strategy, max_missing_fraction=max_missing_fraction)
        .clean(df)
    )
    return (
        AutoCleaner(unit_column="id_unit")
        .standardize_time_periods(treatment_column="treat", time_column="t")
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
        .check_causal_design(unit_column="id_unit", time_column="t")
        .clean(complete)
    )


def create_cross_sectional_cleaner(
    df: pd.DataFrame, *, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1
) -> pd.DataFrame:
    """Clean cross-sectional model columns and validate the remaining groups."""
    _validate_missing_policy(missing_strategy, max_missing_fraction)
    return (
        AutoCleaner(unit_column="id_unit" if "id_unit" in df.columns else None)
        .check_required_columns(required_columns=["treat", "y"])
        .check_for_missing_data(strategy=missing_strategy, max_missing_fraction=max_missing_fraction)
        .infer_and_convert_categoricals(ignore_columns=["treat", "y", "t", "id_unit"])
        .check_causal_design()
        .clean(df)
    )


def _create_shared_head(graph: ExecutableGraph):
    """Creates the shared input and basic cleaning nodes for any graph."""
    # Input node
    graph.create_input_node("df", input_dtype=pd.DataFrame)
    
    # Basic cleaning node (this is a simple wrapper around the basic cleaner)
    graph.create_node('basic_cleaning', action_function=basic_clean_node.get_function(), predecessors=["df"])


def create_panel_decision_structure(graph: ExecutableGraph, predecessor: str = "basic_cleaning") -> None:
    """Create the decision structure for the panel data path."""
    
    # === PANEL DATA PATH (assumes multi-period data) ===
    graph.create_node('panel_cleaned_data', action_function=panel_clean_node.transform({predecessor: 'data_input'}), predecessors=[predecessor])
    
    # === PANEL DATA COMPLEXITY DECISIONS ===
    graph.create_decision_node(
        'single_treated_unit', 
        condition=has_single_treated_unit.transform({'panel_cleaned_data': 'df'}), 
        predecessors=["panel_cleaned_data"]
    )
    
    graph.create_decision_node(
        'multi_post_periods', 
        condition=has_minimum_post_periods.transform({'panel_cleaned_data': 'df'}), 
        predecessors=["single_treated_unit"]
    )
    
    graph.create_decision_node(
        'stag_treat', 
        condition=has_staggered_treatment.transform({'panel_cleaned_data': 'df'}), 
        predecessors=["multi_post_periods"]
    )


def create_cross_sectional_decision_structure(graph: ExecutableGraph, predecessor: str = "basic_cleaning") -> None:
    """Create the decision structure for the cross-sectional data path."""

    # === CROSS-SECTIONAL DATA PATH (assumes non-multi-period data) ===

    
    graph.create_node('cross_sectional_cleaned_data', action_function=cross_sectional_clean_node.transform({predecessor: 'data_input'}), predecessors=[predecessor])
    



def configure_panel_decision_paths(graph: ExecutableGraph) -> None:
    """Configure the routing logic for the panel data decision structure."""
    
    # Panel analysis routing
    graph.when_true("single_treated_unit", "synthdid_spec")
    graph.when_false("single_treated_unit", "multi_post_periods")

    # Multi-unit routing  
    graph.when_true("multi_post_periods", "stag_treat")
    graph.when_false("multi_post_periods", "did_spec")

    # Staggered treatment routing
    graph.when_true("stag_treat", "stag_spec")
    graph.when_false("stag_treat", "event_spec")

    # Callaway & Sant'Anna method selection
    graph.when_true("has_never_treated", "cs_never_treated")
    graph.when_false("has_never_treated", "cs_not_yet_treated")


# Some utils for the graph
@make_transformable
def basic_clean_node(df: pd.DataFrame, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1) -> pd.DataFrame:
    return create_basic_cleaner(df, missing_strategy=missing_strategy, max_missing_fraction=max_missing_fraction)

@make_transformable
def panel_clean_node(data_input: pd.DataFrame, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1) -> pd.DataFrame:
    return create_panel_cleaner(data_input, missing_strategy=missing_strategy, max_missing_fraction=max_missing_fraction)

@make_transformable
def cross_sectional_clean_node(data_input: pd.DataFrame, missing_strategy: str = "drop_rows", max_missing_fraction: float = 0.1) -> pd.DataFrame:
    return create_cross_sectional_cleaner(data_input, missing_strategy=missing_strategy, max_missing_fraction=max_missing_fraction)