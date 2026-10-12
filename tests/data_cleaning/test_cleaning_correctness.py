"""Regression tests for information-preserving cleaning and explicit policies."""

import json

import numpy as np
import pandas as pd
import pytest

from pyautocausal.data_cleaner_interface.autocleaner import AutoCleaner
from pyautocausal.data_cleaning.base import CleaningPlan
from pyautocausal.data_cleaning.hints import StandardizeTimePeriodHint, UpdateColumnTypesHint
from pyautocausal.data_cleaning.operations.time_operations import StandardizeTimePeriodsOperation
from pyautocausal.data_cleaning.operations.schema_operations import UpdateColumnTypesOperation
from pyautocausal.data_validation.checks.missing_data_checks import MissingDataCheck, MissingDataConfig
from pyautocausal.data_validation.validator_base import DataValidationError
from pyautocausal.pipelines.example_graph.core import (
    create_basic_cleaner, create_panel_cleaner, create_cross_sectional_cleaner,
)


@pytest.fixture
def panel():
    return pd.DataFrame({
        "id_unit": ["treated-A"] * 3 + ["control-02"] * 3,
        "t": ["2020-01-01 09:00", "2020-01-01 10:00", "2020-01-02 09:00"] * 2,
        "treat": [0, 1, 1, 0, 0, 0],
        "y": [1., 12., 13., 1., 2., 3.],
    })


@pytest.mark.parametrize("time_kind", ["strings", "datetime", "timezone", "nanoseconds", "large_integers"])
def test_panel_preserves_distinct_periods_and_string_ids(panel, time_kind):
    if time_kind == "datetime":
        panel["t"] = pd.to_datetime(panel["t"])
    elif time_kind == "timezone":
        panel["t"] = pd.to_datetime(panel["t"]).dt.tz_localize("Europe/Berlin")
    elif time_kind == "nanoseconds":
        panel["t"] = list(pd.date_range("2020-01-01", periods=3, freq="ns")) * 2
    elif time_kind == "large_integers":
        panel["t"] = [2**53, 2**53 + 1, 2**53 + 2] * 2
    panel["id_unit"] = panel["id_unit"].astype("string")
    before = panel.copy(deep=True)
    cleaned = create_panel_cleaner(panel)
    assert cleaned["t"].tolist() == [1, 2, 3] * 2
    assert not cleaned.duplicated(["id_unit", "t"]).any()
    pd.testing.assert_series_equal(cleaned.id_unit, before.id_unit)
    pd.testing.assert_frame_equal(panel, before)
    transformations = cleaned.attrs["cleaning_metadata"][-1]["full_metadata"]["transformations"]
    mapping = next(row for row in transformations if row["operation"] == "standardize_time_periods")["details"]
    assert len(mapping["value_mapping"]) == 3
    assert mapping["treatment_start_period"] == 2
    json.dumps(cleaned.attrs["cleaning_metadata"])
    # Mapping is based on chronology, independent of the order in the frame.
    reordered = create_panel_cleaner(panel.sample(frac=1, random_state=3)).sort_index()
    pd.testing.assert_frame_equal(cleaned, reordered)


def test_same_instant_aliases_are_rejected():
    data = pd.DataFrame({"time": ["2020-01-01 09:00Z", "2020-01-01 10:00+01:00"], "treatment": [0, 1]})
    with pytest.raises(DataValidationError, match="same period"):
        AutoCleaner().standardize_time_periods().clean(data)


def test_time_operation_never_leaves_unmapped_values_untouched():
    with pytest.raises(ValueError, match="Unmapped values"):
        StandardizeTimePeriodsOperation().apply(
            pd.DataFrame({"t": [1, 2, 3]}),
            StandardizeTimePeriodHint(time_column="t", value_mapping={1: 1, 2: 2}),
        )


def test_missing_time_can_be_dropped_without_silent_standardization_failure():
    data = pd.DataFrame({"time": [None, "2020-01-01 09:00", "2020-01-01 10:00"], "treatment": [0, 0, 1]})
    cleaned = (AutoCleaner().standardize_time_periods()
               .check_for_missing_data(max_missing_fraction=0.5).clean(data))
    assert cleaned.time.tolist() == [1, 2]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_missingness_thresholds_fail_immediately(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        MissingDataConfig(max_missing_fraction=threshold)


def test_empty_or_no_checked_columns_have_no_missing_hints():
    for frame, columns in [(pd.DataFrame(), None), (pd.DataFrame({"x": [None]}), [])]:
        result = MissingDataCheck(config=MissingDataConfig(check_columns=columns)).validate(frame)
        assert result.passed
        assert not result.cleaning_hints


def test_threshold_boundary_reject_and_drop_provenance():
    data = pd.DataFrame({"id_unit": ["keep-1", "drop", "keep-2", "keep-3"],
                         "treat": [0, 0, 1, 1], "y": [1., None, 3., 4.]})
    with pytest.raises(DataValidationError, match="exceeding threshold"):
        create_basic_cleaner(data)
    cleaned = create_basic_cleaner(data, max_missing_fraction=0.25)
    assert len(cleaned) == 3
    stage = next(s for s in cleaned.attrs["cleaning_metadata"] if s["rows_dropped"])
    assert stage["units_dropped"] == ["drop"]
    details = stage["full_metadata"]["transformations"][0]["details"]
    assert details["dropped_row_positions"] == [1]
    assert details["dropped_row_indices"] == [1]
    assert details["missing_counts"]["y"] == 1
    with pytest.raises(DataValidationError, match="exceeding threshold"):
        create_basic_cleaner(data, missing_strategy="reject", max_missing_fraction=1)


def test_missing_treatment_is_dropped_before_integer_cast():
    data = pd.DataFrame({"treat": [0, 1, None], "y": [1., 2., 3.]})
    cleaned = create_basic_cleaner(data, max_missing_fraction=0.5)
    assert cleaned.treat.tolist() == [0, 1]
    assert cleaned.treat.dtype == np.dtype("int64")


def test_dropping_entire_treatment_group_is_rejected():
    data = pd.DataFrame({"treat": [0, 0, 1], "y": [1., 2., None]})
    with pytest.raises(DataValidationError, match="both treatment groups") as raised:
        create_basic_cleaner(data, max_missing_fraction=1)
    assert raised.value.cleaning_metadata["post_validation_passed"] is False


def test_missing_covariate_cannot_remove_only_control_group():
    data = pd.DataFrame({"treat": [0, 0, 1, 1], "y": [1., 2., 3., 4.], "x": [None, None, 1., 2.]})
    with pytest.raises(DataValidationError, match="both treatment groups"):
        create_cross_sectional_cleaner(data, max_missing_fraction=1)


def test_panel_missingness_cannot_remove_all_pre_treatment_data(panel):
    panel.loc[0, "y"] = np.nan
    with pytest.raises(DataValidationError, match="pre-treatment periods"):
        create_panel_cleaner(panel, max_missing_fraction=1)


def test_panel_rejects_missing_comparison_group_after_drop(panel):
    panel.loc[panel.id_unit == "control-02", "y"] = np.nan
    with pytest.raises(DataValidationError, match="at least two units"):
        create_panel_cleaner(panel, max_missing_fraction=1)


def test_panel_rejects_duplicate_keys_instead_of_silently_dropping(panel):
    data = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate unit-time"):
        create_panel_cleaner(data)


def test_panel_rejects_treatment_reversal(panel):
    panel.loc[2, "treat"] = 0
    with pytest.raises(DataValidationError, match="treatment reversals"):
        create_panel_cleaner(panel)


@pytest.mark.parametrize("dtype", ["int64", "bool", "boolean", "uint8"])
def test_design_check_rejects_reversals_for_each_binary_dtype(panel, dtype):
    panel.loc[2, "treat"] = 0
    panel["treat"] = panel["treat"].astype(dtype)
    with pytest.raises(DataValidationError, match="treatment reversals"):
        (AutoCleaner().check_causal_design(unit_column="id_unit", time_column="t")
         .clean(panel))


def test_identical_cross_sectional_rows_are_valid_observations():
    data = pd.DataFrame({"treat": [0, 0, 1, 1], "y": [1., 1., 2., 2.]})
    assert len(create_basic_cleaner(data)) == 4


def test_cleaning_plan_stops_on_failed_operation():
    plan = CleaningPlan()
    plan.add_operation(UpdateColumnTypesOperation(), UpdateColumnTypesHint(type_mapping={"x": int}))
    with pytest.raises(RuntimeError, match="update_column_types"):
        plan(pd.DataFrame({"x": ["not a number"]}))
    assert plan.get_metadata().transformations[0].details["error"]


def test_fill_respects_explicit_empty_and_ignored_columns():
    data = pd.DataFrame({"x": [None], "y": [None]})
    empty = AutoCleaner().check_for_missing_data(strategy="fill", check_columns=[], fill_value=0).clean(data)
    pd.testing.assert_frame_equal(data, empty)
    ignored = (AutoCleaner().check_for_missing_data(strategy="fill", ignore_columns=["y"],
                                                   fill_value=0, max_missing_fraction=1).clean(data))
    assert ignored.x.tolist() == [0]
    assert ignored.y.isna().all()
