"""Exercise policies and string identifiers through the public graph factories."""

import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.example_graph import create_cross_sectional_graph, create_panel_graph
from pyautocausal.pipelines.mock_data import generate_mock_data
from pyautocausal.data_validation.validator_base import DataValidationError


def test_factory_missing_threshold_reaches_pipeline_nodes(tmp_path):
    data = generate_mock_data(n_units=80, n_periods=1, n_treated=40, staggered_treatment=False)
    data.loc[data.index[::5], "y"] = np.nan
    graph = create_cross_sectional_graph(tmp_path, max_missing_fraction=0.25)
    graph.fit(df=data)
    cleaned = graph.get("cross_sectional_cleaned_data").get_result_value()
    assert len(cleaned) == 64
    assert sum(s["rows_dropped"] for s in cleaned.attrs["cleaning_metadata"]) == 16
    assert graph.get("ols_stand").state.name == "COMPLETED"


def test_numeric_and_string_unit_labels_produce_equivalent_public_panel_estimates(minimum_wage_data, tmp_path):
    original_fixture = minimum_wage_data.copy(deep=True)
    data = minimum_wage_data.rename(columns={"countyreal": "id_unit", "year": "t", "lemp": "y"})
    data["treat"] = ((data["first.treat"] > 0) & (data["t"] >= data["first.treat"])).astype(int)
    data = data[["id_unit", "t", "treat", "y", "lpop"]]
    models = []
    for labels in ("numeric", "string"):
        frame = data.copy(deep=True)
        if labels == "string":
            frame["id_unit"] = frame["id_unit"].map(lambda value: f"county-{value}").astype("string")
        original = frame.copy(deep=True)
        graph = create_panel_graph(tmp_path / labels)
        graph.fit(df=frame)
        cleaned = graph.get("panel_cleaned_data").get_result_value()
        pd.testing.assert_series_equal(cleaned.id_unit, original.id_unit)
        pd.testing.assert_frame_equal(frame, original)
        assert graph.get("cs_never_treated").state.name == "COMPLETED"
        assert not [node.name for node in graph.nodes() if node.state.name == "FAILED"]
        models.append(graph.get("cs_never_treated").get_result_value().model)

    reference, relabeled = models
    for field in ("group", "year"):
        np.testing.assert_array_equal(reference["att_gt_object"].results[field],
                                      relabeled["att_gt_object"].results[field])
    np.testing.assert_allclose(reference["att_gt_object"].results["att"],
                               relabeled["att_gt_object"].results["att"], rtol=0, atol=1e-12)
    for aggregation in ("overall_effect", "dynamic_effects", "group_effects"):
        expected = reference[aggregation].atte
        actual = relabeled[aggregation].atte
        np.testing.assert_allclose(actual["overall_att"], expected["overall_att"], rtol=0, atol=1e-12)
        if expected["att_egt"] is not None:
            np.testing.assert_array_equal(actual["egt"], expected["egt"])
            np.testing.assert_allclose(actual["att_egt"], expected["att_egt"], rtol=0, atol=1e-12)
    # Relabeling can reorder units assigned finite bootstrap draws; this checks
    # point-estimate invariance, not Monte Carlo equality of standard errors.
    pd.testing.assert_frame_equal(minimum_wage_data, original_fixture)


@pytest.mark.parametrize("kwargs", [{"missing_strategy": "fill"}, {"max_missing_fraction": 1.5}])
def test_invalid_factory_policy_rejected_at_construction(tmp_path, kwargs):
    with pytest.raises(ValueError):
        create_panel_graph(tmp_path, **kwargs)


def test_factory_reject_policy_stops_before_estimation(tmp_path):
    data = pd.DataFrame({"treat": [0, 0, 1, 1], "y": [1., None, 2., 3.]})
    graph = create_cross_sectional_graph(tmp_path, missing_strategy="reject", max_missing_fraction=1)
    with pytest.raises(DataValidationError, match="exceeding threshold"):
        graph.fit(df=data)
    assert graph.get("basic_cleaning").state.name == "FAILED"
    assert graph.get("ols_stand").state.name != "COMPLETED"
