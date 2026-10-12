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


def test_string_unit_labels_through_callaway_santanna_graph(minimum_wage_data, tmp_path):
    data = minimum_wage_data.rename(columns={"countyreal": "id_unit", "year": "t", "lemp": "y"})
    data["treat"] = ((data["first.treat"] > 0) & (data["t"] >= data["first.treat"])).astype(int)
    data["id_unit"] = data["id_unit"].map(lambda value: f"county-{value}").astype("string")
    data = data[["id_unit", "t", "treat", "y", "lpop"]]
    graph = create_panel_graph(tmp_path)
    graph.fit(df=data)
    cleaned = graph.get("panel_cleaned_data").get_result_value()
    assert set(cleaned.id_unit) == set(data.id_unit)
    assert cleaned.id_unit.dtype == data.id_unit.dtype
    assert graph.get("cs_never_treated").state.name == "COMPLETED"
    assert not [node.name for node in graph.nodes() if node.state.name == "FAILED"]


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
