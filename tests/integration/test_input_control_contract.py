"""Control selection must survive default cleaning without changing the model."""

import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.example_graph import create_cross_sectional_graph
from pyautocausal.orchestration.nodes import NodeExecutionError
from pyautocausal.pipelines.example_graph.core import create_panel_cleaner
from pyautocausal.pipelines.library.estimators import fit_ols
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, create_did_specification,
    create_event_study_specification, create_staggered_did_specification,
    create_synthdid_specification, create_uplift_specification,
)


def confounded_sample():
    data = pd.DataFrame({'x': [0] * 12 + [1] * 12,
                         'treat': [0] * 10 + [1] * 2 + [0] * 2 + [1] * 10})
    data['y'] = 2 * data.treat + 10 * data.x
    return data


@pytest.mark.parametrize('id_kind', ['numeric', 'string'])
def test_public_factory_preserves_binary_confounder_and_excludes_labels(tmp_path, id_kind):
    data = confounded_sample()
    data['id_unit'] = np.arange(len(data))
    if id_kind == 'string':
        data['id_unit'] = data.id_unit.astype(str)
    data['t'] = 2020
    original = data.copy(deep=True)
    graph = create_cross_sectional_graph(tmp_path)
    graph.fit(df=data)
    result = graph.get('ols_stand').get_result_value()
    assert result.control_cols == ['x']
    assert result.model.params['treat'] == pytest.approx(2)
    assert result.data.x.dtype == original.x.dtype
    pd.testing.assert_frame_equal(data, original)


def test_direct_cross_sectional_spec_excludes_custom_id_and_time():
    data = confounded_sample().assign(person=np.arange(24), period=2020)
    spec = create_cross_sectional_specification(data, unit_col='person', time_col='period')
    assert spec.control_cols == ['x']
    assert fit_ols(spec).model.params['treat'] == pytest.approx(2)


def test_uplift_inference_excludes_structural_labels():
    data = confounded_sample().assign(id_unit=np.arange(24), t=2020)
    data['y'] = data.x
    assert create_uplift_specification(data).control_cols == ['x']


@pytest.mark.parametrize('dtype', ['category', 'object', 'string'])
def test_automatic_inference_refuses_to_silently_drop_categorical_control(tmp_path, dtype):
    data = confounded_sample()
    data['x'] = data.x.map({0: 'low', 1: 'high'}).astype(dtype)
    with pytest.raises(ValueError, match='Encode categorical controls explicitly'):
        create_cross_sectional_specification(data)
    graph = create_cross_sectional_graph(tmp_path)
    with pytest.raises(NodeExecutionError, match='Encode categorical controls explicitly') as raised:
        graph.fit(df=data)
    assert isinstance(raised.value.original_exception, ValueError)
    assert graph.get('ols_stand').state.name != 'COMPLETED'
    # Deliberate formula controls retain the categorical adjustment.
    spec = create_cross_sectional_specification(data, control_cols=['x'])
    if dtype != 'string':  # Patsy does not support pandas' extension string dtype.
        assert fit_ols(spec).model.params['treat'] == pytest.approx(2)


def panel():
    return pd.DataFrame([
        {'id_unit': unit, 't': time, 'treat': int(unit == 0 and time >= 2),
         'x': (unit + time) % 2, 'y': unit + time + 3 * int(unit == 0 and time >= 2)}
        for unit in range(3) for time in range(4)
    ])


def test_panel_cleaning_preserves_low_cardinality_numeric_controls():
    data = panel()
    cleaned = create_panel_cleaner(data)
    assert cleaned.x.dtype == data.x.dtype
    spec = create_synthdid_specification(cleaned)
    assert spec.control_cols == ['x']
    assert spec.X.shape == (3, 4, 1)


@pytest.mark.parametrize('constructor', [create_event_study_specification, create_staggered_did_specification])
@pytest.mark.parametrize('name', ['treatment_time', 'relative_time', 'event_post0', 'event_period0', 'event_pre2'])
def test_generated_columns_preserve_input_controls(constructor, name):
    data = panel().rename(columns={'x': name})
    original = data.copy(deep=True)
    spec = constructor(data, control_cols=[name])
    pd.testing.assert_series_equal(spec.data[name], original[name])
    assert spec.control_cols == [name]
    assert name not in spec.event_time_map
    if name == 'treatment_time' and hasattr(spec, 'treatment_time_col'):
        assert spec.treatment_time_col != name
    if name == 'relative_time' and hasattr(spec, 'relative_time_col'):
        assert spec.relative_time_col != name
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('name', ['post', 'treat_post'])
def test_did_generated_columns_preserve_input_controls(name):
    data = panel().rename(columns={'x': name})
    original = data.copy(deep=True)
    spec = create_did_specification(data, control_cols=[name])
    pd.testing.assert_series_equal(spec.data[name], original[name])
    assert spec.control_cols == [name]
    if name == 'post':
        assert spec.post_col != name
    else:
        assert '_pac_treat_post' in spec.formula
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('constructor', [create_event_study_specification, create_staggered_did_specification])
@pytest.mark.parametrize('dtype', ['bool', 'boolean', 'uint8'])
def test_direct_panel_specs_reject_treatment_reversals_for_binary_dtypes(constructor, dtype):
    data = panel()
    data.loc[(data.id_unit == 0) & (data.t == 3), 'treat'] = 0
    data['treat'] = data.treat.astype(dtype)
    with pytest.raises(ValueError, match='no treatment reversals'):
        constructor(data, control_cols=[])


def test_factory_missing_threshold_uses_original_sample_for_all_columns(tmp_path):
    data = confounded_sample()
    data.loc[0, 'y'] = np.nan
    data.loc[1, 'x'] = np.nan
    graph = create_cross_sectional_graph(tmp_path, max_missing_fraction=1 / len(data))
    graph.fit(df=data)
    cleaned = graph.get('cross_sectional_cleaned_data').get_result_value()
    assert len(cleaned) == len(data) - 2
    assert sum(stage['rows_dropped'] for stage in cleaned.attrs['cleaning_metadata']) == 2
    assert graph.get('ols_stand').get_result_value().model.params['treat'] == pytest.approx(2)


@pytest.mark.parametrize('constructor, name', [
    (create_event_study_specification, 'event_period0'),
    (create_staggered_did_specification, 'event_post0'),
])
def test_event_plot_uses_generated_event_columns_not_named_controls(constructor, name):
    from pyautocausal.pipelines.library.plots import _extract_event_study_results
    spec = constructor(panel().rename(columns={'x': name}), control_cols=[name])
    params = pd.Series({column: float(time + 3) for column, time in spec.event_time_map.items()})
    params[name] = 999.
    ci = pd.DataFrame({'lower': params - .1, 'upper': params + .1})
    result = _extract_event_study_results(spec, params, ci)
    assert result.loc[result.period == 0, 'coef'].tolist() == [3.]
    assert 999. not in result.coef.tolist()
    assert result.loc[result.period == -1, 'coef'].tolist() == [0.]


@pytest.fixture(scope='module')
def cs_collision_reference():
    from pyautocausal.pipelines.library.estimators import fit_callaway_santanna_estimator
    rng = np.random.default_rng(41)
    rows = []
    for unit in range(120):
        cohort = 3 if unit < 40 else 4 if unit < 80 else 0
        x = rng.normal(loc=.8 if cohort else 0.)
        for time in range(1, 6):
            treatment = int(cohort > 0 and time >= cohort)
            rows.append({'id_unit': unit, 't': time, 'x': x, 'treat': treatment,
                         'y': unit * .01 + time * (x + 1) + 2 * treatment + rng.normal(scale=.1)})
    data = pd.DataFrame(rows)
    result = fit_callaway_santanna_estimator(
        create_staggered_did_specification(data, control_cols=['x'])).model['att_gt_object']
    return data, np.asarray(result.results['att'])


@pytest.mark.parametrize('name', ['treatment_time', 'relative_time', 'event_post0'])
def test_cs_result_preserves_selected_control_even_when_name_matches_generated_column(cs_collision_reference, name):
    from pyautocausal.pipelines.library.estimators import fit_callaway_santanna_estimator
    data, reference = cs_collision_reference
    renamed = data.rename(columns={'x': name})
    spec = create_staggered_did_specification(renamed, control_cols=[name])
    fitted = fit_callaway_santanna_estimator(spec).model['att_gt_object']
    np.testing.assert_allclose(fitted.results['att'], reference, atol=1e-10)
    pd.testing.assert_series_equal(spec.data[name], renamed[name])
