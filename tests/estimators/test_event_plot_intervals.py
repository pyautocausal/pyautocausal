"""Rendered event plots must retain uncertainty for estimated zero effects."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyautocausal.pipelines.library.plots import event_study_plot
from pyautocausal.pipelines.library.specifications import EventStudySpec, StaggeredDiDSpec


@pytest.mark.parametrize('spec_type', [EventStudySpec, StaggeredDiDSpec])
@pytest.mark.parametrize('reference_period', [-1, -2])
def test_zero_effect_keeps_its_confidence_interval_in_rendered_plot(spec_type, reference_period):
    # Collision-safe names also exercise the explicit coefficient/event map.
    parameters = pd.Series({'_pac_event_period0': 0., 'event_post1': 1.})
    intervals = pd.DataFrame({'lower': [-.5, .5], 'upper': [.5, 1.5]},
                             index=parameters.index)
    model = SimpleNamespace(params=parameters, conf_int=lambda **kwargs: intervals)
    fields = {
        'data': pd.DataFrame(), 'formula': '', 'outcome_col': 'y',
        'treatment_cols': ['treat'], 'control_cols': [], 'time_col': 't',
        'unit_col': 'id_unit', 'model': model, 'reference_period': reference_period,
        'event_time_map': {'_pac_event_period0': 0, 'event_post1': 1},
    }
    if spec_type is EventStudySpec:
        fields.update(relative_time_col='relative_time', event_cols=parameters.index.tolist())
    else:
        fields.update(treatment_time_col='treatment_time')
    figure = event_study_plot(spec_type(**fields))
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        segments = [segment for collection in axis.collections
                    for segment in collection.get_segments() if len(segment)]
        assert len(segments) == 2
        np.testing.assert_allclose(segments[0], [[0., -.5], [0., .5]])
        np.testing.assert_allclose(segments[1], [[1., .5], [1., 1.5]])
        # The reference remains a separate zero marker without an error bar.
        reference = next(line for line in axis.lines if line.get_label() == 'Reference period')
        np.testing.assert_array_equal(reference.get_xdata(), [reference_period])
        np.testing.assert_array_equal(reference.get_ydata(), [0.])
        assert all(segment[0, 0] != reference_period for segment in segments)
    finally:
        plt.close(figure)
