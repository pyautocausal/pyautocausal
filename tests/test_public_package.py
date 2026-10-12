import re
import numpy as np
import pandas as pd
import pytest
from pyautocausal.causal_methods.double_ml import double_ml_estimation


def sample(seed=410, n=500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = 0.6 * x + rng.normal(size=n)
    y = 2 * d + 0.8 * x + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({'y': y, 'treat': d, 'x': x})


@pytest.mark.parametrize('learner', ['lasso', 'random_forest'])
def test_doubleml_known_effect_and_reproducibility(learner):
    pytest.importorskip('doubleml')
    frame = sample()
    original = frame.copy(deep=True)
    before = np.random.get_state()
    first = double_ml_estimation(frame, ml_learner=learner, n_folds=3, n_rep=2)
    second = double_ml_estimation(frame, ml_learner=learner, n_folds=3, n_rep=2)
    after = np.random.get_state()
    assert first == second
    coefficient = float(re.search(r'Coefficient: ([-\d.]+)', first).group(1))
    assert coefficient == pytest.approx(2, abs=0.15)
    assert '95% Confidence Interval:' in first
    pd.testing.assert_frame_equal(frame, original)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize('kwargs', [{'covariates': []}, {'n_folds': 1}, {'n_rep': 0}, {'covariates': ['missing']}])
def test_invalid_doubleml_inputs(kwargs):
    pytest.importorskip('doubleml')
    with pytest.raises(ValueError):
        double_ml_estimation(sample(), **kwargs)


def test_every_public_dataset_resolves():
    import pyautocausal.datasets as data
    for name in data.__all__:
        assert not getattr(data, name).empty
    assert list(data.PENN.columns) == ['country', 'year', 'log_gdp', 'dem', 'educ']
