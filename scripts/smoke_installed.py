"""Run with python -I outside the repository after installing the built wheel."""
import argparse
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyautocausal
from pyautocausal import ExecutableGraph
from pyautocausal import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['base', 'all'], required=True)
    args = parser.parse_args()
    assert 'site-packages' in str(Path(pyautocausal.__file__).resolve()), pyautocausal.__file__
    assert pyautocausal.__version__ == '0.1.2'
    for name in datasets.__all__:
        assert not getattr(datasets, name).empty
    assert datasets.minimum_wage.shape == (2500, 5)
    assert 'country' in datasets.PENN.columns
    from pyautocausal.pipelines.library.specifications import create_cross_sectional_specification
    from pyautocausal.pipelines.library.estimators import fit_ols
    frame = pd.DataFrame({'treat': [0, 0, 1, 1], 'y': [1., 2., 11., 12.]})
    fitted = fit_ols(create_cross_sectional_specification(frame, control_cols=[]))
    assert abs(fitted.model.params['treat'] - 10) < 1e-8
    with tempfile.TemporaryDirectory() as directory:
        def answer() -> int:
            return 42
        graph = ExecutableGraph().configure_runtime(output_path=directory)
        graph.create_node('answer', answer, save_node=True).fit()
        assert graph.get('answer').get_result_value() == 42
        if args.mode == 'all':
            import nbformat
            from pyautocausal.persistence.notebook_runner import run_notebook_and_create_html
            path = Path(directory) / 'smoke.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(6 * 7)')]), path)
            html = run_notebook_and_create_html(path)
            assert html.exists() and '42' in html.read_text()
            from pyautocausal.causal_methods.double_ml import double_ml_estimation
            rng = np.random.default_rng(12)
            x = rng.normal(size=300)
            d = x + rng.normal(size=300)
            frame = pd.DataFrame({'x': x, 'treat': d, 'y': 2*d + x + rng.normal(scale=.2, size=300)})
            result = double_ml_estimation(frame, ml_learner='lasso', n_folds=3)
            assert '95% Confidence Interval:' in result
        else:
            assert importlib.util.find_spec('doubleml') is None
            assert importlib.util.find_spec('nbconvert') is None
            from pyautocausal.causal_methods.double_ml import double_ml_estimation
            try:
                double_ml_estimation(pd.DataFrame())
            except ImportError as exc:
                assert 'pyautocausal[doubleml]' in str(exc)
            else:
                raise AssertionError('Missing extra must produce an actionable error')
    print(f'Installed {args.mode} wheel smoke checks passed: {pyautocausal.__file__}')


if __name__ == '__main__':
    main()
