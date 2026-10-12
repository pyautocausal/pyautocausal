"""Optional, cross-fitted partially linear DoubleML wrapper.

API: https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLR.html
The parameter is a partially linear treatment coefficient, not a CATE.
"""
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def double_ml_estimation(
    df: pd.DataFrame, outcome: str = 'y', treatment: str = 'treat',
    covariates: Optional[List[str]] = None, ml_learner: str = 'random_forest',
    n_folds: int = 5, n_rep: int = 1, random_state: int = 42,
) -> str:
    """Fit a partially linear model and return its statistical summary.

    Install pyautocausal[doubleml]. Inputs must be finite and numeric, with
    at least one covariate and varying treatment. Lasso scaling is fitted
    inside each training fold. Splits use a local RNG, not global state.
    """
    try:
        from doubleml import DoubleMLData, DoubleMLPLR
    except ImportError as exc:
        raise ImportError("Install the optional dependency with: pip install 'pyautocausal[doubleml]'") from exc
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError('df must be a DataFrame with unique column names')
    if outcome == treatment:
        raise ValueError('Outcome and treatment must be different columns')
    covariates = [c for c in df.columns if c not in [outcome, treatment]] if covariates is None else list(covariates)
    if not covariates:
        raise ValueError('DoubleML requires at least one explicitly identified covariate')
    if len(set(covariates)) != len(covariates) or set(covariates) & {outcome, treatment}:
        raise ValueError('Covariates must be unique and exclude outcome and treatment')
    columns = [outcome, treatment, *covariates]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'Required columns not found: {missing}')
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in columns):
        raise ValueError('DoubleML requires numeric outcome, treatment and covariates')
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError('DoubleML inputs must be finite and contain no missing values')
    if df[treatment].nunique() < 2:
        raise ValueError('Treatment must contain at least two distinct values')
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 2 <= n_folds <= len(df):
        raise ValueError('n_folds must be an integer between 2 and the sample size')
    if isinstance(n_rep, bool) or not isinstance(n_rep, int) or n_rep < 1:
        raise ValueError('n_rep must be a positive integer')
    min_train_size = len(df) - int(np.ceil(len(df) / n_folds))
    if min_train_size < 2:
        raise ValueError('Each training fold must contain at least two observations')
    if ml_learner == 'random_forest':
        ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
        ml_m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    elif ml_learner == 'lasso':
        cv = min(5, min_train_size)
        ml_l = make_pipeline(StandardScaler(), LassoCV(cv=cv, random_state=random_state))
        ml_m = make_pipeline(StandardScaler(), LassoCV(cv=cv, random_state=random_state))
    else:
        raise ValueError("ml_learner must be 'random_forest' or 'lasso'")
    data = DoubleMLData(df[columns].copy(), y_col=outcome, d_cols=treatment, x_cols=covariates)
    model = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, n_rep=n_rep, draw_sample_splitting=False)
    rng = np.random.default_rng(random_state)
    splits = [list(KFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1))).split(df)) for _ in range(n_rep)]
    model.set_sample_splitting(splits)
    model.fit()
    coefficient = float(np.asarray(model.coef).reshape(-1)[0])
    se = float(np.asarray(model.se).reshape(-1)[0])
    t_stat = float(np.asarray(model.t_stat).reshape(-1)[0])
    pvalue = float(np.asarray(model.pval).reshape(-1)[0])
    low, high = model.confint(level=0.95).iloc[0].to_numpy(dtype=float)
    if not np.isfinite([coefficient, se, t_stat, pvalue, low, high]).all():
        raise ValueError('DoubleML returned non-finite inference; check sample size and identification')
    return '\n'.join([
        'Double Machine Learning Estimation Results', '=' * 40,
        'Estimand: partially linear treatment coefficient', f'ML Learner: {ml_learner}',
        f'Number of observations: {len(df)}', f'Number of covariates: {len(covariates)}',
        f"Covariates: {', '.join(map(str, covariates))}",
        f'Cross-fitting: {n_folds} folds, {n_rep} repetitions; seed={random_state}',
        f'Coefficient: {coefficient:.4f}', f'Std. Error: {se:.4f}',
        f't-statistic: {t_stat:.4f}', f'P-value: {pvalue:.4f}',
        f'95% Confidence Interval: [{low:.4f}, {high:.4f}]',
    ])
