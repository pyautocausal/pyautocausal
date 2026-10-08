"""
Estimator functions for causal inference models.
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import patsy
from statsmodels.base.model import Results
import copy 
import re
from linearmodels import PanelOLS, RandomEffects, FirstDifferenceOLS, BetweenOLS
from pyautocausal.pipelines.library.synthdid.synthdid import synthdid_estimate
from pyautocausal.pipelines.library.synthcontrol.SyntheticControlMethods import Synth
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.pipelines.library.specifications import (
    BaseSpec,
    DiDSpec,
    EventStudySpec,
    StaggeredDiDSpec,
    SynthDIDSpec,
)
from pyautocausal.pipelines.library.base_estimator import format_statsmodels_result
from pyautocausal.pipelines.library.csdid.att_gt import ATTgt
from pyautocausal.pipelines.library.synthdid.vcov import synthdid_se
from pyautocausal.pipelines.library.specifications import UpliftSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import warnings


def create_model_matrices(spec: BaseSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create model matrices from a specification using patsy.
    
    Args:
        spec: A specification dataclass with data and formula
        
    Returns:
        Tuple of (y, X) arrays for modeling
    """
    data = spec.data
    formula = spec.formula
    
    # Parse the formula into outcome and predictors
    outcome_expr, predictors_expr = formula.split('~', 1)
    outcome_expr = outcome_expr.strip()
    predictors_expr = predictors_expr.strip()
    
    # Create design matrices
    y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
    
    return y, X


# Experimental probability-based uplift estimators. Class balancing changes
# target probabilities and must not be used for potential outcome estimation.
def _validate_uplift_data(X, y, t):
    if len(X) != len(y) or len(X) != len(t):
        raise ValueError("X, y, and t must have the same length")
    if not np.isin(t, [0, 1]).all() or len(np.unique(t)) != 2:
        raise ValueError("Treatment must contain both binary arms (0, 1)")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Outcome must be binary (0, 1)")
    if not np.isfinite(X.astype(float)).all():
        raise ValueError("Uplift features must be finite numeric values")
    if min(np.sum(t == 0), np.sum(t == 1)) < 10:
        warnings.warn("Fewer than ten observations in a treatment arm", UserWarning)


def _uplift_arrays(spec):
    X = spec.data[spec.control_cols].to_numpy(dtype=float)
    y = spec.data[spec.outcome_col].to_numpy()
    t = spec.data[spec.treatment_cols[0]].to_numpy()
    _validate_uplift_data(X, y, t)
    # A constant feature supports the explicitly unadjusted model.
    if X.shape[1] == 0:
        X = np.ones((len(y), 1))
    return X, y, t


def _outcome_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_split=50, min_samples_leaf=20,
                                  random_state=42)


def _positive_probability(model, X):
    """predict_proba for class 1, including single-outcome-class arms."""
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, np.flatnonzero(classes == 1)[0]]


def _store_uplift(spec, name, cate, models):
    spec.model_type = name.replace('_', '-')
    spec.models = spec.model = models
    spec.cate_estimates = {name: cate}
    spec.ate_estimate = float(np.mean(cate))
    # Resampling already fitted predictions conditions on the fitted models and
    # is not an interval for the causal ATE. No interval is claimed here.
    spec.ate_ci = None
    spec.inference = {
        'status': 'unavailable', 'estimand': 'sample-average predicted CATE',
        'reason': 'Model-fitting uncertainty is not estimated for this experimental meta-learner.'
    }
    return spec


@make_transformable
def fit_s_learner(spec: UpliftSpec) -> UpliftSpec:
    """Experimental S-learner; point predictions only (ATE inference unavailable)."""
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    X, y, t = _uplift_arrays(spec)
    model = _outcome_forest()
    model.fit(np.column_stack([X, t]), y)
    treated = _positive_probability(model, np.column_stack([X, np.ones(len(X))]))
    control = _positive_probability(model, np.column_stack([X, np.zeros(len(X))]))
    return _store_uplift(spec, 's_learner', treated - control, {'s_model': model})


@make_transformable
def fit_t_learner(spec: UpliftSpec) -> UpliftSpec:
    """Experimental T-learner; retains natural outcome class probabilities."""
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    X, y, t = _uplift_arrays(spec)
    treated, control = _outcome_forest(), _outcome_forest()
    treated.fit(X[t == 1], y[t == 1])
    control.fit(X[t == 0], y[t == 0])
    cate = _positive_probability(treated, X) - _positive_probability(control, X)
    return _store_uplift(spec, 't_learner', cate,
                         {'treated_model': treated, 'control_model': control})


@make_transformable
def fit_x_learner(spec: UpliftSpec) -> UpliftSpec:
    """Experimental X-learner (Künzel et al.); no causal interval is claimed."""
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    from sklearn.ensemble import RandomForestRegressor
    X, y, t = _uplift_arrays(spec)
    treated, control = _outcome_forest(), _outcome_forest()
    treated.fit(X[t == 1], y[t == 1])
    control.fit(X[t == 0], y[t == 0])
    pseudo_treated = y[t == 1] - _positive_probability(control, X[t == 1])
    pseudo_control = _positive_probability(treated, X[t == 0]) - y[t == 0]
    params = dict(n_estimators=200, max_depth=8, min_samples_split=30,
                  min_samples_leaf=15, random_state=42)
    tau_treated, tau_control = RandomForestRegressor(**params), RandomForestRegressor(**params)
    tau_treated.fit(X[t == 1], pseudo_treated)
    tau_control.fit(X[t == 0], pseudo_control)
    propensity = LogisticRegression(random_state=42, max_iter=1000).fit(X, t)
    e = _positive_probability(propensity, X)
    # g(x) weights the effect model fitted on controls (tau_0).
    cate = e * tau_control.predict(X) + (1 - e) * tau_treated.predict(X)
    spec.propensity_score = e
    return _store_uplift(spec, 'x_learner', cate, {
        'treated_model': treated, 'control_model': control,
        'tau_treated_model': tau_treated, 'tau_control_model': tau_control,
        'propensity_model': propensity,
    })


@make_transformable
def fit_double_ml_binary(spec: UpliftSpec) -> UpliftSpec:
    """Experimental cross-fitted AIPW ATE for independent observations.

    Uses separate outcome regressions and the orthogonal ATE score. The normal
    interval requires unconfoundedness, overlap and adequate nuisance-model
    convergence. It does not estimate CATE or support clustered observations.
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    from sklearn.model_selection import StratifiedKFold
    X, y, t = _uplift_arrays(spec)
    if min(np.sum(t == 0), np.sum(t == 1)) < 5:
        raise ValueError("Cross-fitting requires at least five observations in each arm")
    if spec.unit_col in spec.data and spec.data[spec.unit_col].duplicated().any():
        raise ValueError("AIPW inference requires independent units; repeated unit IDs are unsupported")
    mu0, mu1, propensity = (np.zeros(len(y)) for _ in range(3))
    fold_models = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in folds.split(X, t):
        control, treated = _outcome_forest(), _outcome_forest()
        control.fit(X[train][t[train] == 0], y[train][t[train] == 0])
        treated.fit(X[train][t[train] == 1], y[train][t[train] == 1])
        prop = LogisticRegression(random_state=42, max_iter=1000).fit(X[train], t[train])
        mu0[test] = _positive_probability(control, X[test])
        mu1[test] = _positive_probability(treated, X[test])
        propensity[test] = _positive_probability(prop, X[test])
        fold_models.append({'control_model': control, 'treated_model': treated,
                            'propensity_model': prop})
    if np.any((propensity <= 0.01) | (propensity >= 0.99)):
        raise ValueError("Estimated propensity scores violate required overlap (0.01, 0.99)")
    scores = mu1 - mu0 + t * (y - mu1) / propensity - (1 - t) * (y - mu0) / (1 - propensity)
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    from scipy.stats import norm
    margin = norm.ppf(0.975) * se
    spec.model_type = 'double-ml'
    spec.models = spec.model = {'fold_models': fold_models}
    spec.cate_estimates = None
    spec.ate_estimate = ate
    spec.ate_ci = (ate - margin, ate + margin)
    spec.propensity_score = propensity
    spec.inference = {
        'status': 'asymptotic', 'method': 'cross-fitted AIPW influence function',
        'estimand': 'ATE', 'standard_error': se, 'confidence_level': 0.95,
        'assumptions': 'Independent units, unconfoundedness, overlap, and consistent nuisance fits at sufficient rates.',
        'cate_status': 'unavailable',
    }
    return spec


def extract_treatment_from_design(X: pd.DataFrame, treatment_col: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract treatment variable from design matrix and return treatment and controls separately.
    
    Args:
        X: Design matrix containing treatment and control variables
        treatment_col: Name of the treatment column (in original data)
        
    Returns:
        Tuple of (treatment array, control design matrix)
    """
    # Look for columns that start with the treatment column name
    # This handles Patsy's transformations while avoiding partial matches
    treatment_cols = [col for col in X.columns if col.startswith(treatment_col)]
    
    if not treatment_cols:
        raise ValueError(f"Treatment column '{treatment_col}' not found in design matrix")
    
    # If there are multiple matching columns (e.g., interaction terms)
    if len(treatment_cols) > 1:
        # Prefer exact match if available
        exact_matches = [col for col in treatment_cols if col == treatment_col]
        if exact_matches:
            treatment_col_name = exact_matches[0]
        else:
            # Otherwise use the first match
            treatment_col_name = treatment_cols[0]
    else:
        treatment_col_name = treatment_cols[0]
    
    # Extract treatment
    d = X[treatment_col_name].values
    
    # Create control matrix without the treatment
    X_controls = X.drop(columns=treatment_col_name)
    
    return d, X_controls


@make_transformable
def fit_ols(spec: BaseSpec, weights: Optional[np.ndarray] = None) -> Results:
    """
    Estimate treatment effect using OLS regression.
    
    Args:
        spec: A specification dataclass with data and formula
        weights: Optional sample weights for weighted regression
        
    Returns:
        Specification with model field set to the fitted model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    # We can still use the formula interface directly for OLS
    data = spec.data
    formula = spec.formula
    
    
    if weights is not None:
        model = sm.WLS.from_formula(formula, data=data, weights=weights).fit(cov_type='HC1')
    else:
        model = sm.OLS.from_formula(formula, data=data).fit(cov_type='HC1')
    
    # Set the model field (ensure all spec types handle this correctly)
    spec.model = model
            
    return spec


@make_transformable
def format_regression_results(model_result: Results) -> str:
    """
    Format regression model results as a readable string.
    
    Args:
        model_result: Statsmodels regression results object
        
    Returns:
        Formatted string with model summary
    """
    return format_statsmodels_result(model_result)


@make_transformable
def fit_weighted_ols(spec: BaseSpec) -> Results:
    """
    Estimate treatment effect using Weighted OLS regression.
    
    Args:
        spec: A specification dataclass with data and formula
        weights: Sample weights for the regression
        
    Returns:
        Fitted statsmodels fit_weighted_ols model results
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    if "weights" in spec.data.columns:
        weights = spec.data['weights']
    else:
        raise ValueError("Weights must be provided for weighted OLS")
    
    # Ensure weights are valid (no NaN, inf, or zero sum)
    if np.isnan(weights).any() or np.isinf(weights).any() or np.sum(weights) == 0:
        # In case of invalid weights, use uniform weights as fallback
        weights = np.ones_like(weights) / len(weights)
        
    # Ensure weights are properly scaled
    if np.sum(weights) != 1.0 and np.sum(weights) != 0.0:
        weights = weights / np.sum(weights)
        
    return fit_ols(spec, weights)


@make_transformable
def fit_double_lasso(
    spec: BaseSpec,
    alpha: float = 0.1,
    cv_folds: int = 5
) -> Results:
    """
    Estimate treatment effect using Double/Debiased Lasso.
    
    Implementation follows the algorithm from:
    Chernozhukov et al. (2018) - Double/Debiased Machine Learning
    
    Args:
        spec: A specification dataclass with data and column information
        alpha: Regularization strength for Lasso
        cv_folds: Number of cross-validation folds
            
    Returns:
        Fitted OLS model for the final stage regression
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    # Use patsy to create design matrices
    y_df, X_df = create_model_matrices(spec)
    y = y_df.values.ravel()  # Convert to 1D array
    
    # Extract treatment and controls
    d, X_controls = extract_treatment_from_design(X_df, spec.treatment_cols[0])
    
    # Check if we have control variables
    if X_controls.shape[1] == 0:
        raise ValueError("Control variables are required for double lasso")
    
    # Initialize cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Arrays to store residuals
    y_resid = np.zeros_like(y)
    d_resid = np.zeros_like(d)
    
    # Cross-fitting
    for train_idx, test_idx in kf.split(X_controls):
        # Split data
        X_train, X_test = X_controls.iloc[train_idx], X_controls.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]
        
        # First stage: outcome equation
        lasso_y = Lasso(alpha=alpha, random_state=42)
        lasso_y.fit(X_train, y_train)
        y_pred = lasso_y.predict(X_test)
        y_resid[test_idx] = y_test - y_pred
        
        # First stage: treatment equation
        lasso_d = Lasso(alpha=alpha, random_state=42)
        lasso_d.fit(X_train, d_train)
        d_pred = lasso_d.predict(X_test)
        d_resid[test_idx] = d_test - d_pred
    
    # Second stage: treatment effect estimation
    final_model = sm.OLS(
        y_resid,
        sm.add_constant(d_resid)
    ).fit(cov_type='HC1')
    
    return final_model



def _cs_backend_spec(spec):
    """Protect input columns from temporary names used by the C&S backend."""
    reserved = {'Intercept', 'w', 'w1', 'rowid', 'C', 'G_m', 'y_main',
                'y0', 'y1', 'dy', 'never_treated'}
    columns = list(dict.fromkeys([
        spec.outcome_col, spec.unit_col, spec.time_col, spec.treatment_time_col,
        *(spec.control_cols or []),
    ]))
    occupied = set(spec.data.columns)
    mapping = {}
    for col in columns:
        if col in reserved:
            alias = f'__pyautocausal_cs_{len(mapping)}'
            while alias in occupied:
                alias += '_'
            mapping[col] = alias
            occupied.add(alias)
    backend_spec = copy.copy(spec)
    backend_spec.data = spec.data.rename(columns=mapping)
    for field in ['outcome_col', 'unit_col', 'time_col', 'treatment_time_col']:
        value = getattr(spec, field)
        setattr(backend_spec, field, mapping.get(value, value))
    backend_spec.control_cols = [mapping.get(col, col) for col in spec.control_cols or []]
    return backend_spec, mapping


def _cs_covariate_formula(spec):
    controls = list(spec.control_cols or [])
    missing = [c for c in controls if c not in spec.data]
    if missing:
        raise ValueError(f"Missing C&S covariates: {missing}")
    # The bundled estimator repeatedly builds its formula after preprocessing;
    # transformed/categorical columns are not preserved by that backend.
    for col in controls:
        if not col.isidentifier() or not pd.api.types.is_numeric_dtype(spec.data[col]):
            raise ValueError("C&S covariates must be numeric columns with identifier names")
        if not np.isfinite(spec.data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"C&S covariate {col} must be finite")
    formula = f"{spec.outcome_col} ~ " + (" + ".join(controls) if controls else "1")
    try:
        patsy.dmatrices(formula, spec.data, NA_action='raise')
    except Exception as exc:
        raise ValueError(f"Invalid C&S covariate formula: {formula}") from exc
    return formula


def _prepare_cs_data(spec):
    data = spec.data.copy()
    unit, time, cohort = spec.unit_col, spec.time_col, spec.treatment_time_col
    if data.duplicated([unit, time]).any():
        raise ValueError("C&S requires one observation per unit and time")
    if data.groupby(unit)[cohort].nunique(dropna=False).gt(1).any():
        raise ValueError("Treatment cohort must be constant within each unit")
    # Zero is reserved for never treated in the backend. Map observed dates or
    # zero-based periods and their actual treatment cohorts together.
    times = sorted(data[time].unique())
    mapping = {value: i + 1 for i, value in enumerate(times)}
    nonmissing = data[cohort].notna()
    if not data.loc[nonmissing, cohort].isin(times).all():
        raise ValueError("Treatment cohorts must be observed time values; use NaN for never treated")
    data[time] = data[time].map(mapping)
    data[cohort] = data[cohort].map(mapping).fillna(0)
    # IDs are labels; provide an invertible local encoding to numeric backend.
    data[unit] = pd.factorize(data[unit], sort=True)[0] + 1
    _cs_covariate_formula(spec)
    for col in spec.control_cols or []:
        data[col] = data[col].astype(float)
    return data


@make_transformable
def fit_callaway_santanna_estimator(spec: StaggeredDiDSpec) -> StaggeredDiDSpec:
    """
    Wrapper function to fit the Callaway and Sant'Anna (2021) DiD estimator using never-treated units as the control group.
    
    This function uses the new csdid module implementation for more robust and comprehensive results.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    backend_spec, column_mapping = _cs_backend_spec(spec)
    
    
    # Extract necessary information from spec
    data = spec.data
    outcome_col = backend_spec.outcome_col
    time_col = backend_spec.time_col
    unit_col = backend_spec.unit_col
    treatment_time_col = backend_spec.treatment_time_col
    control_cols = spec.control_cols if hasattr(spec, 'control_cols') and spec.control_cols else []
    formula = spec.formula

    # Prepare data for csdid format
    data_cs = _prepare_cs_data(backend_spec)


    # Ensure never-treated units have 0 in treatment_time_col, not NaN
    # This is required for the Callaway & Sant'Anna estimator
    data_cs[treatment_time_col] = data_cs[treatment_time_col].fillna(0)

    # create a new column that is 1 if the unit is treated at any time
    data_cs["never_treated"] = data_cs.groupby(unit_col)[treatment_time_col].transform('max') == 0


    att_gt = ATTgt(
        yname=outcome_col,
        tname=time_col,
        idname=unit_col,
        gname=treatment_time_col,
        data=data_cs,
        control_group=['nevertreated'],
        xformla=_cs_covariate_formula(backend_spec),
        panel=True,
        allow_unbalanced_panel=True,
        anticipation=0,
        cband=True,
        biters=1000,
        alp=0.05
    )
    
    # Fit the model
    att_gt.fit(est_method='dr', base_period='varying', bstrap=True)
    
    # Generate summary table
    att_gt.summ_attgt(n=4)
    
    # Create separate copies for each aggregation to avoid overwriting
    
    # Compute aggregated treatment effects
    # Overall effect
    att_gt_overall = copy.deepcopy(att_gt)
    att_gt_overall.aggte(typec="simple", bstrap=True, cband=True)
    
    # Dynamic (event study) effects  
    att_gt_dynamic = copy.deepcopy(att_gt)
    att_gt_dynamic.aggte(typec="dynamic", bstrap=True, cband=True)
    
    # Group-specific effects
    att_gt_group = copy.deepcopy(att_gt)
    att_gt_group.aggte(typec="group", bstrap=True, cband=True)
    
    # Store all results in the spec
    spec.model = {
        'att_gt_object': att_gt,
        'overall_effect': att_gt_overall,
        'dynamic_effects': att_gt_dynamic,
        'group_effects': att_gt_group,
        'column_mapping': column_mapping,
        'control_group': 'never_treated',
        'estimator': 'callaway_santanna_csdid'
    }
    
    return spec



@make_transformable
def fit_callaway_santanna_nyt_estimator(spec: StaggeredDiDSpec) -> StaggeredDiDSpec:
    """
    Wrapper function to fit the Callaway and Sant'Anna (2021) DiD estimator using never-treated units as the control group.
    
    This function uses the new csdid module implementation for more robust and comprehensive results.
    
    Args:
        spec: A StaggeredDiDSpec object with data and column information
        
    Returns:
        StaggeredDiDSpec with fitted model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    backend_spec, column_mapping = _cs_backend_spec(spec)
    
    
    # Extract necessary information from spec
    data = spec.data
    outcome_col = backend_spec.outcome_col
    time_col = backend_spec.time_col
    unit_col = backend_spec.unit_col
    treatment_time_col = backend_spec.treatment_time_col
    control_cols = spec.control_cols if hasattr(spec, 'control_cols') and spec.control_cols else []
    formula = spec.formula

    # Prepare data for csdid format
    data_cs = _prepare_cs_data(backend_spec)
    
    # Ensure never-treated units have 0 in treatment_time_col, not NaN
    # This is required for the Callaway & Sant'Anna estimator
    data_cs[treatment_time_col] = data_cs[treatment_time_col].fillna(0)


    att_gt = ATTgt(
        yname=outcome_col,
        tname=time_col,
        idname=unit_col,
        gname=treatment_time_col,
        data=data_cs,
        control_group=['notyettreated'],
        xformla=_cs_covariate_formula(backend_spec),
        panel=True,
        allow_unbalanced_panel=True,
        anticipation=0,
        cband=True,
        biters=1000,
        alp=0.05
    )
    
    # Fit the model
    att_gt.fit(est_method='dr', base_period='varying', bstrap=True)
    
    # Generate summary table
    att_gt.summ_attgt(n=4)
    
    # Create separate copies for each aggregation to avoid overwriting
    
    # Compute aggregated treatment effects
    # Overall effect
    att_gt_overall = copy.deepcopy(att_gt)
    att_gt_overall.aggte(typec="simple", bstrap=True, cband=True)
    
    # Dynamic (event study) effects  
    att_gt_dynamic = copy.deepcopy(att_gt)
    att_gt_dynamic.aggte(typec="dynamic", bstrap=True, cband=True)
    
    # Group-specific effects
    att_gt_group = copy.deepcopy(att_gt)
    att_gt_group.aggte(typec="group", bstrap=True, cband=True)
    
    # Store all results in the spec
    spec.model = {
        'att_gt_object': att_gt,
        'overall_effect': att_gt_overall,
        'dynamic_effects': att_gt_dynamic,
        'group_effects': att_gt_group,
        'column_mapping': column_mapping,
        'control_group': 'not_yet_treated',
        'estimator': 'callaway_santanna_csdid'
    }
    
    return spec






@make_transformable
def fit_synthdid_estimator(spec) -> object:
    """
    Fit a Synthetic Difference-in-Differences estimator.
    
    Args:
        spec: A SynthDIDSpec object with data and matrix information
        
    Returns:
        SynthDIDSpec with fitted model (SynthDIDEstimate object)
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    
    
    # Extract matrices from spec
    Y = spec.Y
    N0 = spec.N0
    T0 = spec.T0
    X = spec.X
    
    # Fit the synthetic DiD model
    # Fit the model and get point estimate
    estimate = synthdid_estimate(Y, N0, T0, X=X)
    
    # Inference needs enough never-treated donors; preserve the point estimate
    # and explicitly report unavailable uncertainty otherwise.
    if N0 > len(Y) - N0:
        estimate.inference = synthdid_se(estimate, method='placebo', random_state=42)
    else:
        estimate.inference = {'status': 'unavailable', 'se': None, 'ci': None,
                              'reason': 'Placebo inference requires more donors than treated units.'}

    # Store results
    spec.model = estimate  # Store the actual SynthDIDEstimate object
    
    return spec


@make_transformable
def fit_panel_ols(spec: BaseSpec) -> BaseSpec:
    """
    Fit Panel OLS regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted PanelOLS model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index([spec.unit_col, spec.time_col])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    formula = spec.formula
    
    # Set up clustering (default to entity clustering)
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model using from_formula with special effects variables
    model = PanelOLS.from_formula(formula, data_indexed, drop_absorbed=True, check_rank=False)
    
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_did_panel(spec: BaseSpec) -> BaseSpec:
    """
    Fit Difference-in-Differences using Panel OLS with entity and time fixed effects.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted PanelOLS model for DiD
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index([spec.unit_col, spec.time_col])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    formula = spec.formula
    
    # Set up clustering for DiD (cluster by entity)
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model with entity and time effects (standard DiD setup)
    model = PanelOLS.from_formula(formula, data_indexed, drop_absorbed=True, check_rank=False)
    
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)

    return spec


@make_transformable
def fit_random_effects(spec: BaseSpec) -> BaseSpec:
    """
    Fit Random Effects regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted RandomEffects model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index([spec.unit_col, spec.time_col])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: RandomEffects doesn't use EntityEffects/TimeEffects syntax, 
    # so we need to clean those if present
    formula = spec.formula
    
    # For RandomEffects, remove EntityEffects and TimeEffects if present
    # since it handles random effects differently
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Fit model
    model = RandomEffects.from_formula(formula, data_indexed)
    result = model.fit(cov_type='robust')
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_first_difference(spec: BaseSpec) -> BaseSpec:
    """
    Fit First Difference regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted FirstDifferenceOLS model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index([spec.unit_col, spec.time_col])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: FirstDifferenceOLS doesn't use EntityEffects/TimeEffects and cannot include constants
    formula = spec.formula
    
    # For FirstDifferenceOLS, remove EntityEffects, TimeEffects, and constants
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    formula = formula.replace("+ 1", "").replace("1 +", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Set up clustering
    cluster_config = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Fit model
    model = FirstDifferenceOLS.from_formula(formula, data_indexed)
    result = model.fit(**cluster_config)
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_between_estimator(spec: BaseSpec) -> BaseSpec:
    """
    Fit Between estimator regression using linearmodels.
    
    Args:
        spec: A specification object with data and column information
        
    Returns:
        Specification with model field set to the fitted BetweenOLS model
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    data = spec.data
    
    # Create MultiIndex with standard column names
    if not isinstance(data.index, pd.MultiIndex):
        data_indexed = data.set_index([spec.unit_col, spec.time_col])
    else:
        data_indexed = data.copy()
    
    # Use the formula from the spec directly (now already in linearmodels format)
    # Note: BetweenOLS doesn't use EntityEffects/TimeEffects syntax
    formula = spec.formula
    
    # For BetweenOLS, remove EntityEffects and TimeEffects if present
    formula = formula.replace("+ EntityEffects", "").replace("+ TimeEffects", "")
    formula = formula.replace("EntityEffects +", "").replace("TimeEffects +", "")
    formula = formula.replace("EntityEffects", "").replace("TimeEffects", "")
    
    # Clean up any resulting double spaces or trailing/leading operators
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'\+\s*\+', '+', formula)
    formula = re.sub(r'~\s*\+', '~', formula)
    formula = re.sub(r'\+\s*$', '', formula)
    formula = formula.strip()
    
    # Fit model
    model = BetweenOLS.from_formula(formula, data_indexed)
    result = model.fit(cov_type='robust')
    
    # Set the model field
    spec.model = result

    # Print the model summary
    print(result.summary)
    
    return spec


@make_transformable
def fit_hainmueller_synth_estimator(spec: SynthDIDSpec) -> SynthDIDSpec:
    """
    Fit a Hainmueeller Synthetic Control estimator using the SyntheticControlMethods package.
    
    Args:
        spec: A SynthDIDSpec object with data and matrix information
        
    Returns:
        SynthDIDSpec with fitted model (Synth object)
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    
    # Extract information from spec
    data = spec.data.copy()
    outcome_col = spec.outcome_col
    time_col = spec.time_col
    unit_col = spec.unit_col
    treatment_col = spec.treatment_cols[0]
    
    # Convert unit column to string to avoid TypeError in Synth library
    if pd.api.types.is_numeric_dtype(data[unit_col]):
        data[unit_col] = data[unit_col].astype(str)

    # Find the treated unit and treatment period
    treated_units = data[data[treatment_col] == 1][unit_col].unique()
    if len(treated_units) != 1:
        raise ValueError("Hainmueller synthetic control requires exactly one treated unit")

    treated_unit = treated_units[0]
    
    treatment_period = int(data[data[treatment_col] == 1][time_col].min())
    
    # Prepare control columns to exclude (treatment column and any non-numeric columns)
    exclude_columns = [treatment_col]
    for col in data.columns:
        if col not in [outcome_col, unit_col, time_col] and not pd.api.types.is_numeric_dtype(data[col]):
            exclude_columns.append(col)
    
    # Fit Synthetic Control using the SyntheticControlMethods package
    sc = Synth(
        dataset=data,
        outcome_var=outcome_col,
        id_var=unit_col, 
        time_var=time_col,
        treatment_period=treatment_period,
        treated_unit=treated_unit,
        n_optim=5,
        pen="auto",
        exclude_columns=exclude_columns,
        random_seed=42
    )
    
    # Print results for notebook display
    print(f"\nHainmueeller Synthetic Control Results:")
    print(f"Weight DataFrame:")
    print(sc.original_data.weight_df)
    print(f"\nComparison DataFrame:")
    print(sc.original_data.comparison_df.head())
    if hasattr(sc.original_data, 'pen'):
        print(f"\nPenalty parameter: {sc.original_data.pen}")
    
    # Store the Synth object in spec
    spec.hainmueller_model = sc
    
    return spec



@make_transformable
def fit_hainmueller_placebo_test(
    spec: SynthDIDSpec, n_placebo: int = 1, random_state: Optional[int] = 42
) -> SynthDIDSpec:
    """
    Perform in-space and in-time placebo tests with a reproducible time draw.

    ``random_state`` controls the local placebo-period draw; the fitted model's
    copied generator controls its optimization. Neither changes NumPy's global
    random state.
    """
    # A shared specification can feed several independent graph branches.
    spec = copy.copy(spec)
    spec.hainmueller_model = copy.deepcopy(spec.hainmueller_model)
    min_period = spec.data[spec.time_col].min()
    treatment_period = spec.data[spec.data[spec.treatment_cols[0]] == 1][spec.time_col].min()
    periods = np.sort(spec.data[spec.time_col].unique())
    eligible = periods[(periods > min_period) & (periods < treatment_period)]
    if not len(eligible):
        raise ValueError("In-time placebo requires at least two observed pre-treatment periods")
    # Choose an observed period with pretreatment data on either side, retaining
    # the existing interior buffer when the observed grid permits it.
    buffer = (treatment_period - min_period) // 4
    interior = eligible[(eligible >= min_period + buffer) &
                        (eligible < treatment_period - buffer)]
    candidates = interior if len(interior) else eligible
    placebo_period = np.random.default_rng(random_state).choice(candidates).item()
    spec.hainmueller_placebo_metadata = {
        'random_state': random_state, 'placebo_period': placebo_period,
    }

    spec.hainmueller_model.in_space_placebo(n_placebo)

    print(f"Placebo period: {placebo_period}")
    spec.hainmueller_model.in_time_placebo(placebo_period)


    return spec
    

