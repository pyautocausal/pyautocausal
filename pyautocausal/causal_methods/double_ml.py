from typing import Optional, Union, List
import pandas as pd
import numpy as np
from doubleml import DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def double_ml_estimation(
    df: pd.DataFrame,
    outcome: str = 'y',
    treatment: str = 'treat',
    covariates: Optional[List[str]] = None,
    ml_learner: str = 'random_forest',
    n_folds: int = 5,
    n_rep: int = 1,
    random_state: int = 42
) -> str:
    """
    Estimate treatment effects using Double Machine Learning with cross-fitting.
    
    This implementation uses the DoubleML package which implements the double/debiased 
    machine learning framework of Chernozhukov et al. (2018). It supports various ML methods
    for nuisance parameter estimation.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing outcome, treatment, and covariates
        outcome (str): Name of the outcome variable column. Default is 'y'
        treatment (str): Name of the treatment variable column. Default is 'treat'
        covariates (List[str], optional): List of covariate column names. If None, uses all columns 
            except outcome and treatment
        ml_learner (str): ML method to use for nuisance parameter estimation. 
            Options: 'random_forest', 'lasso'. Default is 'random_forest'
        n_folds (int): Number of folds for cross-fitting. Default is 5
        n_rep (int): Number of repetitions for cross-fitting. Default is 1
        random_state (int): Random seed for reproducibility. Default is 42
        
    Returns:
        str: Formatted string containing the estimation results
        
    Raises:
        ValueError: If required columns are missing or if invalid ml_learner is specified
    """
    # Input validation
    if outcome not in df.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in DataFrame")
    if treatment not in df.columns:
        raise ValueError(f"Treatment variable '{treatment}' not found in DataFrame")
    
    # If covariates not specified, use all columns except outcome and treatment
    if covariates is None:
        covariates = [col for col in df.columns if col not in [outcome, treatment]]
    
    # Validate all covariates exist in DataFrame
    missing_covs = [col for col in covariates if col not in df.columns]
    if missing_covs:
        raise ValueError(f"Covariates {missing_covs} not found in DataFrame")
    
    # Prepare data
    y = df[outcome].values
    t = df[treatment].values
    X = df[covariates].copy()
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Set up ML learners
    if ml_learner == 'random_forest':
        ml_m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
        ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    elif ml_learner == 'lasso':
        ml_m = LassoCV(random_state=random_state)
        ml_g = LassoCV(random_state=random_state)
    else:
        raise ValueError(f"Invalid ml_learner: {ml_learner}. Choose 'random_forest' or 'lasso'")
    
    # Initialize and fit DoubleML model
    dml_plr = DoubleMLPLR(
        obj_dml_data=X,
        treatment=t,
        outcome=y,
        ml_l=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        random_state=random_state
    )
    
    dml_plr.fit()
    
    # Format results
    results = []
    results.append("Double Machine Learning Estimation Results")
    results.append("=" * 40)
    results.append(f"\nML Learner: {ml_learner}")
    results.append(f"Number of observations: {len(df)}")
    results.append(f"Number of covariates: {len(covariates)}")
    results.append("\nTreatment Effect Estimates:")
    results.append("-" * 25)
    results.append(f"Coefficient: {dml_plr.coef:.4f}")
    results.append(f"Std. Error: {dml_plr.se:.4f}")
    results.append(f"t-statistic: {dml_plr.t_stat:.4f}")
    results.append(f"P-value: {dml_plr.pval:.4f}")
    results.append(f"\n95% Confidence Interval: [{dml_plr.ci[0]:.4f}, {dml_plr.ci[1]:.4f}]")
    
    return "\n".join(results) 