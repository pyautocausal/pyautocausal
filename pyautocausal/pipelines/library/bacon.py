"""
Goodman-Bacon Decomposition

This module provides functions for performing the Goodman-Bacon decomposition for
differences-in-differences with variation in treatment timing, with or without
time-varying covariates.

This is a Python implementation of the R bacon() function originally developed in the
R bacondecomp package.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import patsy
from typing import List, Dict, Tuple, Union, Optional


def bacon(formula: str, data: pd.DataFrame, id_var: str, time_var: str, 
          quietly: bool = True) -> Union[pd.DataFrame, Dict]:
    """
    Performs the Goodman-Bacon decomposition for differences-in-differences with 
    variation in treatment timing.

    Parameters
    ----------
    formula : str
        Formula in patsy syntax. Must be of the form y ~ D + controls, where y is the outcome
        variable, D is the binary treatment indicator, and controls can be any additional
        control variables.
    data : pd.DataFrame
        DataFrame containing the variables in the model.
    id_var : str
        Name of the ID variable for units.
    time_var : str
        Name of the time variable.
    quietly : bool, default=True
        If True, does not print the summary of estimates/weights by type.

    Returns
    -------
    Union[pd.DataFrame, Dict]
        If control variables are included in the formula, returns a dictionary with:
            - 'Omega': weight of the within timing group coefficient
            - 'beta_hat_w': within timing group coefficient
            - 'two_by_twos': DataFrame with covariate adjusted 2x2 estimates and weights
        
        If no control variables are included, returns only the two_by_twos DataFrame.
    """
    # Unpack variable names from formula
    vars_dict = unpack_variable_names(formula)
    outcome_var = vars_dict['outcome_var']
    treated_var = vars_dict['treated_var']
    control_vars = vars_dict['control_vars']
    
    # Rename variables for consistency with original implementation
    df = data.copy()
    df = rename_vars(df, id_var, time_var, outcome_var, treated_var)
    
    # Check for NA observations
    required_cols = ['id', 'time', 'outcome', 'treated']
    has_nas = df[required_cols].isna().any().any()
    
    if len(control_vars) > 0:
        control_cols = [c for c in control_vars if c in df.columns]
        has_nas_control = df[control_cols].isna().any().any() if control_cols else False
        has_nas = has_nas or has_nas_control
    
    if has_nas:
        raise ValueError("NA observations found in data")
    
    # Create treatment groups
    treatment_group_result = create_treatment_groups(df, control_vars, return_merged_df=True)
    two_by_twos = treatment_group_result['two_by_twos']
    df = treatment_group_result['data']
    
    # Uncontrolled case
    if len(control_vars) == 0:
        # Iterate through treatment group dyads
        for i in range(len(two_by_twos)):
            treated_group = two_by_twos.iloc[i]['treated']
            untreated_group = two_by_twos.iloc[i]['untreated']
            
            # Subset data for this comparison
            data1 = subset_data(df, treated_group, untreated_group)
            
            # Calculate weight and estimate
            weight = calculate_weights(data1, treated_group, untreated_group)
            
            # Use statsmodels for fixed effects regression
            formula = "outcome ~ treated + C(time) + C(id)"
            model = sm.OLS.from_formula(formula, data=data1)
            results = model.fit()
            estimate = results.params['treated']
            
            # Store results
            two_by_twos.at[i, 'estimate'] = estimate
            two_by_twos.at[i, 'weight'] = weight
        
        # Rescale weights to sum to 1
        two_by_twos = scale_weights(two_by_twos)
        
        if not quietly:
            bacon_summary(two_by_twos)
            
        return two_by_twos
    
    else:  # Controlled case
        # Run FWL (Frisch-Waugh-Lovell) procedure 
        control_formula = "+".join(control_vars)
        df = run_fwl(df, control_formula)
        
        # Calculate within treatment group estimate and its weight
        Omega = calculate_Omega(df)
        beta_hat_w = calculate_beta_hat_w(df)
        
        # Collapse controls and predicted treatment to treatment group/year level
        collapsed_result = collapse_x_p(df, control_vars)
        df = collapsed_result['data']
        g_control_formula = collapsed_result['g_control_formula']
        
        # Iterate through treatment group dyads
        for i in range(len(two_by_twos)):
            treated_group = two_by_twos.iloc[i]['treated']
            untreated_group = two_by_twos.iloc[i]['untreated']
            
            # Subset data for this comparison
            data1 = df[df['treat_time'].isin([treated_group, untreated_group])].copy()
            
            # Calculate between group estimate and weight
            weight_est = calc_controlled_beta_weights(data1, g_control_formula)
            s_kl = weight_est['s_kl']
            beta_hat_d_bkl = weight_est['beta_hat_d_bkl']
            
            # Store results
            two_by_twos.at[i, 'weight'] = s_kl
            two_by_twos.at[i, 'estimate'] = beta_hat_d_bkl
        
        # Rescale weights to sum to 1
        two_by_twos = scale_weights(two_by_twos)
        
        if not quietly:
            bacon_summary(two_by_twos)
        
        # Return results
        return {
            'beta_hat_w': beta_hat_w,
            'Omega': Omega,
            'two_by_twos': two_by_twos
        }


def bacon_summary(two_by_twos: pd.DataFrame, return_df: bool = False) -> Optional[pd.DataFrame]:
    """
    Summarizes the two-by-two output produced by bacon() to produce average 2x2 estimate 
    and total weight for the following three comparisons:
    - Earlier vs. Later (Good)
    - Treated vs. Untreated (Good)
    - Later vs. Earlier (Bad)
    
    Parameters
    ----------
    two_by_twos : pd.DataFrame
        DataFrame produced by bacon().
    return_df : bool, default=False
        If True, the summary DataFrame is returned.
        
    Returns
    -------
    Optional[pd.DataFrame]
        If return_df is True, returns the summary DataFrame.
    """
    summary = two_by_twos.groupby('type')['weight'].sum().reset_index()
    
    # Calculate weighted averages by type
    def weighted_mean(group):
        return np.average(group['estimate'], weights=group['weight'])
    
    avg_est = two_by_twos.groupby('type').apply(weighted_mean).reset_index()
    avg_est.columns = ['type', 'avg_est']
    
    # Merge summary statistics
    summary = pd.merge(summary, avg_est, on='type')
    
    # Round for better display
    summary['weight'] = summary['weight'].round(5)
    summary['avg_est'] = summary['avg_est'].round(5)
    
    print(summary)
    
    if return_df:
        return summary
    return None


def unpack_variable_names(formula: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse a formula string to extract variable names.
    
    Parameters
    ----------
    formula : str
        Model formula string (outcome ~ treatment + controls)
        
    Returns
    -------
    Dict
        Dictionary with keys:
        - outcome_var: Name of outcome variable
        - treated_var: Name of treatment variable
        - control_vars: List of control variable names
    """
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")
    
    outcome_var = parts[0].strip()
    right_side = parts[1].strip()
    
    # Split right side by + and get first term as treatment
    right_side_vars = [v.strip() for v in right_side.split('+')]
    treated_var = right_side_vars[0]
    control_vars = [v for v in right_side_vars[1:] if v]  # Skip empty strings
    
    return {
        'outcome_var': outcome_var,
        'treated_var': treated_var,
        'control_vars': control_vars
    }


def rename_vars(data: pd.DataFrame, id_var: str, time_var: str, 
                outcome_var: str, treated_var: str) -> pd.DataFrame:
    """
    Rename variables in the DataFrame for consistent internal naming.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    id_var : str
        Name of ID variable
    time_var : str
        Name of time variable
    outcome_var : str
        Name of outcome variable
    treated_var : str
        Name of treatment variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed variables
    """
    df = data.copy()
    rename_dict = {
        id_var: 'id',
        time_var: 'time',
        outcome_var: 'outcome',
        treated_var: 'treated'
    }
    return df.rename(columns=rename_dict)


def create_treatment_groups(data: pd.DataFrame, control_vars: List[str], 
                           return_merged_df: bool = False) -> Dict:
    """
    Create treatment groups based on first time of treatment.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with 'id', 'time', and 'treated' columns
    control_vars : List[str]
        List of control variable names
    return_merged_df : bool, default=False
        If True, return the modified DataFrame along with two_by_twos
        
    Returns
    -------
    Dict
        Dictionary with 'two_by_twos' DataFrame and optionally 'data'
    """
    # Get first treatment time for each ID
    df_treat = data[data['treated'] == 1].copy()
    if df_treat.empty:
        raise ValueError("No treated units found in data")
    
    # Get min treatment time for each ID
    first_treatment = df_treat.groupby('id')['time'].min().reset_index()
    first_treatment.columns = ['id', 'treat_time']
    
    # Merge back to original data
    df = pd.merge(data, first_treatment, on='id', how='left')
    
    # Set treatment time to infinity for never treated units
    INF = float('inf')
    df['treat_time'] = df['treat_time'].fillna(INF)
    
    # Check for weakly increasing treatment
    valid_treatment = (
        (df['treat_time'] == INF) | 
        ((df['time'] >= df['treat_time']) & (df['treated'] == 1)) |
        ((df['time'] < df['treat_time']) & (df['treated'] == 0))
    )
    
    if not valid_treatment.all():
        raise ValueError("Treatment not weakly increasing with time")
    
    # Create DataFrame of all possible 2x2 estimates
    if len(control_vars) == 0:
        # For uncontrolled case, we create all dyad combinations
        treatment_times = sorted(df['treat_time'].unique())
        
        # Create all combinations of treatment times
        treated_groups = []
        untreated_groups = []
        for t1 in treatment_times:
            for t2 in treatment_times:
                if t1 != t2:
                    treated_groups.append(t1)
                    untreated_groups.append(t2)
        
        two_by_twos = pd.DataFrame({
            'treated': treated_groups,
            'untreated': untreated_groups
        })
        
        # Remove entries with treated == INF and first time period
        min_time = data['time'].min()
        two_by_twos = two_by_twos[
            (two_by_twos['treated'] != INF) & 
            (two_by_twos['treated'] != min_time)
        ]
        
        # Add placeholder columns
        two_by_twos['estimate'] = np.nan
        two_by_twos['weight'] = np.nan
        
        # Classify estimate type
        two_by_twos['type'] = np.where(
            two_by_twos['untreated'] == INF, 
            'Treated vs Untreated',
            np.where(
                two_by_twos['untreated'] == min_time,
                'Later vs Always Treated',
                np.where(
                    two_by_twos['treated'] > two_by_twos['untreated'],
                    'Later vs Earlier Treated',
                    'Earlier vs Later Treated'
                )
            )
        )
    else:
        # For controlled case, each dyad appears once
        two_by_twos = pd.DataFrame(columns=['treated', 'untreated', 'weight', 'estimate'])
        treatment_times = sorted([t for t in df['treat_time'].unique() if t != INF])
        
        for i in treatment_times:
            for j in treatment_times:
                if j > i:
                    two_by_twos = pd.concat([
                        two_by_twos, 
                        pd.DataFrame({
                            'treated': [i], 
                            'untreated': [j], 
                            'weight': [np.nan],
                            'estimate': [np.nan]
                        })
                    ], ignore_index=True)
        
        # Add treatment vs never treated
        for i in treatment_times:
            two_by_twos = pd.concat([
                two_by_twos, 
                pd.DataFrame({
                    'treated': [i], 
                    'untreated': [INF], 
                    'weight': [np.nan],
                    'estimate': [np.nan]
                })
            ], ignore_index=True)
        
        # Classify estimate type
        min_time = data['time'].min()
        two_by_twos['type'] = np.where(
            two_by_twos['untreated'] == INF, 
            'Treated vs Untreated',
            np.where(
                (two_by_twos['treated'] == min_time) | 
                (two_by_twos['untreated'] == min_time),
                'Later vs Always Treated',
                'Both Treated'
            )
        )
    
    if return_merged_df:
        return {'two_by_twos': two_by_twos, 'data': df}
    else:
        return {'two_by_twos': two_by_twos}


def subset_data(data: pd.DataFrame, treated_group: float, untreated_group: float) -> pd.DataFrame:
    """
    Subset data for a specific treatment group comparison.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with 'treat_time' column
    treated_group : float
        Treatment time for the treated group
    untreated_group : float
        Treatment time for the untreated group
        
    Returns
    -------
    pd.DataFrame
        Subset of data for the comparison
    """
    # Select only observations from these two groups
    df = data[data['treat_time'].isin([treated_group, untreated_group])].copy()
    
    # Additional time restrictions based on comparison type
    if treated_group < untreated_group:
        # For early vs late (before late is treated), restrict to times before late treatment
        df = df[df['time'] < untreated_group]
    elif treated_group > untreated_group:
        # For late vs early (after early is treated), restrict to times after early treatment
        df = df[df['time'] >= untreated_group]
    
    return df


def calculate_weights(data: pd.DataFrame, treated_group: float, 
                     untreated_group: float) -> float:
    """
    Calculate weights for a 2x2 decomposition.
    
    Parameters
    ----------
    data : pd.DataFrame
        Subset of data for this comparison
    treated_group : float
        Treatment time for the treated group
    untreated_group : float
        Treatment time for the untreated group
        
    Returns
    -------
    float
        Weight for this comparison
    """
    INF = float('inf')
    
    if untreated_group == INF:
        # Treated vs untreated
        n_u = (data['treat_time'] == untreated_group).sum()
        n_k = (data['treat_time'] == treated_group).sum()
        n_ku = n_k / (n_k + n_u)
        
        D_k = data[data['treat_time'] == treated_group]['treated'].mean()
        V_ku = n_ku * (1 - n_ku) * D_k * (1 - D_k)
        
        weight = (n_k + n_u)**2 * V_ku
        
    elif treated_group < untreated_group:
        # Early vs late (before late is treated)
        n_k = (data['treat_time'] == treated_group).sum()
        n_l = (data['treat_time'] == untreated_group).sum()
        n_kl = n_k / (n_k + n_l)
        
        D_k = data[data['treat_time'] == treated_group]['treated'].mean()
        D_l = data[data['treat_time'] == untreated_group]['treated'].mean()
        
        V_kl = n_kl * (1 - n_kl) * (D_k - D_l) / (1 - D_l) * (1 - D_k) / (1 - D_l)
        weight = ((n_k + n_l) * (1 - D_l))**2 * V_kl
        
    elif treated_group > untreated_group:
        # Late vs early (after early is treated)
        n_k = (data['treat_time'] == untreated_group).sum()
        n_l = (data['treat_time'] == treated_group).sum()
        n_kl = n_k / (n_k + n_l)
        
        D_k = data[data['treat_time'] == untreated_group]['treated'].mean()
        D_l = data[data['treat_time'] == treated_group]['treated'].mean()
        
        V_kl = n_kl * (1 - n_kl) * (D_l / D_k) * (D_k - D_l) / D_k
        weight = ((n_k + n_l) * D_k)**2 * V_kl
    
    return weight


def scale_weights(two_by_twos: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale weights to sum to 1.
    
    Parameters
    ----------
    two_by_twos : pd.DataFrame
        DataFrame with 'weight' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rescaled weights
    """
    sum_weight = two_by_twos['weight'].sum()
    two_by_twos['weight'] = two_by_twos['weight'] / sum_weight
    return two_by_twos


def run_fwl(data: pd.DataFrame, control_formula: str) -> pd.DataFrame:
    """
    Run Frisch-Waugh-Lovell procedure to partial out controls from treatment.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data 
    control_formula : str
        Formula with control variables
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for residualized treatment
    """
    df = data.copy()
    
    # First stage regression of treatment on controls with fixed effects
    formula = f"treated ~ {control_formula} + C(time) + C(id)"
    model = sm.OLS.from_formula(formula, data=df)
    fit_fwl = model.fit()
    
    # Predicted values and residuals
    df['p'] = fit_fwl.predict()
    df['d'] = fit_fwl.resid
    df['d_it'] = df['d']
    
    # Demean residuals
    # Unit means
    df['d_i_bar'] = df.groupby('id')['d'].transform('mean')
    # Time means
    df['d_t_bar'] = df.groupby('time')['d_it'].transform('mean')
    # Overall mean
    df['d_bar_bar'] = df['d_it'].mean()
    
    # Double-demeaned residuals
    df['d_it_til'] = df['d_it'] - df['d_i_bar'] - df['d_t_bar'] + df['d_bar_bar']
    
    # Treatment group-time means
    df['d_kt_bar'] = df.groupby(['treat_time', 'time'])['d_it'].transform('mean')
    # Treatment group means
    df['d_k_bar'] = df.groupby('treat_time')['d_it'].transform('mean')
    
    # Treatment group-time demeaned residuals
    df['d_ikt_til'] = df['d_it'] - df['d_i_bar'] - (df['d_kt_bar'] - df['d_k_bar'])
    df['d_kt_til'] = (df['d_kt_bar'] - df['d_k_bar']) - (df['d_t_bar'] - df['d_bar_bar'])
    
    return df


def calculate_Omega(data: pd.DataFrame) -> float:
    """
    Calculate weight of the within timing group coefficient.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with demeaned residuals
        
    Returns
    -------
    float
        Weight of the within timing group coefficient
    """
    N = len(data)
    Vd_w = np.var(data['d_ikt_til'], ddof=1) * (N - 1) / N
    V_d = np.var(data['d_it_til'], ddof=1) * (N - 1) / N
    Omega = Vd_w / V_d
    return Omega


def calculate_beta_hat_w(data: pd.DataFrame) -> float:
    """
    Calculate within timing group coefficient.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with demeaned residuals
        
    Returns
    -------
    float
        Within timing group coefficient
    """
    N = len(data)
    C = np.cov(data['outcome'], data['d_ikt_til'], ddof=1)[0, 1] * (N - 1) / N
    Vd_w = np.var(data['d_ikt_til'], ddof=1) * (N - 1) / N
    beta_hat_w = C / Vd_w
    return beta_hat_w


def collapse_x_p(data: pd.DataFrame, control_vars: List[str]) -> Dict:
    """
    Collapse controls and predicted treatment to treatment group/year level.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with controls and predicted treatment
    control_vars : List[str]
        List of control variable names
        
    Returns
    -------
    Dict
        Dictionary with processed data and control formula
    """
    df = data.copy()
    
    # Create group-level controls by averaging within treatment time-year groups
    for var in control_vars:
        if var in df.columns:
            df[f'g_{var}'] = df.groupby(['treat_time', 'time'])[var].transform('mean')
    
    # Group level predicted treatment
    df['g_p'] = df.groupby(['treat_time', 'time'])['p'].transform('mean')
    
    # Create formula for group controls
    g_control_vars = [f'g_{var}' for var in control_vars if var in df.columns]
    g_control_formula = '+'.join(g_control_vars) if g_control_vars else "1"
    
    return {'data': df, 'g_control_formula': g_control_formula}


def calc_controlled_beta_weights(data: pd.DataFrame, g_control_formula: str) -> Dict:
    """
    Calculate between group estimates and weights for controlled case.
    
    Parameters
    ----------
    data : pd.DataFrame
        Subset of data for this comparison
    g_control_formula : str
        Formula with group-level controls
        
    Returns
    -------
    Dict
        Dictionary with s_kl (weight) and beta_hat_d_bkl (estimate)
    """
    # Calculate VD
    df = data.copy()
    formula = "treated ~ 1 + C(time) + C(id)"
    fit_D = sm.OLS.from_formula(formula, data=df)
    results_D = fit_D.fit()
    df['Dtilde'] = results_D.resid
    N = len(df)
    VD = np.var(df['Dtilde'], ddof=1) * (N - 1) / N
    
    # Partial out group-level controls
    Rsq = 0
    if g_control_formula != "1":
        # Create formula for each group control to partial out time and unit fixed effects
        g_vars = g_control_formula.split('+')
        g_vars = [var.strip() for var in g_vars]
        
        # Add column for each partialled group variable
        for var in g_vars:
            if var != "1":  # Skip intercept term
                g_formula = f"{var} ~ 1 + C(time) + C(id)"
                fit_g = sm.OLS.from_formula(g_formula, data=df)
                results_g = fit_g.fit()
                df[f'p_{var}'] = results_g.resid
        
        # Regress Dtilde on partialled group controls
        p_g_vars = [f'p_{var}' for var in g_vars if var != "1"]
        
        if p_g_vars:
            p_g_formula = f"Dtilde ~ {'+'.join(p_g_vars)}"
            fit_pgj = sm.OLS.from_formula(p_g_formula, data=df)
            results_pgj = fit_pgj.fit()
            df['pgjtilde'] = results_pgj.predict()
            Rsq = results_pgj.rsquared
    else:
        df['pgjtilde'] = 0
    
    # Calculate Vdp
    formula = "g_p ~ 1 + C(time) + C(id)"
    fit_p = sm.OLS.from_formula(formula, data=df)
    results_p = fit_p.fit()
    df['ptilde'] = results_p.resid
    df['dp'] = df['pgjtilde'] - df['ptilde']
    Vdp = np.var(df['dp'], ddof=1) * (N - 1) / N
    
    # Calculate BD - coefficient from OLS with full controls
    if g_control_formula != "1":
        formula = f"outcome ~ treated + {g_control_formula} + C(id) + C(time)"
    else:
        formula = "outcome ~ treated + C(id) + C(time)"
    
    fit_BD = sm.OLS.from_formula(formula, data=df)
    results_BD = fit_BD.fit()
    BD = results_BD.params['treated']
    
    # Calculate Bb - coefficient from regression on dp
    formula = "outcome ~ dp + C(time) + C(id)"
    fit_Bb = sm.OLS.from_formula(formula, data=df)
    results_Bb = fit_Bb.fit()
    Bb = results_Bb.params['dp']
    
    # Calculate final estimate and weight
    beta_hat_d_bkl = ((1 - Rsq) * VD * BD + Vdp * Bb) / ((1 - Rsq) * VD + Vdp)
    s_kl = N**2 * ((1 - Rsq) * VD + Vdp)
    
    return {'s_kl': s_kl, 'beta_hat_d_bkl': beta_hat_d_bkl}

# Helper functions will be added here