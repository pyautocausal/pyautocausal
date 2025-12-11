"""
Plot functions for causal inference visualizations
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
import re
import warnings
import scipy.stats as stats

from pyautocausal.pipelines.library.specifications import (
    BaseSpec, DiDSpec, StaggeredDiDSpec, EventStudySpec
)
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from pyautocausal.pipelines.library.synthdid.plot import plot_synthdid


from pyautocausal.pipelines.library.specifications import UpliftSpec

def _get_confidence_intervals(model, confidence_level: float):
    """
    Helper function to get confidence intervals from both statsmodels and linearmodels results.
    
    Args:
        model: Fitted model (statsmodels or linearmodels result)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        DataFrame with confidence intervals with standardized column names
    """
    try:
        # Try linearmodels API first (uses level parameter)
        conf_int = model.conf_int(level=confidence_level)
        # Standardize column names if they're not already 'lower' and 'upper'
        if list(conf_int.columns) == [0, 1]:
            conf_int.columns = ['lower', 'upper']
        return conf_int
    except TypeError:
        try:
            # Fall back to statsmodels API (uses alpha parameter)
            alpha = 1 - confidence_level
            conf_int = model.conf_int(alpha=alpha)
            # Standardize column names if they're not already 'lower' and 'upper'
            if list(conf_int.columns) == [0, 1]:
                conf_int.columns = ['lower', 'upper']
            return conf_int
        except Exception as e:
            raise ValueError(f"Could not get confidence intervals from model: {e}")


def _plot_uplift_curve_native(y_true, uplift_scores, treatment, ax=None, **kwargs):
    """
    Native uplift curve plotting using only matplotlib and pandas.
    
    Creates uplift curve showing treatment effect by score deciles.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Create analysis dataframe
    df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_scores,
        'treatment': treatment
    })
    
    # Sort by uplift score (descending) and create deciles
    df = df.sort_values('uplift', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(range(len(df)), 10, labels=False) + 1
    
    # Calculate metrics by decile
    decile_results = []
    print(f"DEBUG: Total observations: {len(df)}")
    print(f"DEBUG: Treatment balance: {df['treatment'].value_counts().to_dict()}")
    print(f"DEBUG: Outcome rate: {df['y'].mean():.4f}")
    
    for decile in range(1, 11):
        decile_data = df[df['decile'] == decile]
        
        # Treatment and control groups within this decile
        treated = decile_data[decile_data['treatment'] == 1]
        control = decile_data[decile_data['treatment'] == 0]
        
        # Calculate rates
        treated_rate = treated['y'].mean() if len(treated) > 0 else 0
        control_rate = control['y'].mean() if len(control) > 0 else 0
        uplift = treated_rate - control_rate
        
        # Sample sizes
        n_treated = len(treated)
        n_control = len(control)
        
        # Debug information for problematic deciles
        if n_treated == 0 or n_control == 0:
            print(f"DEBUG: Decile {decile} has imbalanced groups - Treated: {n_treated}, Control: {n_control}")
        
        # Standard error for uplift (approximate)
        if n_treated > 1 and n_control > 1:
            se_treated = np.sqrt(treated_rate * (1 - treated_rate) / n_treated) if treated_rate > 0 else 0
            se_control = np.sqrt(control_rate * (1 - control_rate) / n_control) if control_rate > 0 else 0
            uplift_se = np.sqrt(se_treated**2 + se_control**2)
        else:
            uplift_se = 0
        
        decile_results.append({
            'decile': decile,
            'uplift': uplift,
            'uplift_se': uplift_se,
            'treated_rate': treated_rate,
            'control_rate': control_rate,
            'n_treated': n_treated,
            'n_control': n_control,
            'avg_score': decile_data['uplift'].mean()
        })
    
    results_df = pd.DataFrame(decile_results)
    
    # Print diagnostic information
    print(f"DEBUG: Decile analysis summary:")
    for _, row in results_df.iterrows():
        print(f"  Decile {int(row['decile'])}: uplift={row['uplift']:.4f}, n_treated={row['n_treated']}, n_control={row['n_control']}")
    
    # Create the plot
    x = results_df['decile']
    y = results_df['uplift']
    yerr = results_df['uplift_se']
    
    # Bar plot with error bars - use different colors for different uplift levels
    colors = ['darkred' if val < -0.001 else 'darkgreen' if val > 0.001 else 'gray' for val in y]
    bars = ax.bar(x, y, alpha=0.7, color=colors, edgecolor='navy', linewidth=1)
    ax.errorbar(x, y, yerr=yerr, fmt='none', color='red', capsize=5, capthick=2)
    
    # Styling
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.set_xlabel('Uplift Score Decile (1 = Highest Predicted Uplift)', fontsize=12)
    ax.set_ylabel('Observed Uplift (Treated Rate - Control Rate)', fontsize=12)
    ax.set_title('Uplift Curve: Observed vs Predicted', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    # Force y-axis to show small values clearly
    y_max = max(0.01, max(abs(y.min()), abs(y.max())))
    ax.set_ylim(-y_max * 1.2, y_max * 1.2)
    
    # Add value labels on bars - show all values including zeros
    for i, (bar, row) in enumerate(zip(bars, results_df.itertuples())):
        height = bar.get_height()
        # Show all values, not just meaningful ones
        label_y = height + row.uplift_se + y_max * 0.05 if height >= 0 else height - row.uplift_se - y_max * 0.05
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
               f'{height:.4f}', ha='center', 
               va='bottom' if height >= 0 else 'top', 
               fontsize=9, fontweight='bold')
    
    # Add sample size annotations with treatment/control breakdown
    for i, row in results_df.iterrows():
        ax.text(row['decile'], -y_max * 1.1, 
               f'T:{row["n_treated"]}/C:{row["n_control"]}', 
               ha='center', va='top', fontsize=8, alpha=0.7)
    
    # Enhanced summary statistics in text box
    overall_ate = np.mean(y_true[treatment == 1]) - np.mean(y_true[treatment == 0])
    predicted_ate = np.mean(uplift_scores)
    
    # Count deciles with both groups
    valid_deciles = sum(1 for _, row in results_df.iterrows() if row['n_treated'] > 0 and row['n_control'] > 0)
    
    textstr = f'Overall ATE: {overall_ate:.4f}\nPredicted ATE: {predicted_ate:.4f}\nValid Deciles: {valid_deciles}/10'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax


@make_transformable
def uplift_curve_plot(spec: UpliftSpec, **kwargs) -> plt.Figure:
    """
    Create uplift curve plot using native implementation.
    """
    if spec.cate_estimates is None or spec.ate_estimate is None:
        raise ValueError("No uplift estimates found. Ensure model is fitted.")

    y_true = spec.data[spec.outcome_col].values
    uplift_scores = list(spec.cate_estimates.values())[0]
    treatment = spec.data[spec.treatment_cols[0]].values

    fig, ax = _plot_uplift_curve_native(y_true, uplift_scores, treatment, **kwargs)
    ax.set_title(f"Uplift Curve - {spec.model_type.title()}", fontsize=14, fontweight='bold')
    
    return fig




def _extract_event_study_results(spec: Union[EventStudySpec, StaggeredDiDSpec], params: pd.Series, conf_int: pd.DataFrame) -> pd.DataFrame:
    """Extract event study results from EventStudySpec or StaggeredDiDSpec."""
    periods = []
    coeffs = []
    lower_ci = []
    upper_ci = []
    
    event_pattern_pre = re.compile(r'event_pre(\d+)')   # For negative periods: event_pre1 = -1
    event_pattern_post = re.compile(r'event_post(\d+)') # For positive periods: event_post1 = +1
    
    # Determine which parameters to check
    if isinstance(spec, EventStudySpec):
        # For EventStudySpec, use the specific event columns
        param_names = sorted(spec.event_cols)
    else:
        # For StaggeredDiDSpec, check all parameters
        param_names = sorted(params.index)
    
    # Process parameters to find event study coefficients
    for param_name in param_names:
        if param_name not in params.index:
            continue
            
        # Check for pre-treatment period (pre prefix)
        match_pre = event_pattern_pre.match(param_name)
        if match_pre:
            period = -int(match_pre.group(1))  # Convert to negative number
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
            continue
            
        # Check for post-treatment period (post prefix)
        match_post = event_pattern_post.match(param_name)
        if match_post:
            period = int(match_post.group(1))  # Already positive
            periods.append(period)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
            continue
        
        # Check for event_period0 (t=0)
        if param_name == 'event_period0':
            periods.append(0)
            coeffs.append(params[param_name])
            lower_ci.append(conf_int.loc[param_name, 'lower'])
            upper_ci.append(conf_int.loc[param_name, 'upper'])
    
    # Handle reference period
    if isinstance(spec, EventStudySpec):
        # For EventStudySpec, use the specified reference period
        if hasattr(spec, 'reference_period') and spec.reference_period not in periods:
            periods.append(spec.reference_period)
            coeffs.append(0)
            lower_ci.append(0)
            upper_ci.append(0)
        elif not hasattr(spec, 'reference_period') and -1 not in periods:
            # Default to -1 as reference period if not specified
            periods.append(-1)
            coeffs.append(0)
            lower_ci.append(0)
            upper_ci.append(0)
    else:
        # For StaggeredDiDSpec and other specs, add a reference period
        # Use -1 as reference period if we have pre-treatment periods, otherwise use 0
        if any(p < 0 for p in periods) and -1 not in periods:
            # If we have pre-treatment periods, use -1 as reference
            periods.append(-1)
            coeffs.append(0.0)
            lower_ci.append(0.0)
            upper_ci.append(0.0)
        elif 0 not in periods:
            # Otherwise use 0 as reference
            periods.append(0)
            coeffs.append(0.0)
            lower_ci.append(0.0)
            upper_ci.append(0.0)
    
    return pd.DataFrame({
        'period': periods,
        'coef': coeffs,
        'lower': lower_ci,
        'upper': upper_ci
    })


@make_transformable
def event_study_plot(spec: Union[StaggeredDiDSpec, EventStudySpec], 
                    confidence_level: float = 0.95,
                    figsize: tuple = (12, 8),
                    title: str = "Event Study Plot",
                    xlabel: str = "Event Time",
                    ylabel: str = "Coefficient Estimate",
                    reference_line_color: str = "gray",
                    reference_line_style: str = "--",
                    effect_color: str = "blue",
                    marker: str = "o") -> plt.Figure:
    """
    Create an event study plot from a specification with a fitted model.
    
    Args:
        spec: A specification object with fitted model
        confidence_level: Confidence level for intervals (default: 0.95)
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        reference_line_color: Color for the zero reference line
        reference_line_style: Line style for the zero reference line
        effect_color: Color for the effect point estimates
        marker: Marker style for point estimates
        
    Returns:
        Matplotlib figure with the event study plot
    """
    # Check if model exists
    if spec.model is None:
        raise ValueError("Specification must contain a fitted model")

    # For standard OLS models
    params = spec.model.params
    conf_int = _get_confidence_intervals(spec.model, confidence_level)
    
    # Extract results from the specification
    results_df = _extract_event_study_results(spec, params, conf_int)

    # Sort by period
    results_df = results_df.sort_values('period')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate reference period from other periods for plotting
    ref_periods = results_df[results_df['coef'] == 0]
    non_ref_periods = results_df[results_df['coef'] != 0]
    
    # Plot non-reference coefficients with markers only (no connecting lines)
    ax.plot(non_ref_periods['period'], non_ref_periods['coef'], 
            marker=marker, 
            color=effect_color, 
            linestyle='none',  # This removes the connecting lines
            label='Coefficient',
            markersize=8)
    
    # Add error bars (whiskers) with same color as points
    yerr = [non_ref_periods['coef'] - non_ref_periods['lower'], 
            non_ref_periods['upper'] - non_ref_periods['coef']]
    ax.errorbar(non_ref_periods['period'], non_ref_periods['coef'], 
                yerr=yerr, 
                fmt='none',  # No additional markers
                ecolor=effect_color,  # Same color as points
                elinewidth=2,
                capsize=5,
                label=f'{int(confidence_level*100)}% CI')
    
    # Plot reference period
    ax.plot(ref_periods['period'], ref_periods['coef'], 
            marker=marker,  # Same marker as other coefficients
            color=effect_color,  # Same color as other coefficients
            linestyle='none',
            markersize=8)  # Same size as other coefficients
    
    # Add zero reference line
    ax.axhline(y=0, color=reference_line_color, linestyle=reference_line_style)
    
    # Add vertical line at period 0 (treatment time)
    ax.axvline(x=0, color=reference_line_color, linestyle=reference_line_style, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_did_plot(spec: DiDSpec, 
                   figsize: tuple = (12, 8),
                   title: str = "Difference-in-Differences Plot",
                   xlabel: str = "Time",
                   ylabel: str = "Outcome",
                   treatment_color: str = "red",
                   control_color: str = "blue",
                   confidence_level: float = 0.95) -> plt.Figure:
    """
    Create a Difference-in-Differences plot showing pre and post trends.
    
    Args:
        spec: DiDSpec object with data
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        treatment_color: Color for the treatment group line
        control_color: Color for the control group line
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Matplotlib figure with the DiD plot
    """
    # Check if data exists
    if spec.data is None:
        raise ValueError("DiD specification must contain data")
    
    # Extract column names
    data = spec.data
    time_col = spec.time_col
    unit_col = spec.unit_col
    outcome_col = spec.outcome_col
    treatment_col = spec.treatment_cols[0]  # Use first treatment column
    post_col = spec.post_col
    
    # Aggregate data by time and treatment status with standard error
    grouped = data.groupby([time_col, treatment_col])
    means = grouped[outcome_col].mean().reset_index()
    
    # Calculate standard errors for whiskers
    std_errors = grouped[outcome_col].agg(lambda x: x.std() / np.sqrt(len(x))).reset_index()
    
    # Join means and standard errors
    result = pd.merge(means, std_errors, on=[time_col, treatment_col], suffixes=('_mean', '_se'))
    
    # Calculate confidence interval multiplier
    import scipy.stats as stats
    ci_multiplier = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Calculate confidence intervals
    result['ci_lower'] = result[f'{outcome_col}_mean'] - ci_multiplier * result[f'{outcome_col}_se']
    result['ci_upper'] = result[f'{outcome_col}_mean'] + ci_multiplier * result[f'{outcome_col}_se']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot treatment group with error bars (no connecting lines)
    treated_data = result[result[treatment_col] == 1]
    ax.errorbar(treated_data[time_col], treated_data[f'{outcome_col}_mean'], 
              yerr=[treated_data[f'{outcome_col}_mean'] - treated_data['ci_lower'], 
                    treated_data['ci_upper'] - treated_data[f'{outcome_col}_mean']],
              fmt='o', color=treatment_color, ecolor=treatment_color, alpha=0.7,
              elinewidth=1.5, capsize=4, label='Treatment Group',
              linestyle='none')  # Remove connecting lines
    
    # Plot control group with error bars (no connecting lines)
    control_data = result[result[treatment_col] == 0]
    ax.errorbar(control_data[time_col], control_data[f'{outcome_col}_mean'], 
              yerr=[control_data[f'{outcome_col}_mean'] - control_data['ci_lower'], 
                    control_data['ci_upper'] - control_data[f'{outcome_col}_mean']],
              fmt='s', color=control_color, ecolor=control_color, alpha=0.7,
              elinewidth=1.5, capsize=4, label='Control Group',
              linestyle='none')  # Remove connecting lines
    
    # Find treatment timing (first period where post==1 for treated units)
    first_post_period = data[(data[treatment_col] == 1) & (data[post_col] == 1)][time_col].min()
    
    # Add vertical line for treatment timing
    if pd.notna(first_post_period):
        ax.axvline(x=first_post_period, color='gray', linestyle='--', 
                  label=f'Treatment Time ({first_post_period})')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

@make_transformable
def synthdid_plot(spec, output_config: Optional[OutputConfig] = None, **kwargs) -> plt.Figure:
    """
    Create a synthetic difference-in-differences plot.
    
    Args:
        spec: A SynthDIDSpec object with fitted model
        output_config: Configuration for saving the plot
        **kwargs: Additional arguments passed to plot_synthdid
        
    Returns:
        matplotlib figure object
    """
        
    # Create the plot using the synthdid plotting function
    fig, ax = plot_synthdid(spec.model, **kwargs)
    
    # Save if output config provided
    if output_config:
        if output_config.output_type == OutputType.PNG:
            fig.savefig(f"{output_config.output_filename}.png", 
                       dpi=300, bbox_inches='tight')
        elif output_config.output_type == OutputType.PDF:
            fig.savefig(f"{output_config.output_filename}.pdf", 
                       bbox_inches='tight')
    
    return fig


@make_transformable
def callaway_santanna_summary_table(spec: StaggeredDiDSpec) -> pd.DataFrame:
    """
    Create a comprehensive summary table for Callaway & Sant'Anna results.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model using csdid
        
    Returns:
        DataFrame with formatted results table
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    
    # Get the summary table from the att_gt object
    summary_table = att_gt.summary2.copy()
    
    # Round numerical columns for better presentation
    numerical_cols = summary_table.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        summary_table[col] = summary_table[col].round(4)
    
    return summary_table


@make_transformable
def callaway_santanna_group_event_plot(
    spec: StaggeredDiDSpec,
    figsize: tuple = (15, 10),
    title: str = "Group-Time Treatment Effects (Callaway & Sant'Anna)",
    xlabel: str = "Time Period",
    ylabel: str = "ATT Estimate",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create group-specific treatment effect plots using the csdid module.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        figsize: Figure size as (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure with group-specific plots
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    
    # Close any existing figures to avoid multiple windows
    plt.close('all')
    
    # Use the ATTgt plotting method - it will create its own figure
    fig = att_gt.plot_attgt(
        title=title,
        xlab=xlabel,
        ylab=ylabel,
        theming=True,
        ref_line=0,
        legend=True
    )
    
    # Set the figure size
    fig.set_size_inches(figsize)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


@make_transformable
def callaway_santanna_comprehensive_results(spec: StaggeredDiDSpec) -> str:
    """
    Generate simplified results summary for Callaway & Sant'Anna estimation.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        
    Returns:
        Formatted string with key results
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    overall_effect = spec.model.get('overall_effect')
    control_group = spec.model.get('control_group', 'unknown')
    
    import io
    buffer = io.StringIO()
    
    # Header
    buffer.write("=" * 60 + "\n")
    buffer.write("CALLAWAY & SANT'ANNA DiD RESULTS\n")
    buffer.write("=" * 60 + "\n")
    buffer.write(f"Control Group: {control_group.replace('_', ' ').title()}\n")
    buffer.write(f"Sample: {len(spec.data)} obs, {spec.data[spec.unit_col].nunique()} units\n\n")
    
    # Overall ATT - simplified and safer
    if overall_effect is not None:
        try:
            overall_att = float(overall_effect.atte['overall_att'])
            overall_se = float(overall_effect.atte['overall_se'])
            
            buffer.write("OVERALL TREATMENT EFFECT:\n")
            buffer.write("-" * 30 + "\n")
            buffer.write(f"ATT:        {overall_att:>8.4f}\n")
            buffer.write(f"Std Error:  {overall_se:>8.4f}\n")
            
            # Calculate confidence intervals
            ci_lower = overall_att - 1.96 * overall_se
            ci_upper = overall_att + 1.96 * overall_se
            buffer.write(f"95% CI:     [{ci_lower:>7.4f}, {ci_upper:>7.4f}]\n")
            
            # Simple significance test
            if overall_se > 0:
                t_stat = overall_att / overall_se
                significance = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
                buffer.write(f"Significant: {significance if significance else 'No'}\n")
            
            buffer.write("\n")
        except Exception as e:
            buffer.write(f"Overall effect formatting error: {str(e)}\n\n")
    
    # Simple summary table - just use the built-in summary
    buffer.write("DETAILED RESULTS:\n")
    buffer.write("-" * 30 + "\n")
    try:
        summary_table = att_gt.summary2
        buffer.write(summary_table.to_string(index=False))
        buffer.write("\n\n")
    except Exception as e:
        buffer.write(f"Summary table error: {str(e)}\n\n")
    
    buffer.write("=" * 60 + "\n")
    buffer.write("Note: *** p<0.01, ** p<0.05, * p<0.1\n")
    buffer.write("=" * 60 + "\n")
    
    return buffer.getvalue()


@make_transformable  
def callaway_santana_event_study_plot(
    spec: StaggeredDiDSpec,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive diagnostic plots for Callaway & Sant'Anna estimation.
    
    Args:
        spec: StaggeredDiDSpec with fitted Callaway & Sant'Anna model
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure with multiple diagnostic subplots
    """
    if spec.model is None or 'att_gt_object' not in spec.model:
        raise ValueError("No Callaway & Sant'Anna csdid results found in spec.model")
    
    att_gt = spec.model['att_gt_object']
    overall_effect = spec.model.get('overall_effect')
    dynamic_effects = spec.model.get('dynamic_effects')
    group_effects = spec.model.get('group_effects')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Callaway & Sant\'Anna DiD - Comprehensive Results', fontsize=16, fontweight='bold')
    
    # Get the dynamic effects object
    dynamic_effects = spec.model['dynamic_effects']
    
    # Use the built-in plotting function from att_gt instead of manual plotting
    # This ensures we get the correct event times
    plt.close('all')  # Close any existing plots
    
    # Manual plotting using the aggregated results
    if hasattr(dynamic_effects, 'atte') and dynamic_effects.atte is not None:
        atte_results = dynamic_effects.atte
        event_times = atte_results['egt']
        att_estimates = atte_results['att_egt'] 
        se_estimates = atte_results['se_egt'][0]  # se_egt is usually a list with one element
        
        # Create the plot manually
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(event_times, att_estimates, yerr=1.96*np.array(se_estimates), 
                   fmt='o', color='blue', capsize=5, capthick=2)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Event Time')
        ax.set_ylabel('ATT')
        ax.set_title('Dynamic Treatment Effects')
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    else:
        raise ValueError("Dynamic effects object does not have aggregated results")

@make_transformable
def hainmueller_synth_effect_plot(spec, save_path: Optional[str] = None, 
                          figsize: tuple = (12, 8), **kwargs) -> plt.Figure:
    """
    Create a plot for Hainmueeller synthetic effect size.
    
    Args:
        spec: A SynthDIDSpec object with fitted Hainmueeller model
        output_config: Configuration for saving the plot
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments for plot customization
        
    Returns:
        matplotlib figure object
    """

    # Call the plot method - it will handle display appropriately based on environment
    fig = spec.hainmueller_model.plot(
        ["original", "pointwise", "cumulative"], 
        treated_label="Treated Unit", 
        synth_label=f"Synthetic Control",
        figsize=figsize
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

@make_transformable
def hainmueller_synth_validity_plot(spec, save_path: Optional[str] = None, 
                          figsize: tuple = (12, 8), **kwargs) -> plt.Figure:
    """
    Create a plot for Hainmueeller synthetic control results with placebo tests.
    
    Args:
        spec: A SynthDIDSpec object with fitted Hainmueeller model
        output_config: Configuration for saving the plot
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments for plot customization
        
    Returns:
        matplotlib figure object
    """

    # Call the plot method - it will handle display appropriately based on environment
    fig = spec.hainmueller_model.plot(
        ["in-space placebo", "in-time placebo"], 
        treated_label="Treated Unit", 
        synth_label=f"Synthetic Control",
        figsize=figsize
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


@make_transformable
def plot_balance_coefficients(spec, **kwargs) -> plt.Figure:
    """
    Create horizontal error bar plot showing standardized coefficients for balance tests.
    
    Shows point estimates with horizontal error bars and vertical dashed line at 0.
    
    Args:
        spec: Specification object with balance_stats attribute containing balance results
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(spec, 'balance_stats'):
        raise ValueError("No balance statistics found. Run balance tests first.")
    
    balance_df = spec.balance_stats
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(balance_df) * 0.4)))
    
    # Extract data for plotting
    variables = balance_df['covariate']
    differences = balance_df['diff']
    std_errors = balance_df['se_diff']
    
    # Create y positions
    y_positions = range(len(variables))
    
    # Plot horizontal error bars with point estimates
    ax.errorbar(differences, y_positions, xerr=1.96*std_errors, 
                fmt='o', color='blue', capsize=5, capthick=2, 
                ecolor='blue', alpha=0.7, markersize=6)
    
    # Add vertical reference line at 0
    ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Standardized Coefficient (95% CI)', fontsize=12)
    ax.set_ylabel('Covariates', fontsize=12)
    ax.set_title('Balance Test Results', fontsize=14, fontweight='bold')
    
    # Add grid for easier reading
    ax.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig



def _plot_uplift_curve_imbalanced(y_true, uplift_scores, treatment, ax=None, **kwargs):
    """
    Uplift curve plotting optimized for highly imbalanced datasets.
    
    Uses cumulative gains and response rates instead of deciles when 
    treatment/control groups are severely imbalanced.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Create analysis dataframe
    df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_scores,
        'treatment': treatment
    })
    
    # DEBUG: Print raw input data characteristics
    print(f"DEBUG: RAW INPUT DATA:")
    print(f"  - Total observations: {len(df)}")
    print(f"  - Uplift scores range: [{uplift_scores.min():.8f}, {uplift_scores.max():.8f}]")
    print(f"  - Uplift scores std: {uplift_scores.std():.8f}")
    print(f"  - Unique uplift values: {len(np.unique(uplift_scores))}")
    print(f"  - Non-zero uplift values: {np.sum(np.abs(uplift_scores) > 1e-8)}/{len(uplift_scores)}")
    print(f"  - Treatment distribution: {np.bincount(treatment)}")
    print(f"  - Outcome distribution: {np.bincount(y_true.astype(int))}")
    
    # Sort by uplift score (descending)
    df = df.sort_values('uplift', ascending=False).reset_index(drop=True)
    
    # DEBUG: Print sorted data sample
    print(f"DEBUG: SORTED DATA SAMPLE (first 20 rows):")
    print(df.head(20)[['y', 'uplift', 'treatment']].to_string())
    print(f"DEBUG: SORTED DATA SAMPLE (last 20 rows):")
    print(df.tail(20)[['y', 'uplift', 'treatment']].to_string())
    
    # Check treatment balance
    treatment_balance = df['treatment'].mean()
    print(f"DEBUG: Treatment proportion: {treatment_balance:.3f}")
    
    # For severely imbalanced data, use cumulative approach instead of deciles
    if treatment_balance > 0.8 or treatment_balance < 0.2:
        print("Using cumulative approach for imbalanced data")
        
        # Create percentile groups
        n_groups = 10
        group_size = len(df) // n_groups
        
        # Calculate cumulative and incremental metrics
        cumulative_results = []
        for i in range(1, n_groups + 1):
            # Take top i*10% of predictions
            top_data = df.iloc[:i * group_size]
            
            # Calculate cumulative metrics
            treated = top_data[top_data['treatment'] == 1]
            control = top_data[top_data['treatment'] == 0]
            
            treated_rate = treated['y'].mean() if len(treated) > 0 else 0
            control_rate = control['y'].mean() if len(control) > 0 else 0
            uplift = treated_rate - control_rate
            
            # Calculate incremental lift for this decile only
            if i == 1:
                incremental_data = df.iloc[:group_size]
            else:
                incremental_data = df.iloc[(i-1)*group_size:i*group_size]
            
            inc_treated = incremental_data[incremental_data['treatment'] == 1]
            inc_control = incremental_data[incremental_data['treatment'] == 0]
            inc_treated_rate = inc_treated['y'].mean() if len(inc_treated) > 0 else 0
            inc_control_rate = inc_control['y'].mean() if len(inc_control) > 0 else 0
            inc_uplift = inc_treated_rate - inc_control_rate
            
            # DEBUG: Print detailed calculations
            print(f"  DECILE {i} CALCULATION DETAILS:")
            print(f"    Incremental data range: [{(i-1)*group_size}:{i*group_size}]")
            print(f"    Inc Treated: {len(inc_treated)} obs, {inc_treated['y'].sum()} conversions, rate={inc_treated_rate:.8f}")
            print(f"    Inc Control: {len(inc_control)} obs, {inc_control['y'].sum()} conversions, rate={inc_control_rate:.8f}")
            print(f"    Inc Uplift: {inc_treated_rate:.8f} - {inc_control_rate:.8f} = {inc_uplift:.8f}")
            print(f"    Uplift Score Range: [{incremental_data['uplift'].min():.8f}, {incremental_data['uplift'].max():.8f}]")
            
            # Additional debugging for zero uplifts
            if abs(inc_uplift) < 1e-8:
                print(f"    WARNING: Near-zero uplift detected!")
                print(f"    Treated outcomes: {inc_treated['y'].tolist()[:10]}...")  # First 10 values
                print(f"    Control outcomes: {inc_control['y'].tolist()[:10]}...")  # First 10 values
            
            cumulative_results.append({
                'percentile': i * 10,
                'cumulative_uplift': uplift,
                'incremental_uplift': inc_uplift,
                'n_treated': len(treated),
                'n_control': len(control),
                'inc_n_treated': len(inc_treated),
                'inc_n_control': len(inc_control),
                'treated_rate': treated_rate,
                'control_rate': control_rate,
                'inc_treated_rate': inc_treated_rate,
                'inc_control_rate': inc_control_rate,
                'uplift_score_range': f"[{incremental_data['uplift'].min():.8f}, {incremental_data['uplift'].max():.8f}]"
            })
        
        results_df = pd.DataFrame(cumulative_results)
        
        # Print detailed diagnostic information
        print("DEBUG: Cumulative Analysis Results:")
        for _, row in results_df.iterrows():
            print(f"  Top {int(row['percentile'])}%: cum_uplift={row['cumulative_uplift']:.6f}, "
                  f"inc_uplift={row['incremental_uplift']:.6f}, "
                  f"T:{row['inc_n_treated']}/C:{row['inc_n_control']}, "
                  f"scores={row['uplift_score_range']}")
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Cumulative uplift
        x = results_df['percentile']
        y1 = results_df['cumulative_uplift']
        
        ax1.plot(x, y1, 'o-', linewidth=2, markersize=6, color='blue', label='Cumulative Uplift')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Top % of Predictions')
        ax1.set_ylabel('Cumulative Uplift')
        ax1.set_title('Cumulative Uplift Curve (Top % Targeting)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Adjust y-axis to show small values clearly
        y_max = max(0.001, max(abs(y1.min()), abs(y1.max())))
        ax1.set_ylim(-y_max * 1.2, y_max * 1.2)
        
        # Add value labels
        for i, (xi, yi) in enumerate(zip(x, y1)):
            ax1.annotate(f'{yi:.6f}', (xi, yi), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # Plot 2: Incremental uplift by decile
        y2 = results_df['incremental_uplift']
        colors = ['darkred' if val < -0.0001 else 'darkgreen' if val > 0.0001 else 'gray' for val in y2]
        
        # Adjust y-axis for small values (calculate first)
        y2_max = max(0.001, max(abs(y2.min()), abs(y2.max())))
        
        # DEBUG: Print exact values being plotted
        print(f"\nDEBUG: BAR CHART VALUES BEING PLOTTED:")
        for i, (percentile, value) in enumerate(zip(results_df['percentile'], y2)):
            print(f"  Bar {i+1} (Top {int(percentile)}%): {value:.8f} (color: {colors[i]})")
        print(f"  Y-axis range will be: [{-y2_max * 1.2:.8f}, {y2_max * 1.2:.8f}]")
        print(f"  Max absolute value: {y2_max:.8f}")
        
        bars = ax2.bar(x, y2, alpha=0.7, color=colors, edgecolor='navy', linewidth=1)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Decile (Top 10%, 20%, etc.)')
        ax2.set_ylabel('Incremental Uplift')
        ax2.set_title('Incremental Uplift by Decile')
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax2.set_ylim(-y2_max * 1.2, y2_max * 1.2)
        
        # DEBUG: Print bar heights after plotting
        print(f"\nDEBUG: ACTUAL BAR HEIGHTS:")
        for i, bar in enumerate(bars):
            height = bar.get_height()
            print(f"  Bar {i+1}: height={height:.8f}, visible={abs(height) > 1e-10}")
        
        # Add value labels for incremental with higher precision
        for i, (bar, yi) in enumerate(zip(bars, y2)):
            height = bar.get_height()
            
            # Get conversion info for this decile
            row = results_df.iloc[i]
            treated_conversions = int(row['inc_n_treated'] * row['inc_treated_rate'])
            control_conversions = int(row['inc_n_control'] * row['inc_control_rate'])
            has_conversions = treated_conversions > 0 or control_conversions > 0
            
            if abs(yi) > 1e-8:  # Non-zero uplift
                ax2.text(bar.get_x() + bar.get_width()/2., height + y2_max * 0.05,
                        f'{yi:.6f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:  # Zero uplift
                if not has_conversions:
                    # No conversions in this decile
                    ax2.text(bar.get_x() + bar.get_width()/2., y2_max * 0.3,
                            'No\nConversions', ha='center', va='center', fontsize=7, 
                            alpha=0.7, style='italic', color='darkred')
                else:
                    # Has conversions but zero uplift
                    ax2.text(bar.get_x() + bar.get_width()/2., y2_max * 0.05,
                            f'{yi:.6f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add sample size info with conversion counts
        for i, row in results_df.iterrows():
            treated_conversions = int(row['inc_n_treated'] * row['inc_treated_rate'])
            control_conversions = int(row['inc_n_control'] * row['inc_control_rate'])
            
            ax2.text(row['percentile'], -y2_max * 1.0,
                    f'T:{row["inc_n_treated"]}/C:{row["inc_n_control"]}',
                    ha='center', va='top', fontsize=7, alpha=0.7)
            
            # Add conversion counts below
            ax2.text(row['percentile'], -y2_max * 1.15,
                    f'Conv: {treated_conversions}+{control_conversions}',
                    ha='center', va='top', fontsize=6, alpha=0.6, style='italic')
        
        # Add data sparsity warning if applicable
        total_conversions = sum(int(row['inc_n_treated'] * row['inc_treated_rate']) + 
                              int(row['inc_n_control'] * row['inc_control_rate']) 
                              for _, row in results_df.iterrows())
        zero_deciles = sum(1 for _, row in results_df.iterrows() 
                          if abs(row['incremental_uplift']) < 1e-8)
        
        if zero_deciles >= 7:  # Most deciles have zero uplift
            warning_text = f'⚠️ Extremely Sparse Data\n{total_conversions} total conversions\n{zero_deciles}/10 deciles have no conversions'
            ax2.text(0.02, 0.02, warning_text, transform=ax2.transAxes, fontsize=8,
                    va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    else:
        # For balanced data, use the standard decile approach
        return _plot_uplift_curve_native(y_true, uplift_scores, treatment, ax, **kwargs)


@make_transformable
def uplift_curve_plot_adaptive(spec: UpliftSpec, **kwargs) -> plt.Figure:
    """
    Create adaptive uplift curve plot that handles both balanced and imbalanced datasets.
    """
    if spec.cate_estimates is None or spec.ate_estimate is None:
        raise ValueError("No uplift estimates found. Ensure model is fitted.")

    y_true = spec.data[spec.outcome_col].values
    uplift_scores = list(spec.cate_estimates.values())[0]
    treatment = spec.data[spec.treatment_cols[0]].values

    # Diagnostic information
    print(f"DIAGNOSTIC: {spec.model_type} Uplift Analysis")
    print(f"  - CATE range: [{np.min(uplift_scores):.6f}, {np.max(uplift_scores):.6f}]")
    print(f"  - CATE std: {np.std(uplift_scores):.6f}")
    print(f"  - ATE: {spec.ate_estimate:.6f}")
    print(f"  - Non-zero CATE values: {np.sum(np.abs(uplift_scores) > 1e-8)}/{len(uplift_scores)}")
    
    # Check for severe imbalance
    treatment_prop = np.mean(treatment)
    
    # Check for very low heterogeneity
    heterogeneity_level = np.std(uplift_scores)
    
    if treatment_prop > 0.8 or treatment_prop < 0.2:
        print(f"  - Using imbalanced approach (treatment prop: {treatment_prop:.3f})")
        fig, axes = _plot_uplift_curve_imbalanced(y_true, uplift_scores, treatment, **kwargs)
        if isinstance(axes, tuple):
            axes[0].set_title(f"Adaptive Uplift Analysis - {spec.model_type.title()}", fontsize=14, fontweight='bold')
            # Add diagnostic text to the plot
            axes[1].text(0.02, 0.98, f"Heterogeneity: {heterogeneity_level:.6f}", 
                        transform=axes[1].transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            axes.set_title(f"Adaptive Uplift Analysis - {spec.model_type.title()}", fontsize=14, fontweight='bold')
    else:
        print(f"  - Using standard approach (treatment prop: {treatment_prop:.3f})")
        fig, ax = _plot_uplift_curve_native(y_true, uplift_scores, treatment, **kwargs)
        ax.set_title(f"Uplift Curve - {spec.model_type.title()}", fontsize=14, fontweight='bold')
        # Add diagnostic text to the plot
        ax.text(0.02, 0.02, f"Heterogeneity: {heterogeneity_level:.6f}", 
               transform=ax.transAxes, fontsize=9, va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    return fig


@make_transformable
def plot_balance_coefficients(spec, **kwargs) -> plt.Figure:
    """
    Create horizontal error bar plot showing standardized coefficients for balance tests.
    
    Shows point estimates with horizontal error bars and vertical dashed line at 0.
    
    Args:
        spec: Specification object with balance_stats attribute containing balance results
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(spec, 'balance_stats'):
        raise ValueError("No balance statistics found. Run balance tests first.")
    
    balance_df = spec.balance_stats
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(balance_df) * 0.4)))
    
    # Extract data for plotting
    variables = balance_df['covariate']
    differences = balance_df['diff']
    std_errors = balance_df['se_diff']
    
    # Create y positions
    y_positions = range(len(variables))
    
    # Plot horizontal error bars with point estimates
    ax.errorbar(differences, y_positions, xerr=1.96*std_errors, 
                fmt='o', color='blue', capsize=5, capthick=2, 
                ecolor='blue', alpha=0.7, markersize=6)
    
    # Add vertical reference line at 0
    ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Standardized Coefficient (95% CI)', fontsize=12)
    ax.set_ylabel('Covariates', fontsize=12)
    ax.set_title('Balance Test Results', fontsize=14, fontweight='bold')
    
    # Add grid for easier reading
    ax.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig