from typing import Optional
import io
import statsmodels.api as sm
from statsmodels.base.model import Results
from statsmodels.base.wrapper import ResultsWrapper
from sklearn.base import BaseEstimator
from pyautocausal.persistence.parameter_mapper import make_transformable
from pyautocausal.persistence.output_config import OutputConfig, OutputType
from typing import Any
import pandas as pd
from pyautocausal.pipelines.library.specifications import BaseSpec
from linearmodels.shared.base import _SummaryStr


@make_transformable
def write_linear_models_to_summary(res: BaseSpec) -> str:
    """Write linear models summary to text."""
    # Handle the case where res is a BaseSpec object with a model attribute
    if hasattr(res, 'model'):
        result = res.model
    else:
        raise ValueError("res must be a BaseSpec object with a model attribute")
    if not isinstance(result, _SummaryStr):
        raise ValueError("res must be a BaseSpec object with a model attribute that is a (linearmodels) _SummaryStr object")
    
    # Create summary
    try:
        buffer = io.StringIO()
        buffer.write(str(result))
        return buffer.getvalue()
    except Exception as e:
        return f"Error creating summary: {str(e)}"
    

@make_transformable
def write_statsmodels_to_summary(res: BaseSpec) -> str:
    """Write statsmodels summary to text."""
    # Handle the case where res is a BaseSpec object with a model attribute
    if hasattr(res, 'model'):
        result = res.model
    else:
        raise ValueError("res must be a BaseSpec object with a model attribute")
    if not isinstance(result, Results) and not isinstance(result, ResultsWrapper):
        raise ValueError("res must be a BaseSpec object with a model attribute that is a (statsmodels) Results or ResultsWrapper object")
    
    # Create summary
    try:
        buffer = io.StringIO()
        buffer.write(str(result.summary()))
        return buffer.getvalue()
    except Exception as e:
        return f"Error creating summary: {str(e)}"


@make_transformable
def write_sklearn_summary(res: BaseEstimator) -> str:
    """Write sklearn model summary to text."""
    output = []
    output.append(f"Model Type: {type(res).__name__}")
    output.append("\nModel Parameters:")
    output.append(str(res.get_params()))
    
    # Print for notebook display
    print("\n".join(output))
    
    return "\n".join(output)


@make_transformable
def write_statsmodels_summary_notebook(output: Any) -> None:
    """Display statsmodels summary in a notebook."""
    print(output)


def display_cs_results_notebook(output: Any) -> None:
    """Display Callaway & Sant'Anna results in a notebook."""
    print(output)


@make_transformable
def write_hainmueller_summary(spec, output_config: Optional[OutputConfig] = None) -> str:
    """
    Write a text summary of Hainmueeller Synthetic Control results.
    
    Args:
        spec: A specification object with fitted Hainmueeller model
        output_config: Optional output configuration for saving
        
    Returns:
        String containing the formatted summary
    """
    
    
    model = spec.hainmueller_model
    
    # Check if we have a SyntheticControlMethods Synth object
    if not hasattr(model, 'original_data'):
        raise ValueError("Hainmueeller model must be a SyntheticControlMethods Synth object")
    
    # Use SyntheticControlMethods results
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("HAINMUEELLER SYNTHETIC CONTROL RESULTS")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Display weight DataFrame
    summary_lines.append("DONOR UNIT WEIGHTS:")
    summary_lines.append("-" * 20)
    weight_df_str = str(model.original_data.weight_df)
    summary_lines.append(weight_df_str)
    summary_lines.append("")
    
    # Display comparison DataFrame
    summary_lines.append("COMPARISON RESULTS:")
    summary_lines.append("-" * 20)
    comparison_df_str = str(model.original_data.comparison_df)
    summary_lines.append(comparison_df_str)
    summary_lines.append("")
    
    # Display penalty parameter if available
    summary_lines.append(f"Penalty parameter: {model.original_data.pen}")
    summary_lines.append("")
    
    # Display RMSPE results if placebo test was run
    summary_lines.append("PLACEBO TEST RESULTS (RMSPE):")
    summary_lines.append("-" * 30)
    rmspe_df_str = str(model.original_data.rmspe_df)
    summary_lines.append(rmspe_df_str)
    summary_lines.append("")

    summary_lines.append("=" * 60)
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Print for notebook display
    print(summary_text)
    
    # Save to file if output config provided
    if output_config:
        # Implementation for saving would go here
        pass
    
    return summary_text


@make_transformable 
def write_balance_summary_table(spec) -> str:
    """
    Write a text summary of balance test results.
    
    Args:
        spec: A specification object with balance_stats attribute
        
    Returns:
        String containing the formatted balance test summary
    """
    if not hasattr(spec, 'balance_stats'):
        raise ValueError("No balance statistics found. Run balance tests first.")
    
    balance_df = spec.balance_stats
    
    # Create formatted summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("BALANCE TEST RESULTS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Column headers
    summary_lines.append(f"{'Variable':<15} {'Treated':<10} {'Control':<10} {'Difference':<12} {'Std. Err.':<10} {'t-stat':<8} {'p-value':<8}")
    summary_lines.append("-" * 80)
    
    # Add rows for each covariate
    for _, row in balance_df.iterrows():
        summary_lines.append(
            f"{row['covariate']:<15} "
            f"{row['treated_mean']:<10.3f} "
            f"{row['control_mean']:<10.3f} "
            f"{row['diff']:<12.3f} "
            f"{row['se_diff']:<10.3f} "
            f"{row['t_stat']:<8.3f} "
            f"{row['p_value']:<8.3f}"
        )
    
    summary_lines.append("")
    summary_lines.append("Notes:")
    summary_lines.append("- Difference = Treated Mean - Control Mean")
    summary_lines.append("- Tests use Welch's t-test (unequal variances)")
    summary_lines.append("=" * 80)
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Print for notebook display
    print(summary_text)
    
    return summary_text


@make_transformable
def write_uplift_summary(spec) -> str:
    """
    Write a text summary of uplift modeling results.
    
    Args:
        spec: An UpliftSpec object with fitted model and evaluation metrics
        
    Returns:
        String containing the formatted uplift summary
    """
    
    # Create formatted summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("UPLIFT MODELING RESULTS")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Method information
    if hasattr(spec, 'method_name'):
        summary_lines.append(f"Method: {spec.method_name}")
    summary_lines.append(f"Model Type: {type(spec.model).__name__ if hasattr(spec, 'model') else 'Unknown'}")
    summary_lines.append("")
    
    # ATE estimates
    if hasattr(spec, 'ate_estimate'):
        summary_lines.append(f"Average Treatment Effect (ATE): {spec.ate_estimate:.4f}")
    
    if hasattr(spec, 'ate_ci'):
        summary_lines.append(f"95% Confidence Interval: [{spec.ate_ci[0]:.4f}, {spec.ate_ci[1]:.4f}]")
    
    summary_lines.append("")
    
    # Evaluation metrics
    if hasattr(spec, 'evaluation_metrics') and spec.evaluation_metrics:
        summary_lines.append("EVALUATION METRICS:")
        summary_lines.append("-" * 20)
        for metric, value in spec.evaluation_metrics.items():
            summary_lines.append(f"{metric}: {value:.4f}")
        summary_lines.append("")
    
    # CATE statistics 
    if hasattr(spec, 'cate_estimates'):
        summary_lines.append("INDIVIDUAL TREATMENT EFFECTS (CATE):")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Mean CATE: {spec.cate_estimates.mean():.4f}")
        summary_lines.append(f"Std Dev CATE: {spec.cate_estimates.std():.4f}")
        summary_lines.append(f"Min CATE: {spec.cate_estimates.min():.4f}")
        summary_lines.append(f"Max CATE: {spec.cate_estimates.max():.4f}")
        summary_lines.append("")
    
    summary_lines.append("=" * 60)
    
    # Join all lines
    summary_text = "\n".join(summary_lines)
    
    # Print for notebook display
    print(summary_text)
    
    return summary_text