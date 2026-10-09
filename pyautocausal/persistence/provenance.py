"""Compact, machine-readable run diagnostics without serializing raw datasets."""
from importlib.metadata import PackageNotFoundError, version
import platform
import math
import numpy as np
import pandas as pd


def _safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(item) for item in value]
    return str(value)


def graph_execution_report(graph):
    """Capture actual states, method settings, sample sizes and fitted estimates.

    Non-finite diagnostics are represented by null; this report never replaces
    the strict JSON policy used for saved result values. Raw data and fitted
    model objects are deliberately excluded.
    """
    packages = {}
    for package in ('pyautocausal', 'numpy', 'pandas', 'statsmodels', 'scikit-learn', 'linearmodels'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'not installed'
    report = {
        'schema_version': 1,
        'python_version': platform.python_version(),
        'package_versions': packages,
        'run_context': _safe(graph.run_context.metadata) if graph.run_context else {},
        'diagnostic_nonfinite_policy': 'non-finite diagnostics are null',
        'nodes': {},
    }
    fields = ('formula', 'outcome_col', 'treatment_cols', 'control_cols', 'time_col',
              'unit_col', 'N0', 'T0', 'reference_period', 'model_type', 'ate_estimate',
              'ate_ci', 'experimental', 'inference', 'treatment_mapping', 'random_state', 'seed',
              'balance_diagnostics', 'hainmueller_placebo_metadata')
    for node in graph.nodes:
        details = {'state': node.state.value, 'execution_count': node.execution_count}
        if getattr(node, 'error', None):
            details['error'] = _safe(node.error)
        if getattr(node, 'decision_result', None) is not None:
            details['decision_result'] = node.decision_result
        if node.is_passed():
            details['reason'] = 'not selected by decision branching'
        if node.is_completed():
            value = node.get_result_value()
            analysis = {key: _safe(getattr(value, key)) for key in fields if hasattr(value, key)}
            data = value if isinstance(value, pd.DataFrame) else getattr(value, 'data', None)
            if isinstance(data, pd.DataFrame):
                analysis['rows'] = len(data)
                analysis['columns'] = list(data.columns)
                analysis['transformations'] = _safe(data.attrs)
                unit_col = getattr(value, 'unit_col', 'id_unit')
                if unit_col in data:
                    analysis['units'] = int(data[unit_col].nunique())
            model = getattr(value, 'model', value)
            for field in ('estimate', 'inference', 'estimator'):
                if getattr(model, field, None) is not None:
                    analysis[field] = _safe(getattr(model, field))
            if isinstance(model, dict):
                for key in ('control_group', 'estimator', 'column_mapping',
                            'random_state', 'bootstrap_iterations'):
                    if key in model:
                        analysis[key] = _safe(model[key])
                overall = getattr(model.get('overall_effect'), 'atte', None)
                if isinstance(overall, dict):
                    analysis['overall_effect'] = {
                        key: _safe(overall[key]) for key in ('overall_att', 'overall_se') if key in overall
                    }
                seeds = {name: _safe(getattr(fitted, 'random_state'))
                         for name, fitted in model.items() if hasattr(fitted, 'random_state')}
                if seeds:
                    analysis['model_random_states'] = seeds
            params = getattr(model, 'params', None)
            if isinstance(params, pd.Series):
                analysis['coefficients'] = _safe(params.to_dict())
            if hasattr(model, 'nobs'):
                analysis['estimation_rows'] = _safe(model.nobs)
            if analysis:
                details['analysis'] = analysis
            if node.output_config:
                details['output'] = {'filename': node.output_config.output_filename or node.name,
                                     'type': node.output_config.output_type.value}
        report['nodes'][node.name] = details
    return report
