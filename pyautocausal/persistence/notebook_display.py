"""Small, explicit rich displays for replayed notebook results."""

from contextlib import contextmanager, redirect_stdout
from html import escape
from io import BytesIO, StringIO
from numbers import Number

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from IPython.display import HTML, Image, display

from pyautocausal.pipelines.library.specifications import BaseSpec
from pyautocausal.pipelines.library.synthdid.synthdid import SynthDIDEstimate


def _notice(text):
    display(HTML(f'<p class="pac-notice">{escape(str(text))}</p>'))


def _table(frame):
    table = frame.to_html(
        classes="pac-table", border=0, escape=True, max_rows=12, max_cols=12,
        float_format=lambda value: f"{value:,.4g}", na_rep="Unavailable",
    )
    # Notebook viewers also need natural column widths without the HTML report stylesheet.
    table = table.replace('<table ', '<table style="table-layout:auto; width:max-content; min-width:100%" ', 1)
    note = f"{len(frame):,} rows × {len(frame.columns):,} columns"
    if len(frame) > 12 or len(frame.columns) > 12:
        note += "; preview shown. The complete result remains in the output variable."
    display(HTML(f'<div class="pac-table-wrap" style="overflow-x:auto">{table}</div>'
                 f'<p class="pac-notice">{note}</p>'))


def _coefficient_table(model):
    from linearmodels.shared.base import _SummaryStr
    from statsmodels.base.model import Results
    from statsmodels.base.wrapper import ResultsWrapper

    if not isinstance(model, (Results, ResultsWrapper, _SummaryStr)):
        return False
    params = model.params
    if not isinstance(params, pd.Series):
        params = pd.Series(params, index=model.model.exog_names)
    coefficients = params.to_frame("Estimate")
    errors = getattr(model, "std_errors", None)
    if errors is None:
        errors = model.bse
    coefficients["Std. error"] = errors
    coefficients["p-value"] = model.pvalues
    try:
        intervals = model.conf_int()
        coefficients[["95% CI lower", "95% CI upper"]] = (
            intervals.to_numpy() if isinstance(intervals, pd.DataFrame) else intervals
        )
    except (ValueError, NotImplementedError):
        coefficients[["95% CI lower", "95% CI upper"]] = float("nan")
    _table(coefficients)
    _notice(f"Observations: {int(model.nobs):,}. Confidence intervals use the fitted model's inference.")
    if coefficients.isna().any().any():
        _notice("Some uncertainty estimates are unavailable; no replacement intervals have been calculated.")
    return True


def _display_spec(spec):
    model = getattr(spec, "model", None)
    if _coefficient_table(model):
        return
    balance = getattr(spec, "balance_stats", None)
    if model is None and isinstance(balance, pd.DataFrame):
        if balance.empty:
            reason = getattr(spec, "balance_diagnostics", {}).get(
                "reason", "No control covariates are available for balance assessment.")
            _notice("Covariate balance unavailable. " + reason)
        else:
            _table(balance)
    if isinstance(model, SynthDIDEstimate):
        inference = model.inference or {}
        ci = inference.get("ci") if inference.get("status") != "unavailable" else None
        _table(pd.DataFrame({
            "Estimate": [model.estimate],
            "Std. error": [inference.get("se") if inference.get("status") != "unavailable" else None],
            "CI lower": [ci[0] if ci is not None else None],
            "CI upper": [ci[1] if ci is not None else None],
        }, index=["Average treatment effect"], dtype=float))
        if ci is None:
            _notice("Confidence interval unavailable. " + inference.get("reason", "Inference was not computed."))
        elif inference.get("method"):
            _notice("Inference method: " + str(inference["method"]))
    elif isinstance(model, dict) and "att_gt_object" in model:
        att_gt = model["att_gt_object"]
        summary = getattr(att_gt, "summary2", None)
        if isinstance(summary, pd.DataFrame):
            _table(summary)
        inference = getattr(att_gt, "inference", {}) or {}
        if inference.get("status") in {"partial", "unavailable"}:
            _notice("Group-time inference " + inference["status"] + ": " + inference.get(
                "reason", "Some confidence intervals are unavailable."))
    synthetic = getattr(spec, "hainmueller_model", None)
    if synthetic is not None:
        data = synthetic.original_data
        for attr, title in [("weight_df", "Donor weights"),
                            ("comparison_df", "Predictor balance"),
                            ("rmspe_df", "Placebo diagnostics (RMSPE)")]:
            frame = getattr(data, attr, None)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                _notice(title)
                _table(frame)


@contextmanager
def capture_step_output():
    """Fold routine stdout while leaving warnings, stderr, and failures visible."""
    output = StringIO()
    try:
        with redirect_stdout(output):
            yield output
    finally:
        if output.getvalue():
            display(HTML('<details><summary>Execution log</summary><pre>'
                         + escape(output.getvalue()) + '</pre></details>'))


def display_result(value, printed=""):
    """Display known result types without dumping internal objects or source data."""
    if isinstance(value, Figure):
        if (len(value.axes) == 1 and not value.axes[0].has_data()
                and value.axes[0].get_title() == "Covariate balance unavailable"):
            _notice("Covariate balance unavailable. " + " ".join(
                text.get_text() for text in value.axes[0].texts))
            plt.close(value)
            return
        buffer = BytesIO()
        value.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        display(Image(data=buffer.getvalue(), format="png"))
        # Explicit image output also works with Agg; close only this figure to avoid duplicates.
        plt.close(value)
    elif isinstance(value, pd.DataFrame):
        _table(value)
    elif isinstance(value, pd.Series):
        _table(value.to_frame())
    elif isinstance(value, BaseSpec):
        _display_spec(value)
    elif isinstance(value, str):
        if value.strip() != printed.strip():
            print(value)
    elif isinstance(value, Number):
        print(value)
