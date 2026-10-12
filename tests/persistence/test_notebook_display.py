from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import HTML, Image

from pyautocausal.persistence import notebook_display
from pyautocausal.pipelines.library.specifications import BaseSpec
from pyautocausal.pipelines.library.synthdid.synthdid import SynthDIDEstimate


def _capture(monkeypatch):
    outputs = []
    monkeypatch.setattr(notebook_display, "display", outputs.append)
    return outputs


def _html(outputs):
    return "\n".join(output.data for output in outputs if isinstance(output, HTML))


def test_figure_embeds_png_with_non_inline_backend_and_closes_only_it(monkeypatch):
    outputs = _capture(monkeypatch)
    with plt.ioff():
        figure, axis = plt.subplots()
        other = plt.figure()
        axis.plot([1, 2], [3, 4])
        notebook_display.display_result(figure)
        assert isinstance(outputs[0], Image)
        assert outputs[0].data.startswith(b"\x89PNG")
        assert "image/png" in outputs[0]._repr_mimebundle_()[0]
        assert not plt.fignum_exists(figure.number)
        assert plt.fignum_exists(other.number)
        plt.close(other)


def test_table_is_bounded_escaped_and_does_not_mutate_data(monkeypatch):
    outputs = _capture(monkeypatch)
    frame = pd.DataFrame({"<script>bad()</script>": ["<img src=x onerror=bad()>"] * 30,
                          "Estimate": np.arange(30, dtype=float)},
                         index=pd.Index(range(30), name="<b>name</b>"))
    original = frame.copy(deep=True)
    notebook_display.display_result(frame)
    html = _html(outputs)
    assert "pac-table" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html and "&lt;img" in html
    assert "&lt;b&gt;name&lt;/b&gt;" in html
    assert "30 rows" in html and "preview shown" in html
    assert html.count("&lt;img") <= 12
    pd.testing.assert_frame_equal(frame, original)


def test_coefficient_table_uses_native_intervals(monkeypatch):
    outputs = _capture(monkeypatch)
    model = sm.OLS(pd.Series([1.1, 1.8, 3.2, 3.9, 5.1]),
                   sm.add_constant(pd.DataFrame({"<x>": np.arange(5)}))).fit()
    spec = BaseSpec(pd.DataFrame(), "y ~ x")
    spec.model = model
    spec.balance_stats = pd.DataFrame({"diff": [123.0]})
    seen = []
    monkeypatch.setattr(notebook_display, "_table", seen.append)
    notebook_display.display_result(spec)
    assert len(seen) == 1
    pd.testing.assert_series_equal(seen[0]["Estimate"], model.params, check_names=False)
    np.testing.assert_allclose(seen[0][["95% CI lower", "95% CI upper"]], model.conf_int())
    assert "Observations: 5" in _html(outputs)


def test_synthetic_zero_estimate_preserved_and_unavailable_inference_visible(monkeypatch):
    outputs = _capture(monkeypatch)
    spec = BaseSpec(pd.DataFrame(), "")
    spec.model = SynthDIDEstimate(0.0, {}, {}, {}, inference={
        "status": "unavailable", "reason": "Too few <donors>", "se": None, "ci": None,
    })
    notebook_display.display_result(spec)
    html = _html(outputs)
    assert "<td>0</td>" in html
    assert "Unavailable" in html and "Confidence interval unavailable" in html
    assert "Too few &lt;donors&gt;" in html
    assert spec.model.estimate == 0


def test_cs_summary_preserves_native_interval_and_inference_notice(monkeypatch):
    outputs = _capture(monkeypatch)
    summary = pd.DataFrame({"ATT": [0.0], "Lower": [np.nan], "Upper": [np.nan]})
    spec = BaseSpec(pd.DataFrame(), "")
    spec.model = {"att_gt_object": SimpleNamespace(summary2=summary, inference={
        "status": "partial", "reason": "Some groups have insufficient observations.",
    })}
    notebook_display.display_result(spec)
    html = _html(outputs)
    assert "<td>0</td>" in html and "Unavailable" in html
    assert "Group-time inference partial" in html
    assert summary.isna().sum().sum() == 2


def test_unfitted_specs_and_arbitrary_objects_are_silent(monkeypatch, capsys):
    outputs = _capture(monkeypatch)
    notebook_display.display_result(BaseSpec(pd.DataFrame({"secret": [42]}), ""))
    notebook_display.display_result(SimpleNamespace(secret=42))
    notebook_display.display_result({"secret": 42})
    notebook_display.display_result(None)
    assert outputs == []
    assert capsys.readouterr().out == ""


def test_strings_preserve_plain_text_and_scalars_preserve_zero(capsys):
    notebook_display.display_result("<script>alert('text')</script>\n=== ASCII summary ===")
    notebook_display.display_result(0)
    assert capsys.readouterr().out == "<script>alert('text')</script>\n=== ASCII summary ===\n0\n"


def test_linearmodels_coefficients_use_native_standard_errors_and_intervals(monkeypatch):
    from linearmodels.panel import PanelOLS

    _capture(monkeypatch)
    index = pd.MultiIndex.from_product([range(4), range(3)], names=["unit", "time"])
    frame = pd.DataFrame({"x": np.arange(12) % 5,
                          "y": np.arange(12) + np.sin(np.arange(12))}, index=index)
    model = PanelOLS.from_formula("y ~ 1 + x + EntityEffects", frame).fit()
    spec = BaseSpec(frame, "y ~ x")
    spec.model = model
    tables = []
    monkeypatch.setattr(notebook_display, "_table", tables.append)
    notebook_display.display_result(spec)
    np.testing.assert_allclose(tables[0]["Std. error"], model.std_errors)
    np.testing.assert_allclose(tables[0][["95% CI lower", "95% CI upper"]], model.conf_int())


def test_empty_balance_figure_becomes_notice_but_data_figures_stay_images(monkeypatch):
    from pyautocausal.pipelines.library.plots import plot_balance_coefficients

    outputs = _capture(monkeypatch)
    spec = SimpleNamespace(balance_stats=pd.DataFrame(), balance_diagnostics={
        "reason": "No <covariates> were specified.",
    })
    figure = plot_balance_coefficients.get_function()(spec)
    notebook_display.display_result(figure)
    assert not any(isinstance(output, Image) for output in outputs)
    assert "Covariate balance unavailable" in _html(outputs)
    assert "No &lt;covariates&gt; were specified." in _html(outputs)
    assert not plt.fignum_exists(figure.number)
    figure, axis = plt.subplots()
    axis.set_title("Covariate balance unavailable")
    axis.plot([1, 2], [3, 4])
    notebook_display.display_result(figure)
    assert isinstance(outputs[-1], Image)


def test_printed_summary_is_folded_once_and_escaped(monkeypatch, capsys):
    from pyautocausal.pipelines.library.output import write_balance_summary_table

    outputs = _capture(monkeypatch)
    spec = SimpleNamespace(balance_stats=pd.DataFrame(), balance_diagnostics={
        "reason": "No <script>covariates</script> were specified.",
    })
    with notebook_display.capture_step_output() as captured:
        result = write_balance_summary_table.get_function()(spec)
    notebook_display.display_result(result, printed=captured.getvalue())
    html = _html(outputs)
    assert "<details><summary>Execution log</summary>" in html
    assert html.count("BALANCE TEST RESULTS") == 1
    assert "&lt;script&gt;covariates&lt;/script&gt;" in html
    assert "<script>" not in html
    assert capsys.readouterr().out == ""


def test_capture_preserves_warnings_stderr_and_exceptions(monkeypatch, capsys):
    import sys
    import warnings
    import pytest

    outputs = _capture(monkeypatch)
    with pytest.warns(UserWarning, match="inference warning"):
        with pytest.raises(RuntimeError, match="step failed"):
            with notebook_display.capture_step_output() as captured:
                print("before failure")
                print("stderr remains visible", file=sys.stderr)
                warnings.warn("inference warning")
                raise RuntimeError("step failed")
    assert captured.getvalue() == "before failure\n"
    assert "before failure" in _html(outputs)
    assert "stderr remains visible" in capsys.readouterr().err


def test_capture_without_stdout_is_silent_and_new_strings_are_displayed(monkeypatch, capsys):
    outputs = _capture(monkeypatch)
    with notebook_display.capture_step_output():
        pass
    assert outputs == []
    notebook_display.display_result("a useful result", printed="routine progress\n")
    assert capsys.readouterr().out == "a useful result\n"


def test_unfitted_balance_spec_displays_numeric_table_and_missing_notice(monkeypatch):
    outputs = _capture(monkeypatch)
    spec = BaseSpec(pd.DataFrame(), "")
    spec.balance_stats = pd.DataFrame({"covariate": ["<income>"], "diff": [1.234567]})
    original = spec.balance_stats.copy(deep=True)
    notebook_display.display_result(spec)
    assert "pac-table" in _html(outputs)
    assert "&lt;income&gt;" in _html(outputs)
    assert "1.235" in _html(outputs)
    pd.testing.assert_frame_equal(spec.balance_stats, original)
    spec.balance_stats = pd.DataFrame()
    spec.balance_diagnostics = {"reason": "No <covariates> selected."}
    notebook_display.display_result(spec)
    assert "Covariate balance unavailable. No &lt;covariates&gt; selected." in _html(outputs)
