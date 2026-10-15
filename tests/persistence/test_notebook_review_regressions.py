from html import escape

import nbformat
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pyautocausal import ExecutableGraph, export_outputs
from pyautocausal.pipelines.library.estimators import fit_s_learner
from pyautocausal.pipelines.library.output import write_uplift_summary
from pyautocausal.pipelines.library.specifications import create_uplift_specification


def _fit_multinomial(data):
    import statsmodels.api as sm
    from pyautocausal.pipelines.library.specifications import BaseSpec

    spec = BaseSpec(data, "category ~ x")
    spec.model = sm.MNLogit(data.category, sm.add_constant(data[["x"]])).fit(disp=False)
    return spec


def _visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("script, style, details:not([open])"):
        element.decompose()
    return soup.get_text(" ", strip=True)


def test_export_keeps_multinomial_and_uplift_summaries_visible(tmp_path):
    rng = np.random.default_rng(1729)
    frame = pd.DataFrame({
        "id_unit": range(80), "treat": np.tile([0, 1], 40),
        "y": rng.binomial(1, .5, 80), "x": rng.normal(size=80),
        "category": rng.integers(0, 3, 80),
    })
    graph = ExecutableGraph().create_input_node("data", pd.DataFrame)
    graph.create_node("multinomial", _fit_multinomial, ["data"])
    graph.create_node("uplift_spec", create_uplift_specification.get_function(), ["data"])
    graph.create_node("uplift_fit", fit_s_learner.transform({"uplift_spec": "spec"}), ["uplift_spec"])
    graph.create_node("uplift_summary", write_uplift_summary.transform({"uplift_fit": "spec"}), ["uplift_fit"])
    graph.fit(data=frame)
    estimate = graph.get("uplift_fit").get_result_value().ate_estimate

    export_outputs(graph, df=frame, output_path=tmp_path)

    notebook = nbformat.read(tmp_path / "pipeline_execution.ipynb", as_version=4)
    outputs = [output for cell in notebook.cells if cell.cell_type == "code"
               for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    notebook_html = "\n".join(
        output.get("data", {}).get("text/html", "")
        + escape(output.get("text", "")) for output in outputs
    )
    for rendered in [notebook_html, (tmp_path / "pipeline_execution.html").read_text()]:
        visible = _visible_text(rendered)
        assert "MNLogit Regression Results" in visible
        assert f"Average Treatment Effect (ATE): {estimate:.4f}" in visible
        assert "Confidence interval: unavailable (no validated interval for this estimator)." in visible
        assert visible.count("UPLIFT MODELING RESULTS") == 1
