"""Executable public quickstart and documentation integrity checks."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_regression_example():
    readme = (ROOT / 'README.md').read_text()
    example = re.findall(r'```python\n(.*?)```', readme, re.S)[0]
    exec(compile(example, str(ROOT / 'README.md'), 'exec'), {})


def test_readme_panel_example(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    readme = (ROOT / 'README.md').read_text()
    example = re.findall(r'```python\n(.*?)```', readme, re.S)[1]
    exec(compile(example, str(ROOT / 'README.md'), 'exec'), {})
    output = tmp_path / 'output' / 'minimum_wage'
    report = json.loads((output / 'execution_report.json').read_text())
    assert report['export']['state'] == 'completed'
    assert all(node['state'] in {'completed', 'passed'} for node in report['nodes'].values())
    assert (output / 'pipeline_execution.html').stat().st_size > 0


def test_readme_local_links_resolve():
    for target in re.findall(r'\]\(([^)]+)\)', (ROOT / 'README.md').read_text()):
        if '://' in target or target.startswith('#'):
            continue
        assert (ROOT / target.split('#')[0]).exists(), target


def test_documented_basic_graph():
    source = (ROOT / 'docs/pipeline-guide.md').read_text()
    example = re.findall(r'```python\n(.*?)```', source, re.S)[0]
    exec(compile(example, str(ROOT / 'docs/pipeline-guide.md'), 'exec'), {})
