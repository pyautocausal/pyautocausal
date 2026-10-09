"""Executable public quickstart and documentation integrity checks."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_regression_example():
    readme = (ROOT / 'README.md').read_text()
    example = re.findall(r'```python\n(.*?)```', readme, re.S)[0]
    exec(compile(example, str(ROOT / 'README.md'), 'exec'), {})


def test_readme_local_links_resolve():
    for target in re.findall(r'\]\(([^)]+)\)', (ROOT / 'README.md').read_text()):
        if '://' in target or target.startswith('#'):
            continue
        assert (ROOT / target.split('#')[0]).exists(), target


def test_documented_basic_graph():
    source = (ROOT / 'docs/pipeline-guide.md').read_text()
    example = re.findall(r'```python\n(.*?)```', source, re.S)[0]
    exec(compile(example, str(ROOT / 'docs/pipeline-guide.md'), 'exec'), {})
