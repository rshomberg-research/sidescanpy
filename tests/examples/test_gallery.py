"""
The gallery scripts in ``examples/gallery/`` and ``examples/theory/`` download real recordings,
so the unit suite doesn't run them (the docs build does, through
sphinx-gallery). This checks what can be checked offline: each script
parses, fetches only registered datasets, and starts with the titled
docstring sphinx-gallery needs.

2026 rshomberg@uri.edu
"""

import ast
import re
from pathlib import Path

import pytest

from sidescanpy import datasets

ROOT = Path(__file__).resolve().parents[2]
GALLERIES = ("gallery", "theory")
SCRIPTS = sorted(p for g in GALLERIES for p in (ROOT / "examples" / g).glob("plot_*.py"))


def _fetched_names(script: Path) -> list[str]:
    tree = ast.parse(script.read_text())
    names = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fetch"):
            names += [a.value for a in node.args if isinstance(a, ast.Constant)]
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and re.fullmatch(r"[a-z_0-9]+\.(jsf|xtf|sdf)", node.value, re.I):
            names.append(node.value)
    return names


def test_gallery_has_scripts_and_header():
    for g in GALLERIES:
        assert list((ROOT / "examples" / g).glob("plot_*.py")), g
        assert (ROOT / "examples" / g / "GALLERY_HEADER.rst").exists(), g


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_gallery_script_fetches_registered_datasets(script):
    names = _fetched_names(script)
    assert names, f"{script.name} fetches no dataset"
    assert set(names) <= set(datasets.REGISTRY), set(names) - set(datasets.REGISTRY)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_gallery_script_docstring_has_title(script):
    doc = ast.get_docstring(ast.parse(script.read_text()), clean=True)
    assert doc, f"{script.name} has no module docstring"
    title, underline = doc.splitlines()[:2]
    assert title and set(underline) == {"="} and len(underline) >= len(title)
