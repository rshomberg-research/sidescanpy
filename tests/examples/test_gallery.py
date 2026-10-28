"""
The gallery scripts in ``examples/gallery/`` download real recordings,
so CI doesn't run them (``make gallery`` does). This checks what can be
checked offline: each script parses, fetches only registered datasets,
and has a docs page that includes it and the image it writes.

2026 rshomberg@uri.edu
"""

import ast
import re
from pathlib import Path

import pytest

from sidescanpy import datasets

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "examples" / "gallery").glob("*.py"))


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


def test_gallery_has_scripts():
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_gallery_script_fetches_registered_datasets(script):
    names = _fetched_names(script)
    assert names, f"{script.name} fetches no dataset"
    assert set(names) <= set(datasets.REGISTRY), set(names) - set(datasets.REGISTRY)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_gallery_script_has_docs_page_and_image(script):
    page = ROOT / "docs" / "examples" / f"{script.stem}.md"
    assert page.exists()
    text = page.read_text()
    assert f'--8<-- "examples/gallery/{script.name}"' in text
    assert (ROOT / "docs" / "assets" / "gallery" / f"{script.stem}.png").exists()
