"""
Run the Python code blocks of the guide pages that have no other test:
``docs/getting_started.md`` (its blocks build on each other, so they run
as one script, in order) and the quickstart on ``docs/index.md``. Each
script runs in a temporary directory and must finish without an
exception; on getting started, every line of each plain output block
(a fence with no language) must appear in what the script prints.

The blocks download a public recording, so like the gallery they need
the example data. Without ``SIDESCANPY_DOCS_CODE=1`` the tests skip
unless every recording they fetch is already in the cache
(:func:`sidescanpy.datasets.data_dir`); with it they download what is
missing. ``make docs-code`` sets it, and the ``docs`` workflow runs that
after the docs build, which has filled the cache.

2026 rshomberg@uri.edu
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sidescanpy import datasets

ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```", re.S | re.M)
FETCHED = re.compile(r"""datasets\.fetch\(\s*["']([^"']+)["']""")

pytestmark = pytest.mark.requires_data


def code_blocks(page: Path) -> list[tuple[str, str]]:
    """``(language, text)`` of every fenced block on a Markdown page, in order."""
    return FENCE.findall(page.read_text())


def python_script(blocks) -> str:
    return "\n".join(text for lang, text in blocks if lang == "python")


def expected_lines(blocks) -> list[str]:
    """The lines of plain output blocks that follow a Python block."""
    lines = []
    for (prev_lang, _), (lang, text) in zip(blocks, blocks[1:]):
        if prev_lang == "python" and lang == "":
            lines += [line for line in text.splitlines() if line.strip()]
    return lines


def need_data(script: str) -> None:
    """Skip unless the recordings the script fetches are cached or may be downloaded."""
    names = FETCHED.findall(script)
    assert names, "the script fetches no dataset"
    if os.environ.get("SIDESCANPY_DOCS_CODE") == "1":
        for name in names:
            datasets.fetch(name)
        return
    missing = [n for n in names if not (datasets.data_dir() / n).exists()]
    if missing:
        pytest.skip(f"not cached: {missing}; set SIDESCANPY_DOCS_CODE=1 (make docs-code) to download")


def run(script: str, cwd: Path) -> str:
    path = cwd / "docs_code.py"
    path.write_text(script)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(path)], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, f"{proc.stderr[-4000:]}"
    return proc.stdout


@pytest.mark.parametrize("page", ["getting_started.md", "index.md"])
def test_page_has_python_blocks(page):
    assert python_script(code_blocks(ROOT / "docs" / page)).strip()


def test_getting_started_runs_and_prints_what_the_page_shows(tmp_path):
    blocks = code_blocks(ROOT / "docs" / "getting_started.md")
    script = python_script(blocks)
    need_data(script)
    stdout = run(script, tmp_path)
    missing = [line for line in expected_lines(blocks) if line not in stdout]
    assert not missing, f"output shown on the page but not printed:\n{missing}\nprinted:\n{stdout}"
    for name in ("MIS11_0.ss.h5", "MIS11_0.corrected.ss.h5", "MIS11_0_raw.png", "MIS11_0_400khz.tif"):
        assert (tmp_path / name).exists(), name


def test_index_quickstart_runs(tmp_path):
    script = python_script(code_blocks(ROOT / "docs" / "index.md"))
    need_data(script)
    run(script, tmp_path)
    assert (tmp_path / "MIS11_0_400khz.tif").exists()
