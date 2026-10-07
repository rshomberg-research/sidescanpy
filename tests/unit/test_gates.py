"""
Source-wide content gates.

2026 rshomberg@uri.edu
"""

import re
from pathlib import Path

import sidescanpy

SRC = Path(sidescanpy.__file__).parent

_STAGE = re.compile(r"\bstage\b", re.IGNORECASE)


def test_no_stage_wording_in_library():
    """Workflow "stages" belong to callers; the library has steps and results."""
    hits = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in {".py", ".json", ".yaml", ".yml", ".txt", ".md"} or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC)
        if _STAGE.search(str(rel)):
            hits.append(f"{rel}: file name")
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if _STAGE.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    assert not hits, "whole-word 'stage' found in src/sidescanpy:\n" + "\n".join(hits)
