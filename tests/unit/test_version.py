"""sidescanpy.__version__ -- must track the installed package's real
metadata, not a hardcoded literal (found stuck at "0.1.0" while
pyproject.toml had moved through the whole 0.4.x series -- a real
provenance bug, since this value gets stamped into real .ss.h5
processing_history records, not just printed for humans).
"""

import tomllib
from pathlib import Path

import sidescanpy


def test_version_matches_installed_metadata():
    from importlib.metadata import version

    assert sidescanpy.__version__ == version("sidescanpy")


def test_version_matches_pyproject_toml():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    assert sidescanpy.__version__ == pyproject["project"]["version"]


def test_version_is_not_the_old_stale_literal():
    assert sidescanpy.__version__ != "0.1.0"
