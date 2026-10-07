"""Every module in the package must import cleanly."""

import importlib
import pkgutil

import pytest

import sidescanpy

# Modules that only import when an optional extra is installed:
# module name -> the optional top-level dependency it needs.
OPTIONAL_DEPS: dict[str, str] = {}

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(sidescanpy.__path__, prefix="sidescanpy.")
)


def test_modules_discovered():
    assert "sidescanpy.core.survey" in MODULES
    assert len(MODULES) > 20


@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name):
    dep = OPTIONAL_DEPS.get(module_name)
    if dep is not None:
        pytest.importorskip(dep)
    importlib.import_module(module_name)
