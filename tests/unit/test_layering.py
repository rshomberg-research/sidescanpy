"""
Enforce the package dependency direction: imports only point leftward in

    core <- io <- processing <- analysis <- grids <- targets <- export <- viz <- cli

Parses every module under ``src/sidescanpy`` with ``ast`` (including
function-level lazy imports). Root modules ``log``, ``_version`` and
``__init__`` count as core-level. ``analysis`` and ``processing`` must also
never import matplotlib (plotting lives in ``viz``).

2026 rshomberg@uri.edu
"""

import ast
import subprocess
import sys
from pathlib import Path

import sidescanpy

_ORDER = ["core", "io", "processing", "analysis", "grids", "targets", "export", "viz", "cli"]
_RANK = {name: i for i, name in enumerate(_ORDER)}
_ROOT_CORE = {"log", "_version", "__init__"}

# (importing module, imported package) pairs that violate the order and are
# known / out of scope for now. Keep as small as possible.
_KNOWN_VIOLATIONS = {
    # Catalog readers build the format-agnostic Target dataclass (targets.core is numpy-only).
    ("io.jsf.targets", "targets"),
    ("io.sdf.targets", "targets"),
}


def _package_of(parts: list[str]) -> str | None:
    """Top-level layer of a dotted module path below ``sidescanpy``."""
    if not parts:
        return "core"
    if parts[0] in _RANK:
        return parts[0]
    if parts[0] in _ROOT_CORE:
        return "core"
    return None


def _imports(tree: ast.AST, module_parts: list[str], is_pkg: bool):
    pkg_parts = module_parts if is_pkg else module_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "sidescanpy":
                    yield alias.name.split(".")[1:]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                yield base + (node.module.split(".") if node.module else [])
            elif node.module and node.module.split(".")[0] == "sidescanpy":
                parts = node.module.split(".")[1:]
                if not parts:
                    # ``from sidescanpy import x`` -- x may be a subpackage.
                    for alias in node.names:
                        yield [alias.name]
                else:
                    yield parts


def _find_violations() -> set[tuple[str, str]]:
    root = Path(sidescanpy.__file__).parent
    found = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        src_layer = _package_of(parts)
        if src_layer is None:
            continue
        tree = ast.parse(path.read_text())
        for target in _imports(tree, parts, is_pkg):
            dst_layer = _package_of(target)
            if dst_layer is None or dst_layer == src_layer:
                continue
            if _RANK[dst_layer] > _RANK[src_layer]:
                found.add((".".join(parts), dst_layer))
    return found


def test_imports_only_point_leftward():
    violations = _find_violations()
    unexpected = sorted(violations - _KNOWN_VIOLATIONS)
    assert not unexpected, f"layering violations (module, imported layer): {unexpected}"


def test_known_violations_are_still_violations():
    stale = sorted(_KNOWN_VIOLATIONS - _find_violations())
    assert not stale, f"remove fixed entries from _KNOWN_VIOLATIONS: {stale}"


_NO_MATPLOTLIB_LAYERS = ("analysis", "processing")


def _imports_matplotlib(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "matplotlib" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module.split(".")[0] == "matplotlib":
                return True
    return False


def test_analysis_and_processing_do_not_import_matplotlib():
    root = Path(sidescanpy.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for layer in _NO_MATPLOTLIB_LAYERS
        for path in sorted((root / layer).rglob("*.py"))
        if _imports_matplotlib(ast.parse(path.read_text()))
    ]
    assert not offenders, f"matplotlib imported in compute layers: {offenders}"


def test_importing_analysis_and_processing_does_not_load_matplotlib():
    code = (
        "import sys, sidescanpy.analysis, sidescanpy.processing; "
        "sys.exit(1 if any(m == 'matplotlib' or m.startswith('matplotlib.') for m in sys.modules) else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
