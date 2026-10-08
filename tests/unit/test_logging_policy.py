"""Logging policy gates and tests for sidescanpy.log.

Library code (everything outside ``cli/``) must not call ``logger.info``,
``print`` or ``basicConfig``, and every module defines ``logger``.
"""

import ast
import logging
from pathlib import Path

import pytest

import sidescanpy
from sidescanpy.log import configure_logging, log_timing

SRC = Path(sidescanpy.__file__).parent

# Modules with no executable logic that legitimately have no logger.
NO_LOGGER_OK: set[str] = {
    "_version.py",  # version lookup only
    "log.py",  # defines the logging helpers themselves
    "core/units.py",  # pure numeric helpers, nothing worth logging
}


def _library_files():
    return sorted(p for p in SRC.rglob("*.py") if "cli" not in p.relative_to(SRC).parts)


def _violations(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "print":
            yield node.lineno, "print("
        elif isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else None
            if f.attr == "info" and base in {"logger", "logging", "log"}:
                yield node.lineno, f"{base}.info("
            elif f.attr == "basicConfig":
                yield node.lineno, "basicConfig"


def test_no_info_print_or_basicconfig_in_library():
    bad = []
    for path in _library_files():
        for lineno, what in _violations(ast.parse(path.read_text())):
            bad.append(f"{path.relative_to(SRC)}:{lineno}: {what}")
    assert not bad, "Library code must log at DEBUG/WARNING only:\n" + "\n".join(bad)


def test_every_module_defines_logger():
    missing = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if path.name == "__init__.py" or str(rel) in NO_LOGGER_OK:
            continue
        tree = ast.parse(path.read_text())
        names = {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        if "logger" not in names:
            missing.append(str(rel))
    assert not missing, f"Modules without module-level `logger`: {missing}"


def test_package_has_null_handler():
    handlers = logging.getLogger("sidescanpy").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, logging.getLogger("sidescanpy").level)
    yield root
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])
    logging.getLogger("sidescanpy").setLevel(saved[2])


def _tagged(root):
    return [h for h in root.handlers if getattr(h, "_sidescanpy_configured", False)]


def test_configure_logging_idempotent(clean_root, tmp_path):
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(_tagged(clean_root)) == 1
    configure_logging("INFO", log_file=tmp_path / "a.log")
    configure_logging("INFO", log_file=tmp_path / "a.log")
    assert len(_tagged(clean_root)) == 2


def test_configure_logging_library_level_and_file(clean_root, tmp_path):
    log_file = tmp_path / "sub" / "run.log"
    configure_logging("WARNING", log_file=log_file, library_level="DEBUG")
    assert logging.getLogger("sidescanpy").level == logging.DEBUG
    logging.getLogger("sidescanpy.test").debug("lib-debug")
    logging.getLogger("other").info("third-party-info")
    for h in _tagged(clean_root):
        h.flush()
    text = log_file.read_text()
    assert "lib-debug" in text and "DEBUG" in text
    assert "third-party-info" not in text


def test_configure_logging_quiets_noisy_loggers(clean_root):
    configure_logging("DEBUG")
    assert logging.getLogger("matplotlib").level == logging.WARNING


def test_configure_logging_rejects_bad_level(clean_root):
    with pytest.raises(ValueError):
        configure_logging("LOUD")


def test_log_timing(caplog):
    lg = logging.getLogger("sidescanpy.test_timing")
    with caplog.at_level(logging.DEBUG, logger=lg.name):
        with log_timing(lg, "step"):
            pass
    assert any(r.getMessage().startswith("step: ") and r.getMessage().endswith("s") for r in caplog.records)


def test_log_timing_logs_on_exception(caplog):
    lg = logging.getLogger("sidescanpy.test_timing")
    with caplog.at_level(logging.DEBUG, logger=lg.name):
        with pytest.raises(RuntimeError):
            with log_timing(lg, "boom", level=logging.INFO):
                raise RuntimeError
    assert any(r.getMessage().startswith("boom: ") and r.levelno == logging.INFO for r in caplog.records)
