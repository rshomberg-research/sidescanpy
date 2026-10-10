"""
Shared CLI plumbing: logging flags/setup, the per-subcommand
``--config``/``--strict-config``/logging options, ``--frequency``
resolution, ``.ss.h5`` directory listing, JSON/PNG output, and the
matplotlib (``viz`` extra) check.

2026 rshomberg@uri.edu
"""

import argparse
import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sidescanpy.log import configure_logging

logger = logging.getLogger(__name__)

# WARNINGs raised while parsing arguments, before logging is configured
# (the package's NullHandler would otherwise swallow them).
_pending_warnings: list[tuple[logging.Logger, str, tuple]] = []


_force_defer = False


@contextlib.contextmanager
def deferred_warnings() -> Iterator[None]:
    """Queue every :func:`warn_deferred` warning while active, even when
    logging is already configured (argument parsing may run twice)."""
    global _force_defer
    previous, _force_defer = _force_defer, True
    try:
        yield
    finally:
        _force_defer = previous


def pending_warnings_mark() -> int:
    """Position in the deferred-warning queue (see :func:`discard_pending_warnings`)."""
    return len(_pending_warnings)


def discard_pending_warnings(mark: int) -> None:
    """Drop warnings queued after ``mark`` (used before a repeated parse)."""
    del _pending_warnings[mark:]


def warn_deferred(log: logging.Logger, msg: str, *args) -> None:
    """``log.warning(msg, *args)`` now if logging is configured, else held
    until :func:`setup_logging_from_args` runs."""
    if logging.getLogger().handlers and not _force_defer:
        log.warning(msg, *args)
    else:
        _pending_warnings.append((log, msg, args))


def add_logging_args(parser: argparse.ArgumentParser, *, subcommand: bool = False) -> None:
    """Add the shared ``-v/--verbose``, ``-q/--quiet`` and ``--log-file`` flags.

    With ``subcommand=True`` the defaults are ``argparse.SUPPRESS`` so a
    copy on a subparser only adds the ability to pass the flag after the
    subcommand -- a plain default would silently overwrite a value already
    set by the top-level parser (e.g. ``sidescan-convert -v survey ...``).
    """
    default = {"default": argparse.SUPPRESS} if subcommand else {}
    parser.add_argument(
        "-v", "--verbose", action="store_true", **default,
        help="Show DEBUG-level library detail (parameters used, defaults applied, fallbacks)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", **default,
        help="Only show warnings and errors",
    )
    parser.add_argument(
        "--log-file", type=Path, default=argparse.SUPPRESS if subcommand else None,
        help="Also write log output to this file",
    )


def add_common_args(parser: argparse.ArgumentParser, *, subcommand: bool = True) -> None:
    """Add ``--config``, ``--strict-config`` and (see :func:`add_logging_args`)
    the logging flags to a (sub)command parser."""
    word = "subcommand" if subcommand else "command"
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"YAML or JSON file seeding this {word}'s defaults -- explicit flags still override",
    )
    parser.add_argument(
        "--strict-config", action="store_true",
        help=f"Treat config-file keys this {word} does not recognize as an error instead of a warning",
    )
    add_logging_args(parser, subcommand=subcommand)


def add_plots_arg(parser: argparse.ArgumentParser, *, help: str = "Write the evidence PNG(s) (needs matplotlib, the 'viz' extra)") -> None:
    """Add ``--plots/--no-plots`` (default on)."""
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help=help)


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Configure logging from the flags added by :func:`add_logging_args`."""
    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)
    log_file = getattr(args, "log_file", None)
    if quiet and not verbose:
        configure_logging("WARNING", log_file=log_file)
    else:
        configure_logging("INFO", log_file=log_file, library_level="DEBUG" if verbose else None)
    flush_pending_warnings()


def flush_pending_warnings() -> None:
    """Log every queued :func:`warn_deferred` warning now."""
    while _pending_warnings:
        log, msg, args = _pending_warnings.pop(0)
        log.warning(msg, *args)


VIZ_INSTALL_HINT = "pip install 'sidescanpy[viz]'"


def has_matplotlib() -> bool:
    """True if matplotlib can be imported; also selects the headless Agg
    backend so later pyplot imports never probe for a GUI."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    return True


def require_matplotlib(command: str) -> bool:
    """:func:`has_matplotlib`, logging a one-line ERROR naming ``command``
    and the ``viz`` extra when it is missing."""
    if has_matplotlib():
        return True
    logger.error(f"{command} needs matplotlib: {VIZ_INSTALL_HINT}")
    return False


def ss_h5_files(directory: Path) -> list[Path]:
    """Sorted ``*.ss.h5`` files in ``directory``; logs an ERROR when there are none."""
    files = sorted(Path(directory).glob("*.ss.h5"))
    if not files:
        logger.error(f"No .ss.h5 files found in {directory}")
    return files


def write_json(result: Any, path: Path) -> Path:
    """Write ``result`` (its ``to_dict()`` if it has one) as indented JSON."""
    data = result.to_dict() if hasattr(result, "to_dict") else result
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def save_plot(fig, path: Path, *, dpi: int = 140, tight: bool = False) -> Path:
    """Save and close ``fig`` (needs matplotlib), logging the path at DEBUG."""
    from sidescanpy.cli._figures import save_figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, dpi=dpi, tight=tight)
    logger.debug(f"wrote {path}")
    return path


def resolve_freq_keys(available: "list[str] | set[str]", requested: str) -> list[str]:
    """Resolve a ``--frequency`` CLI value to one or more real freq_keys.

    ``"all"`` -> every freq_key in ``available`` (sorted). Otherwise:
    JSF/SDF's own convention is ``freq_key = f"hz_{freq_hz_start}"`` --
    tried first, so every existing caller passing a bare frequency
    number (e.g. ``--frequency 1495000``) keeps resolving exactly as
    before. Not every consumer uses that convention (e.g. a vendor HDF5
    format using ``"sensor<N>_data<M>"``), so ``requested`` itself is tried as a
    literal freq_key when the ``"hz_"``-prefixed guess isn't a real key,
    rather than assuming every survey names its channels that way.
    """
    if requested == "all":
        return sorted(available)
    candidate = f"hz_{requested}"
    if candidate in available:
        return [candidate]
    return [requested]
