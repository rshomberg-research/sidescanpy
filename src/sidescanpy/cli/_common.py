"""
Shared CLI helpers: logging flags/setup and ``--frequency`` resolution.

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

from sidescanpy.log import configure_logging

logger = logging.getLogger(__name__)


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


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Configure logging from the flags added by :func:`add_logging_args`."""
    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)
    log_file = getattr(args, "log_file", None)
    if quiet and not verbose:
        configure_logging("WARNING", log_file=log_file)
    else:
        configure_logging("INFO", log_file=log_file, library_level="DEBUG" if verbose else None)


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
