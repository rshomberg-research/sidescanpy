"""
Shared CLI helpers: logging flags/setup and ``--frequency`` resolution.

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

from sidescanpy.log import configure_logging

logger = logging.getLogger(__name__)

# WARNINGs raised while parsing arguments, before logging is configured
# (the package's NullHandler would otherwise swallow them).
_pending_warnings: list[tuple[logging.Logger, str, tuple]] = []


def warn_deferred(log: logging.Logger, msg: str, *args) -> None:
    """``log.warning(msg, *args)`` now if logging is configured, else held
    until :func:`setup_logging_from_args` runs."""
    if logging.getLogger().handlers:
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


def uncorrected_alias(*old_names: str):
    """argparse ``type=`` callable mapping deprecated spellings of the
    uncorrected-signal choice to ``"uncorrected"`` with a WARNING.

    Kept for one release so existing scripts and configs keep working.
    """
    def _convert(value: str) -> str:
        if value in old_names:
            warn_deferred(logger, "%r is deprecated; use 'uncorrected'", value)
            return "uncorrected"
        return value
    return _convert


class _DeprecatedAlias(argparse.Action):
    def __init__(self, option_strings, dest, *, replacement: str, **kwargs):
        self.replacement = replacement
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warn_deferred(logger, "%s is deprecated; use %s", option_string, self.replacement)
        setattr(namespace, self.dest, values)


def add_deprecated_alias(parser: argparse.ArgumentParser, old_flag: str, new_flag: str, **kwargs) -> None:
    """Add a hidden ``old_flag`` that sets ``new_flag``'s value with a WARNING.

    ``kwargs`` (``type``, ``choices``) should match ``new_flag``'s. Kept for
    one release so existing scripts keep working.
    """
    dest = new_flag.lstrip("-").replace("-", "_")
    parser.add_argument(
        old_flag, dest=dest, action=_DeprecatedAlias, replacement=new_flag,
        default=argparse.SUPPRESS, help=argparse.SUPPRESS, **kwargs,
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
    while _pending_warnings:
        log, msg, args = _pending_warnings.pop(0)
        log.warning(msg, *args)


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
