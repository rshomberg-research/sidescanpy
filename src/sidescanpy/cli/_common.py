"""
Shared per-file processing loop for the ``sidescan-correct``/
``sidescan-convert`` in-place ``.ss.h5``-editing CLIs.

Promoted out of the old ``02_crop_farrange.py``/``04_radiometric.py``/
``06_geometric.py``/``08_georef.py`` pipeline scripts, which each
duplicated this exact loop (disk-space check, atomic temp-file+rename
write, per-file skip/success/fail logging, summary) around one
different one-line body.

:func:`process_files_in_place_chunked` is the ping-chunked counterpart,
for callers needing memory bounded to roughly one chunk's worth of
derived arrays instead of the whole file — see
:mod:`sidescanpy.processing.chunked` for the actual per-file mechanics
and the real memory numbers behind why it exists.

2026 rshomberg@uri.edu
"""

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import atomic_write_survey, check_disk_space, read_survey
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


def process_files_in_place_chunked(
    files: list[Path],
    steps: frozenset[str] | set[str] | tuple[str, ...],
    step_params: dict[str, dict] | None,
    ping_chunk_size: int,
    disk_multiplier: float,
    stage_label: str,
) -> int:
    """Ping-chunked counterpart to :func:`process_files_in_place`, for
    callers that need memory bounded to roughly one chunk's worth of
    derived arrays instead of the whole file — see
    :mod:`sidescanpy.processing.chunked`'s own module docstring for why
    this exists and the real memory numbers behind it. Same disk-space
    guard, same per-file success/skipped/failed accounting and log
    shape as :func:`process_files_in_place`, so a caller switching
    between the two sees consistent output either way.

    Unlike the non-chunked path, ``step_fn``/``already_done_fn``
    aren't caller-supplied here — chunking only makes sense for the
    fixed, per-ping-independent step set
    :func:`sidescanpy.processing.chunked.correct_file_chunked` already
    knows how to check/apply per chunk (``crop_farrange`` is rejected
    there, not accepted and silently mishandled here).
    """
    from sidescanpy.processing.chunked import correct_file_chunked

    if not files:
        logger.error("No .ss.h5 files found")
        return 1

    logger.info(f"Found {len(files)} .ss.h5 file(s)")
    start_time = time.time()
    results = []
    for i, path in enumerate(files, 1):
        elapsed = time.time() - start_time
        logger.info(f"[{i}/{len(files)}] {path.name}  (elapsed: {elapsed:.0f}s, chunk={ping_chunk_size} pings)")
        result = {"path": path, "status": "", "error": ""}

        if not check_disk_space(path, disk_multiplier):
            result["status"] = "failed_disk_space"
            result["error"] = (
                f"Insufficient free space to safely process {path.name} "
                f"(need ~{disk_multiplier:.1f}x its current size)"
            )
            results.append(result)
            logger.error(f"    FAILED {result['status']}: {result['error']}")
            continue

        try:
            applied = correct_file_chunked(path, steps, step_params=step_params, ping_chunk_size=ping_chunk_size)
        except Exception as e:
            result["status"] = "failed_process"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    FAILED {result['status']}: {result['error']}")
            continue

        if applied:
            result["status"] = "success"
            logger.debug(f"    {stage_label} applied in place (chunked)")
        else:
            result["status"] = "skipped"
            logger.debug("    skipped (already done)")
        results.append(result)

    total_elapsed = time.time() - start_time
    n_success = sum(1 for r in results if r["status"] == "success")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_failed = sum(1 for r in results if r["status"].startswith("failed"))

    logger.info("=" * 60)
    logger.info(f"{stage_label} complete in {total_elapsed:.0f}s")
    logger.info(f"  Success: {n_success}/{len(files)}")
    logger.info(f"  Skipped: {n_skipped}")
    logger.info(f"  Failed:  {n_failed}")

    if n_failed > 0:
        logger.warning("Failures:")
        for r in results:
            if r["status"].startswith("failed"):
                logger.warning(f"  {r['path'].name}: {r['error']}")
        return 2

    return 0


def process_files_in_place(
    files: list[Path],
    step_fn: Callable[[SideScanSurvey], SideScanSurvey],
    already_done_fn: Callable[[SideScanSurvey], bool],
    disk_multiplier: float,
    stage_label: str,
) -> int:
    """Read, conditionally process, and atomically rewrite each of
    ``files`` in place.

    Parameters
    ----------
    files : list[Path]
        ``.ss.h5`` files to process, in order.
    step_fn : callable
        ``survey -> survey``. Only called for a file where
        ``already_done_fn(survey)`` is False. May itself apply only a
        subset of requested work if part of it is already done (see
        callers in ``convert.py``/``correct.py``) — this loop doesn't
        care, it just reads once, calls this once, and writes once.
    already_done_fn : callable
        ``survey -> bool``. True skips the file entirely (no write).
    disk_multiplier : float
        Conservative flat multiplier on the file's current size used as
        the required-free-space check before reading/processing.
    stage_label : str
        Used only in log messages.

    Returns
    -------
    int
        0 if every file succeeded or was skipped, 1 if ``files`` was
        empty, 2 if any file failed.
    """
    if not files:
        logger.error("No .ss.h5 files found")
        return 1

    logger.info(f"Found {len(files)} .ss.h5 file(s)")
    start_time = time.time()
    results = []
    for i, path in enumerate(files, 1):
        elapsed = time.time() - start_time
        logger.info(f"[{i}/{len(files)}] {path.name}  (elapsed: {elapsed:.0f}s)")
        result = {"path": path, "status": "", "error": ""}

        if not check_disk_space(path, disk_multiplier):
            result["status"] = "failed_disk_space"
            result["error"] = (
                f"Insufficient free space to safely process {path.name} "
                f"(need ~{disk_multiplier:.1f}x its current size)"
            )
            results.append(result)
            logger.error(f"    FAILED {result['status']}: {result['error']}")
            continue

        try:
            survey = read_survey(path)
        except Exception as e:
            result["status"] = "failed_read"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    FAILED {result['status']}: {result['error']}")
            continue

        if already_done_fn(survey):
            result["status"] = "skipped"
            results.append(result)
            logger.debug("    skipped (already done)")
            continue

        try:
            survey = step_fn(survey)
            atomic_write_survey(survey, path)
        except Exception as e:
            result["status"] = "failed_process"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    FAILED {result['status']}: {result['error']}")
            continue

        result["status"] = "success"
        results.append(result)
        logger.debug(f"    {stage_label} applied in place")

    total_elapsed = time.time() - start_time
    n_success = sum(1 for r in results if r["status"] == "success")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_failed = sum(1 for r in results if r["status"].startswith("failed"))

    logger.info("=" * 60)
    logger.info(f"{stage_label} complete in {total_elapsed:.0f}s")
    logger.info(f"  Success: {n_success}/{len(files)}")
    logger.info(f"  Skipped: {n_skipped}")
    logger.info(f"  Failed:  {n_failed}")

    if n_failed > 0:
        logger.warning("Failures:")
        for r in results:
            if r["status"].startswith("failed"):
                logger.warning(f"  {r['path'].name}: {r['error']}")
        return 2

    return 0
