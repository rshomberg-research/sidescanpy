"""
Shared per-file processing loop for the ``sidescan-correct``/
``sidescan-convert`` in-place ``.ss.h5``-editing CLIs.

Promoted out of the old ``02_crop_farrange.py``/``04_radiometric.py``/
``06_geometric.py``/``08_georef.py`` pipeline scripts, which each
duplicated this exact loop (disk-space check, atomic temp-file+rename
write, per-file skip/success/fail logging, summary) around one
different one-line body — confirmed byte-identical apart from that body
across both `project-b` and `project-a`'s own copies.

2026 rshomberg@uri.edu
"""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey

logger = logging.getLogger(__name__)


def check_disk_space(path: Path, multiplier: float) -> bool:
    """True if there's plausibly enough free space to safely rewrite
    ``path`` in place (conservative flat multiplier)."""
    required = path.stat().st_size * multiplier
    free = shutil.disk_usage(path.parent).free
    return free > required


def atomic_write_survey(survey: SideScanSurvey, path: Path) -> None:
    """Write to a temp file, then replace ``path`` only on full success."""
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    write_survey(survey, tmp_path, overwrite=False)
    tmp_path.replace(path)


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
            logger.error(f"    ✗ {result['status']}: {result['error']}")
            continue

        try:
            survey = read_survey(path)
        except Exception as e:
            result["status"] = "failed_read"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    ✗ {result['status']}: {result['error']}")
            continue

        if already_done_fn(survey):
            result["status"] = "skipped"
            results.append(result)
            logger.info("    ⤤ skipped (already done)")
            continue

        try:
            survey = step_fn(survey)
            atomic_write_survey(survey, path)
        except Exception as e:
            result["status"] = "failed_process"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    ✗ {result['status']}: {result['error']}")
            continue

        result["status"] = "success"
        results.append(result)
        logger.info(f"    ✓ {stage_label} applied in place")

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
