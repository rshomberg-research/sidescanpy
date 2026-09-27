"""
Shared per-file processing loop for the ``sidescan-correct``/
``sidescan-convert`` in-place ``.ss.h5``-editing CLIs.

Promoted out of the old ``02_crop_farrange.py``/``04_radiometric.py``/
``06_geometric.py``/``08_georef.py`` pipeline scripts, which each
duplicated this exact loop (disk-space check, atomic temp-file+rename
write, per-file skip/success/fail logging, summary) around one
different one-line body — confirmed byte-identical apart from that body
across both `project-b` and `project-a`'s own copies.

:func:`process_files_in_place_chunked` is the ping-chunked counterpart,
for callers needing memory bounded to roughly one chunk's worth of
derived arrays instead of the whole file — see
:mod:`sidescanpy.processing.chunked` for the actual per-file mechanics
and the real memory numbers behind why it exists.

2026 rshomberg@uri.edu
"""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey

logger = logging.getLogger(__name__)


def resolve_freq_keys(available: "list[str] | set[str]", requested: str) -> list[str]:
    """Resolve a ``--frequency`` CLI value to one or more real freq_keys.

    ``"all"`` -> every freq_key in ``available`` (sorted). Otherwise:
    JSF/SDF's own convention is ``freq_key = f"hz_{freq_hz_start}"`` --
    tried first, so every existing caller passing a bare frequency
    number (e.g. ``--frequency 1495000``) keeps resolving exactly as
    before. Not every consumer uses that convention (e.g. vendor-format's
    ``"sensor<N>_data<M>"``), so ``requested`` itself is tried as a
    literal freq_key when the ``"hz_"``-prefixed guess isn't a real key,
    rather than assuming every survey names its channels that way.
    """
    if requested == "all":
        return sorted(available)
    candidate = f"hz_{requested}"
    if candidate in available:
        return [candidate]
    return [requested]


def concat_survey_frequency(files: list[Path], freq_key: str, chunks: dict) -> tuple["xr.Dataset | None", dict]:
    """Lazily concatenate ``freq_key``'s channel across every file in
    ``files`` (sorted, so typically time order) into one dask-backed
    dataset spanning the whole survey -- no per-file restart.

    Promoted here from ``cli/export.py`` (2026-09-27, unchanged body) so
    ``cli/analyze.py``'s subcommands can accept a multi-file
    ``--input``/``--ss-h5`` directory the same way ``sidescan-export``'s
    ``--ss-h5-dir`` already does -- see :func:`load_channel_dataset`/
    :func:`available_frequencies` below, which are the actual entry
    points most callers want.

    Returns ``(concatenated_ds, file_start_ping)`` where
    ``file_start_ping[path]`` is that file's own first ping's position in
    the concatenated dataset's global ping axis (0 for the first file).
    ``concatenated_ds`` is ``None`` if no file has this frequency at all.
    """
    datasets = []
    file_start_ping: dict[Path, int] = {}
    cumulative = 0
    for path in files:
        survey = read_survey(path, chunks=chunks)
        if freq_key not in survey.frequencies:
            continue
        ds = survey.get_frequency(freq_key).dataset
        file_start_ping[path] = cumulative
        cumulative += ds.sizes["ping"]
        datasets.append(ds)
    if not datasets:
        return None, {}
    return xr.concat(datasets, dim="ping"), file_start_ping


def available_frequencies(input_path: Path, chunks: dict) -> set[str]:
    """Every frequency key present at ``input_path`` -- a single
    ``.ss.h5`` file (its own ``.frequencies``) or a directory of them
    (the union across every file -- not every file need carry every key,
    same as :func:`concat_survey_frequency`'s own per-file tolerance).

    Raises ``FileNotFoundError`` if ``input_path`` is a directory with no
    ``.ss.h5`` files in it.
    """
    if input_path.is_dir():
        files = sorted(input_path.glob("*.ss.h5"))
        if not files:
            raise FileNotFoundError(f"No .ss.h5 files found in {input_path}")
        keys: set[str] = set()
        for p in files:
            keys |= set(read_survey(p, chunks=chunks).frequencies)
        return keys
    return set(read_survey(input_path, chunks=chunks).frequencies)


def load_channel_dataset(input_path: Path, freq_key: str, chunks: dict) -> "xr.Dataset":
    """One frequency channel's dataset from ``input_path``.

    ``input_path`` may be a single ``.ss.h5`` file (original behavior,
    byte-identical -- calls ``read_survey`` exactly as every caller
    already did directly) or a directory of them, lazily dask-concatenated
    in filename/acquisition order via :func:`concat_survey_frequency` --
    the same multi-file convention ``sidescan-export``'s own
    ``--ss-h5-dir`` already uses. Added 2026-09-27 so a real mission that
    was never concatenated into one file at conversion time (a genuine
    memory-safety need on a constrained container -- see
    ``downstream-project``'s own docs) can still be analyzed as one logical survey.

    Raises ``FileNotFoundError``/``ValueError`` if ``freq_key`` isn't
    present anywhere at ``input_path`` -- never returns ``None``, unlike
    :func:`concat_survey_frequency` itself, so every caller here can
    treat this as a normal "the dataset I asked for" return.
    """
    if input_path.is_dir():
        files = sorted(input_path.glob("*.ss.h5"))
        if not files:
            raise FileNotFoundError(f"No .ss.h5 files found in {input_path}")
        ds, _file_start_ping = concat_survey_frequency(files, freq_key, chunks)
        if ds is None:
            raise ValueError(f"{freq_key!r}: no file in {input_path} has this frequency channel")
        return ds
    survey = read_survey(input_path, chunks=chunks)
    if freq_key not in survey.frequencies:
        raise ValueError(f"{freq_key!r}: not present in {input_path} (available: {sorted(survey.frequencies)})")
    return survey.get_frequency(freq_key).dataset


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
        logger.debug(f"[{i}/{len(files)}] {path.name}  (elapsed: {elapsed:.0f}s, chunk={ping_chunk_size} pings)")
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
            applied = correct_file_chunked(path, steps, step_params=step_params, ping_chunk_size=ping_chunk_size)
        except Exception as e:
            result["status"] = "failed_process"
            result["error"] = f"{type(e).__name__}: {e}"
            results.append(result)
            logger.error(f"    ✗ {result['status']}: {result['error']}")
            continue

        if applied:
            result["status"] = "success"
            logger.debug(f"    ✓ {stage_label} applied in place (chunked)")
        else:
            result["status"] = "skipped"
            logger.debug("    ⤤ skipped (already done)")
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
        logger.debug(f"[{i}/{len(files)}] {path.name}  (elapsed: {elapsed:.0f}s)")
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
            logger.debug("    ⤤ skipped (already done)")
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
        logger.debug(f"    ✓ {stage_label} applied in place")

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
