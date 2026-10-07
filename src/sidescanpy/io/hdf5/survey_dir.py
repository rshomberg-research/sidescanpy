"""
Helpers for reading and rewriting directories of ``.ss.h5`` files.

- :func:`read_survey_dir` -- one frequency channel of a directory of files,
  lazily concatenated along ``ping``.
- :func:`concat_survey_frequency`, :func:`available_frequencies`,
  :func:`load_channel_dataset` -- lower-level pieces of the same multi-file
  convention (a file or a directory both work for the latter two).
- :func:`check_disk_space`, :func:`atomic_write_survey` -- safe in-place
  rewrite support.

2026 rshomberg@uri.edu
"""

import logging
import shutil
from pathlib import Path

import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5.reader import read_survey
from sidescanpy.io.hdf5.writer import write_survey

logger = logging.getLogger(__name__)


def concat_survey_frequency(files: list[Path], freq_key: str, chunks: dict | None) -> tuple["xr.Dataset | None", dict]:
    """Lazily concatenate ``freq_key``'s channel across every file in
    ``files`` (sorted, so typically time order) into one dask-backed
    dataset spanning the whole survey -- no per-file restart.

    Most callers want :func:`read_survey_dir` or
    :func:`load_channel_dataset` instead, which wrap this.

    Returns ``(concatenated_ds, file_start_ping)`` where
    ``file_start_ping[path]`` is that file's own first ping's position in
    the concatenated dataset's global ping axis (0 for the first file).
    ``concatenated_ds`` is ``None`` if no file has this frequency at all.
    """
    logger.debug(f"Concatenating {freq_key!r} across {len(files)} file(s)")
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


def read_survey_dir(path: Path, *, freq_key: str, chunks: dict | None = None) -> xr.Dataset:
    """Read a directory of ``.ss.h5`` files as one lazily concatenated
    dataset of ``freq_key``.

    Files are taken in sorted filename order. Thin wrapper over
    :func:`concat_survey_frequency` that drops the per-file start-ping map.

    Raises ``FileNotFoundError`` if ``path`` has no ``.ss.h5`` files and
    ``ValueError`` if none of them carries ``freq_key``.
    """
    files = sorted(Path(path).glob("*.ss.h5"))
    if not files:
        raise FileNotFoundError(f"No .ss.h5 files found in {path}")
    ds, _file_start_ping = concat_survey_frequency(files, freq_key, chunks)
    if ds is None:
        raise ValueError(f"{freq_key!r}: no file in {path} has this frequency channel")
    logger.debug(f"Read {len(files)} file(s) from {path}: {freq_key!r}, {ds.sizes['ping']} pings")
    return ds


def available_frequencies(input_path: Path, chunks: dict | None) -> set[str]:
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


def load_channel_dataset(input_path: Path, freq_key: str, chunks: dict | None) -> "xr.Dataset":
    """One frequency channel's dataset from ``input_path``.

    ``input_path`` may be a single ``.ss.h5`` file (original behavior,
    byte-identical -- calls ``read_survey`` exactly as every caller
    already did directly) or a directory of them, lazily dask-concatenated
    in filename/acquisition order via :func:`concat_survey_frequency` --
    the same multi-file convention ``sidescan-export``'s own
    ``--ss-h5-dir`` already uses. This lets a survey that was never concatenated into one file at
    conversion time (e.g. for memory safety) still be analyzed as one
    logical survey.

    Raises ``FileNotFoundError``/``ValueError`` if ``freq_key`` isn't
    present anywhere at ``input_path`` -- never returns ``None``, unlike
    :func:`concat_survey_frequency` itself, so every caller here can
    treat this as a normal "the dataset I asked for" return.
    """
    if input_path.is_dir():
        return read_survey_dir(input_path, freq_key=freq_key, chunks=chunks)
    survey = read_survey(input_path, chunks=chunks)
    if freq_key not in survey.frequencies:
        raise ValueError(f"{freq_key!r}: not present in {input_path} (available: {sorted(survey.frequencies)})")
    return survey.get_frequency(freq_key).dataset


def check_disk_space(path: Path, multiplier: float) -> bool:
    """True if there's plausibly enough free space to safely rewrite
    ``path`` in place (conservative flat multiplier of its current size)."""
    required = path.stat().st_size * multiplier
    free = shutil.disk_usage(path.parent).free
    logger.debug(f"Disk check {path.name}: need ~{required:.0f} B, free {free} B")
    return free > required


def atomic_write_survey(survey: SideScanSurvey, path: Path) -> None:
    """Write ``survey`` to a temp file next to ``path``, then replace
    ``path`` only on full success (a failed write leaves ``path`` intact)."""
    logger.debug(f"Atomic write {path}")
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    write_survey(survey, tmp_path, overwrite=False)
    tmp_path.replace(path)
