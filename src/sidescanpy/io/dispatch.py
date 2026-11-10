"""
Format-dispatching helpers for sonar file ingestion.

Given a list of file paths, pick the right per-format reader/converter
by extension and produce a validated SideScanSurvey. Used by any script
or caller that wants "just give me a survey from these files" without
hand-rolling the .jsf/.sdf/.xtf branch itself.

- :func:`detect_format`, :func:`read_and_convert` -- raw ``.jsf``/``.sdf``/
  ``.xtf``.
- :func:`read_and_convert_segments` -- one raw file into one or more
  surveys, each with its output stem (XTF files are split at
  range-setting changes).
- :func:`find_sonar_files` -- the raw files in a directory.
- :func:`load_survey` -- a ``.ss.h5`` or a raw file, whichever it is.
- :func:`detect_catalog_format`, :func:`read_target_catalog` -- target
  catalogs, dispatched on the catalog directory's own contents.

2026 rshomberg@uri.edu
"""

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.core.target import Target
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.io.jsf import convert as jsf_convert
from sidescanpy.io.jsf import read_jsf
from sidescanpy.io.jsf.targets import read_discover_targets
from sidescanpy.io.sdf import convert as sdf_convert
from sidescanpy.io.sdf import read_sdf
from sidescanpy.io.sdf.split import split_by_sample_rate as split_by_sample_rate_sdf
from sidescanpy.io.sdf.targets import read_klein_targets
from sidescanpy.io.xtf import convert as xtf_convert
from sidescanpy.io.xtf import read_xtf, split_by_sample_rate
from sidescanpy.io.xtf.split import SPLIT_MIN_PINGS

logger = logging.getLogger(__name__)


def _as_paths(input_paths: str | os.PathLike | Iterable[str | os.PathLike]) -> list[Path]:
    """A path or an iterable of paths (str or path-like) as a list of Path."""
    if isinstance(input_paths, (str, os.PathLike)):
        return [Path(input_paths)]
    return [Path(p) for p in input_paths]


def detect_format(input_paths: str | os.PathLike | Iterable[str | os.PathLike]) -> str:
    """Detect sonar file format from extension; raise on mixed extensions.

    Extensions are compared case-insensitively (``.XTF`` is ``"xtf"``).

    Parameters
    ----------
    input_paths : path or list of paths
        Candidate sonar files (str or Path). All must share the same extension — this
        function is for the "merge these into one survey" case; batch
        callers processing files independently should call it once per
        file instead of once for a whole directory.

    Returns
    -------
    str
        One of: "jsf", "sdf", "xtf".

    Raises
    ------
    ValueError
        If extensions are mixed, or the (single) extension isn't
        supported.
    """
    extensions = {p.suffix.lower() for p in _as_paths(input_paths)}
    if len(extensions) > 1:
        raise ValueError(
            f"All input files must have the same format. "
            f"Got mixed extensions: {sorted(extensions)}"
        )
    ext = next(iter(extensions))
    if ext == ".jsf":
        return "jsf"
    if ext == ".sdf":
        return "sdf"
    if ext == ".xtf":
        return "xtf"
    raise ValueError(
        f"Unsupported format: '{ext}'. Supported: .jsf, .sdf, .xtf"
    )


def read_and_convert(
    input_paths: str | os.PathLike | Iterable[str | os.PathLike],
    fmt: str,
    survey_name: str | None = None,
    *,
    nav_crs=None,
) -> SideScanSurvey:
    """Read sonar files and convert to a SideScanSurvey.

    Parameters
    ----------
    input_paths : path or list of paths
        Paths (str or Path) to .jsf, .sdf or .xtf files (all same format);
        a single path is one file.
    fmt : str
        "jsf", "sdf" or "xtf".
    survey_name : str, optional
        Survey name. Defaults to first input file's stem.
    nav_crs : int, str or pyproj.CRS, optional
        XTF only: projected CRS of files with ``NavUnits`` 0 (see
        :func:`sidescanpy.io.xtf.convert`). Ignored for JSF and SDF.

    Returns
    -------
    SideScanSurvey
    """
    input_paths = _as_paths(input_paths)
    if fmt == "jsf":
        all_messages = []
        for p in input_paths:
            logger.debug(f"  Reading JSF: {p}")
            msgs = read_jsf(p)
            all_messages.extend(msgs)
        return jsf_convert(
            all_messages,
            source_files=input_paths,
            survey_name=survey_name or input_paths[0].stem,
        )

    if fmt == "sdf":
        all_pages = []
        for p in input_paths:
            logger.debug(f"  Reading SDF: {p}")
            pages = read_sdf(p)
            all_pages.extend(pages)
        return sdf_convert(
            all_pages,
            source_files=input_paths,
            survey_name=survey_name or input_paths[0].stem,
        )

    if fmt == "xtf":
        all_pings = []
        for p in input_paths:
            logger.debug(f"  Reading XTF: {p}")
            all_pings.append(read_xtf(p))
        return xtf_convert(
            all_pings,
            source_files=input_paths,
            survey_name=survey_name or input_paths[0].stem,
            nav_crs=nav_crs,
        )

    raise ValueError(f"Unknown format: {fmt}")


def segment_stem(stem: str, index: int, count: int) -> str:
    """Output stem of segment ``index`` (1-based) of ``count``.

    ``stem`` itself for a single segment, else ``<stem>_seg01``,
    ``<stem>_seg02``, ... (at least two digits).
    """
    if count == 1:
        return stem
    return f"{stem}_seg{index:0{max(2, len(str(count)))}d}"


def read_and_convert_segments(
    path: Path,
    *,
    survey_name: str | None = None,
    min_pings: int = SPLIT_MIN_PINGS,
    nav_crs=None,
) -> list[tuple[str, SideScanSurvey]]:
    """Read one raw sonar file into one survey per constant-range segment.

    A ``.jsf`` file gives one survey, exactly as :func:`read_and_convert`.
    ``.xtf`` and ``.sdf`` files are split at sample-rate (range-setting)
    changes (:func:`~sidescanpy.io.xtf.split_by_sample_rate`,
    :func:`~sidescanpy.io.sdf.split.split_by_sample_rate`) and each kept
    run is converted on its own, so every survey has the right
    ``sample_rate_hz``; runs shorter than ``min_pings`` are dropped (for
    XTF, counted in each survey's ``xtf_pings_dropped_short_runs``
    attribute).

    Parameters
    ----------
    path : Path
        One ``.jsf``, ``.sdf`` or ``.xtf`` file.
    survey_name : str, optional
        Base name; defaults to ``path.stem``.
    min_pings : int, default 50
        XTF and SDF: shortest run kept when a file is split.
    nav_crs : int, str or pyproj.CRS, optional
        XTF only: projected CRS of a file with ``NavUnits`` 0 (see
        :func:`sidescanpy.io.xtf.convert`). Ignored for JSF and SDF.

    Returns
    -------
    list[tuple[str, SideScanSurvey]]
        ``(output_stem, survey)`` per segment, in file order. The stem is
        the base name for a single segment, else ``<base>_seg01``,
        ``<base>_seg02``, ... (see :func:`segment_stem`); each survey is
        named after its stem.

    Raises
    ------
    ValueError
        Unsupported extension, or an XTF or SDF file whose runs are all shorter
        than ``min_pings``.
    """
    path = Path(path)
    fmt = detect_format([path])
    base = survey_name or path.stem
    if fmt == "sdf":
        logger.debug(f"  Reading SDF: {path}")
        runs = split_by_sample_rate_sdf(read_sdf(path), min_pings=min_pings, source=path.name)
        if not runs:
            raise ValueError(f"{path}: no run of constant sample rate is at least {min_pings} pings long")
        return [
            (stem := segment_stem(base, i, len(runs)), sdf_convert(run, source_files=[path], survey_name=stem))
            for i, run in enumerate(runs, 1)
        ]
    if fmt != "xtf":
        return [(base, read_and_convert([path], fmt, survey_name=base))]

    logger.debug(f"  Reading XTF: {path}")
    segments = split_by_sample_rate(read_xtf(path), min_pings=min_pings)
    if not segments:
        raise ValueError(
            f"{path}: no run of constant sample rate is at least {min_pings} pings long"
        )
    out = []
    for i, seg in enumerate(segments, 1):
        stem = segment_stem(base, i, len(segments))
        out.append((stem, xtf_convert(seg, source_files=[path], survey_name=stem, nav_crs=nav_crs)))
    return out


SONAR_FORMATS: tuple[str, ...] = ("jsf", "sdf", "xtf")
"""Raw formats :func:`read_and_convert` reads."""

CATALOG_JSF_DISCOVER = "jsf_discover"
CATALOG_SDF_KLEIN = "sdf_klein"

_HDF5_SUFFIXES = (".h5", ".hdf5")


def find_sonar_files(input_dir: Path | str, input_format: str = "auto") -> list[Path]:
    """Raw sonar files directly in ``input_dir`` (not recursive), sorted.

    Parameters
    ----------
    input_dir : path
        Directory to search.
    input_format : {"auto", "jsf", "sdf", "xtf"}, default "auto"
        ``"auto"`` returns ``*.jsf``, ``*.sdf`` and ``*.xtf`` files.
        ``.xtf`` matches in any letter case (``.XTF`` is common);
        ``.jsf`` and ``.sdf`` match lower case only.

    Raises
    ------
    ValueError
        Unknown ``input_format``.
    """
    input_dir = Path(input_dir)
    if input_format == "auto":
        formats = SONAR_FORMATS
    elif input_format in SONAR_FORMATS:
        formats = (input_format,)
    else:
        raise ValueError(f"Unknown input_format: {input_format}")
    found: set[Path] = set()
    for fmt in formats:
        if fmt == "xtf":
            found.update(
                p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xtf"
            )
        else:
            found.update(input_dir.glob(f"*.{fmt}"))
    return sorted(found)


def load_survey(path: Path | str, *, chunks: dict | None = None) -> SideScanSurvey:
    """Read ``path`` as a survey, whatever it is.

    A ``.ss.h5`` (or ``.h5``/``.hdf5``) file is read with
    :func:`~sidescanpy.io.hdf5.read_survey` (``chunks`` passed through);
    a raw ``.jsf``/``.sdf``/``.xtf`` file is converted in memory with
    :func:`read_and_convert` (``chunks`` ignored).
    """
    path = Path(path)
    if path.suffix.lower() in _HDF5_SUFFIXES or path.name.endswith(".ss.h5"):
        return read_survey(path, chunks=chunks)
    return read_and_convert([path], detect_format([path]))


def detect_catalog_format(source: Path | str) -> str:
    """Target-catalog format of the directory ``source``, from its own contents.

    Dispatches on the catalog files present, not on the paired survey's
    format (a JSF survey can be labelled in SonarWiz and vice versa).

    Returns
    -------
    str
        :data:`CATALOG_SDF_KLEIN` (``Target_*.tgt``) or
        :data:`CATALOG_JSF_DISCOVER` (``*.target``).

    Raises
    ------
    ValueError
        Both kinds present (ambiguous), or neither.
    """
    source = Path(source)
    has_klein = any(source.glob("Target_*.tgt"))
    has_jsf = any(source.glob("*.target"))
    if has_klein and has_jsf:
        raise ValueError(f"{source} contains both Klein (Target_*.tgt) and JSF (*.target) catalog files -- ambiguous")
    if has_klein:
        return CATALOG_SDF_KLEIN
    if has_jsf:
        return CATALOG_JSF_DISCOVER
    raise ValueError(f"No target catalog files found in {source} (expected *.target or Target_*.tgt)")


def read_target_catalog(source: Path | str) -> tuple[str, list[Target]]:
    """Read the target catalog in directory ``source``.

    Returns ``(format, targets)`` with ``format`` from
    :func:`detect_catalog_format`. Klein targets come back without a
    resolved position; see :mod:`sidescanpy.targets.correlate` for
    resolving them against ``.ss.h5`` files.
    """
    fmt = detect_catalog_format(source)
    if fmt == CATALOG_JSF_DISCOVER:
        return fmt, read_discover_targets(source)
    return fmt, read_klein_targets(source)
