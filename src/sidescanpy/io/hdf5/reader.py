"""
Read a SideScanSurvey from an HDF5 (.ss.h5) file.

Pass ``chunks=`` for lazy/dask-backed loading of frequency channel data.
Aux records are always loaded eagerly (they're small).

2026 rshomberg@uri.edu
"""

import json
import logging
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

from sidescanpy.core.schema import (
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_DVL,
    HDF5_GROUP_FREQUENCIES,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_SENSORS,
    SS_FORMAT_VERSION,
)
from sidescanpy.core.survey import SideScanSurvey

logger = logging.getLogger(__name__)


_HDF5_ENGINE = "h5netcdf"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(version).split("."))


def check_format_version(root_attrs: dict, path: Path | str = "") -> str | None:
    """Check a ``.ss.h5`` file's ``format_version`` root attribute.

    Parameters
    ----------
    root_attrs : dict
        The file's root attributes.
    path : Path or str, optional
        Used in log messages only.

    Returns
    -------
    str or None
        The file's format version, or None for a file written before the
        attribute existed (sidescanpy 0.8 and earlier), which is logged at
        DEBUG and read as is.

    Notes
    -----
    A file whose version is newer than this sidescanpy knows
    (:data:`~sidescanpy.core.schema.SS_FORMAT_VERSION`) is read anyway and
    logs a WARNING, because a newer layout may hold variables this reader
    does not understand.
    """
    version = root_attrs.get("format_version")
    if version is None:
        logger.debug("%s: no format_version attribute (written by sidescanpy 0.8 or earlier)", path)
        return None
    version = str(version)
    try:
        newer = _version_tuple(version) > _version_tuple(SS_FORMAT_VERSION)
    except ValueError:
        logger.warning("%s: unrecognised format_version %r", path, version)
        return version
    if newer:
        logger.warning(
            "%s: format_version %s is newer than this sidescanpy reads (%s); "
            "upgrade sidescanpy if anything is missing",
            path, version, SS_FORMAT_VERSION,
        )
    return version


def read_survey(
    path: Path | str,
    *,
    chunks: dict[str, int] | None = None,
) -> SideScanSurvey:
    """
    Read a .ss.h5 file back into a SideScanSurvey.

    Parameters
    ----------
    path : Path or str
        Source file path.
    chunks : dict, optional
        Pass to ``xr.open_dataset`` for lazy loading of frequency channel data
        via dask. e.g. ``chunks={"ping": 256}``. None (default) loads eagerly.
        Aux records are always loaded eagerly regardless of this parameter.

    Returns
    -------
    SideScanSurvey
        Validated survey, fully populated from on-disk groups.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any loaded dataset fails its accessor's schema validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HDF5 file not found: {p}")

    # 1. Inspect file structure and read root attrs.
    with h5netcdf.File(str(p), "r") as f:
        root_attrs = dict(f.attrs)
        freq_keys = sorted(f[HDF5_GROUP_FREQUENCIES].groups) \
            if HDF5_GROUP_FREQUENCIES in f.groups else []

        sensor_types: list[str] = []
        aux_groups = f.get("aux_streams")
        if aux_groups is not None and "sensors" in aux_groups.groups:
            sensor_types = sorted(aux_groups["sensors"].groups)

    check_format_version(root_attrs, p)

    # 2. Construct survey shell from root attrs.
    survey_name   = str(root_attrs.get("survey_name", p.stem))
    source_format = str(root_attrs.get("source_format", "unknown"))
    source_files  = [
        Path(s) for s in json.loads(root_attrs.get("source_files_json", "[]"))
    ]

    survey = SideScanSurvey(
        source_files=source_files,
        source_format=source_format,
        survey_name=survey_name,
    )

    # 3. Load each frequency channel (chunked if requested).
    for fk in freq_keys:
        ds = xr.open_dataset(
            p,
            group=f"{HDF5_GROUP_FREQUENCIES}/{fk}",
            engine=_HDF5_ENGINE,
            chunks=chunks,
        )
        ds = _normalize_string_coords(ds)
        survey.add_frequency(fk, ds)

    # 4. Load aux records (always eager).
    survey.navigation = _read_aux(p, HDF5_GROUP_NAVIGATION)
    survey.attitude   = _read_aux(p, HDF5_GROUP_ATTITUDE)
    survey.dvl        = _read_aux(p, HDF5_GROUP_DVL)

    # 5. Load sensor records.
    for sensor_type in sensor_types:
        ds = xr.open_dataset(
            p,
            group=f"{HDF5_GROUP_SENSORS}/{sensor_type}",
            engine=_HDF5_ENGINE,
        )
        survey.add_sensor(sensor_type, ds)

    return survey


# ============================================================
# HELPERS
# ============================================================

def _read_aux(path: Path, group: str) -> xr.Dataset:
    """Read a single aux-record group eagerly. Returns the dataset for
    assignment to the survey's setter (which handles None, validation, etc.)."""
    return xr.open_dataset(
        path,
        group=group,
        engine=_HDF5_ENGINE,
    )


def _normalize_string_coords(ds: xr.Dataset) -> xr.Dataset:
    """Coerce byte-string coords back to native unicode strings.

    h5netcdf can return string coords as bytes (``<S12``) depending on how
    they were written. Downstream code does ``.sel(channel_id="port")`` and
    expects plain Python str.
    """
    for name in ("channel_id",):
        if name not in ds.coords:
            continue
        values = ds.coords[name].values
        if values.dtype.kind == "S":  # byte strings
            decoded = np.array([v.decode("utf-8") for v in values])
            ds = ds.assign_coords({name: (ds.coords[name].dims, decoded)})
    return ds
