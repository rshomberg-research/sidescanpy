"""
Write a SideScanSurvey to an HDF5 (.ss.h5) file.

File layout::

    /                                ← root attrs: survey_name, source_format,
    │                                              source_files_json,
    │                                              sidescanpy_version
    /frequencies/
        hz_<freq>/                   ← per SideScanFrequencyChannel
        ...
    /aux_streams/
        navigation/                  ← always written (empty if unset)
        attitude/                    ← always written (empty if unset)
        dvl/                         ← always written (empty if unset)
        sensors/                     ← only if survey has at least one sensor
            <sensor_type>/

One encoding policy (``_encoding_for``) applies to every frequency
data variable with ndim >= 2: gzip level 4 + shuffle, chunked along ping.
float64 variables are stored as float32 except those in
:data:`FLOAT64_KEEP` (lat/lon need ~1e-7 deg precision). Aux records are
stored uncompressed.

2026 rshomberg@uri.edu
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.schema import (
    HDF5_CHUNK_PING,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_DVL,
    HDF5_GROUP_FREQUENCIES,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_SENSORS,
    SS_FORMAT_NAME,
    SS_FORMAT_VERSION,
    make_empty_attitude_record,
    make_empty_dvl_record,
    make_empty_navigation_record,
)
from sidescanpy.core.survey import SideScanSurvey

logger = logging.getLogger(__name__)


_HDF5_ENGINE = "h5netcdf"
FLOAT64_KEEP = frozenset({
    "latitude", "longitude", "latitude_per_sample", "longitude_per_sample",
})
"""float64 variables that must NOT be downcast to float32 on write
(float32 gives only ~0.4 m at geographic coordinates)."""


def write_survey(
    survey: SideScanSurvey,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """
    Write a SideScanSurvey to a .ss.h5 file.

    Parameters
    ----------
    survey : SideScanSurvey
        The survey to serialize. All frequency channels and aux records must
        already pass their accessor's validate().
    path : Path or str
        Destination file path. By convention, use the .ss.h5 extension.
    overwrite : bool, default False
        If False (default) and ``path`` exists, raise FileExistsError.
        If True, replace the existing file.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    """
    p = Path(path)

    if p.exists():
        if not overwrite:
            raise FileExistsError(
                f"HDF5 file already exists: {p}. Pass overwrite=True to replace."
            )
        p.unlink()

    # 1. Pre-create file with root attrs (low-level h5netcdf for direct attr access).
    _write_root_attrs(p, survey)

    # 2. Write each frequency channel as a group, with backscatter compression.
    for freq_key, channel in survey.freq_channels.items():
        encoding = _build_frequency_encoding(channel.dataset)
        group = f"{HDF5_GROUP_FREQUENCIES}/{freq_key}"
        channel.dataset.to_netcdf(
            path=p,
            group=group,
            mode="a",
            engine=_HDF5_ENGINE,
            encoding=encoding,
        )

    # 3. Write aux records (always nav, attitude, dvl, even if empty).
    nav_ds = survey.navigation if survey.navigation is not None \
        else make_empty_navigation_record()
    nav_ds.to_netcdf(path=p, group=HDF5_GROUP_NAVIGATION,
                     mode="a", engine=_HDF5_ENGINE)

    att_ds = survey.attitude if survey.attitude is not None \
        else make_empty_attitude_record()
    att_ds.to_netcdf(path=p, group=HDF5_GROUP_ATTITUDE,
                     mode="a", engine=_HDF5_ENGINE)

    dvl_ds = survey.dvl if survey.dvl is not None \
        else make_empty_dvl_record()
    dvl_ds.to_netcdf(path=p, group=HDF5_GROUP_DVL,
                     mode="a", engine=_HDF5_ENGINE)

    # 4. Write sensor records (parent group exists only if any sensors present).
    for sensor_type, sensor_ds in survey.sensors.items():
        group = f"{HDF5_GROUP_SENSORS}/{sensor_type}"
        sensor_ds.to_netcdf(path=p, group=group,
                            mode="a", engine=_HDF5_ENGINE)


# ============================================================
# HELPERS
# ============================================================

def _write_root_attrs(path: Path, survey: SideScanSurvey) -> None:
    """Create the file (mode='w') and set root-level survey metadata.

    ``format_name`` and ``format_version`` identify the on-disk layout
    (:data:`~sidescanpy.core.schema.SS_FORMAT_VERSION`);
    ``sidescanpy_version`` records the writer and ``date_created`` when
    the file was written (ISO 8601, UTC).
    """
    with h5netcdf.File(str(path), "w") as f:
        f.attrs["format_name"]       = SS_FORMAT_NAME
        f.attrs["format_version"]    = SS_FORMAT_VERSION
        f.attrs["sidescanpy_version"] = __version__
        f.attrs["date_created"]      = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        f.attrs["survey_name"]       = survey.survey_name
        f.attrs["source_format"]     = survey.source_format
        f.attrs["source_files_json"] = json.dumps(
            [str(sf) for sf in survey.source_files]
        )


def _encoding_for(var: str, da: xr.DataArray) -> dict | None:
    """The single write-encoding policy for one data variable.

    Returns ``None`` for variables left at xarray defaults (ndim < 2 or
    empty). Otherwise gzip + shuffle, chunked along ``ping`` (clipped to
    the array), float64 downcast to float32 unless listed in
    :data:`FLOAT64_KEEP`. ``_FillValue`` is left to xarray (NaN for floats).
    """
    if da.ndim < 2 or da.size == 0:
        return None
    chunks = tuple(
        min(HDF5_CHUNK_PING, n) if dim == "ping" else n
        for dim, n in zip(da.dims, da.shape, strict=True)
    )
    enc: dict = {
        "compression":      HDF5_COMPRESSION,
        "compression_opts": HDF5_COMPRESSION_LEVEL,
        "shuffle":          True,
        "chunksizes":       chunks,
    }
    if da.dtype == np.float64 and var not in FLOAT64_KEEP:
        enc["dtype"] = "float32"
    return enc


def _build_frequency_encoding(ds: xr.Dataset) -> dict[str, dict]:
    """Encoding dict for a frequency-channel dataset (see ``_encoding_for``)."""
    encoding: dict[str, dict] = {}
    for name, da in ds.data_vars.items():
        enc = _encoding_for(str(name), da)
        if enc is not None:
            encoding[str(name)] = enc
    return encoding
