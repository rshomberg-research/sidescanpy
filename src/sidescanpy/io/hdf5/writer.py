"""
Write a SideScanSurvey to an HDF5 (.ss.h5) file.

File layout:

    /                                ← root attrs: survey_name, source_format,
    │                                              source_files_json
    /frequencies/
        hz_<freq>/                   ← per SideScanFrequencyChannel
        ...
    /aux_streams/
        navigation/                  ← always written (empty if unset)
        attitude/                    ← always written (empty if unset)
        dvl/                         ← always written (empty if unset)
        sensors/                     ← only if survey has at least one sensor
            <sensor_type>/

Compression (gzip level 4) is applied to backscatter_raw and
backscatter_intensity only. Aux records are stored uncompressed.

2026 rshomberg@uri.edu
"""

import json
import logging
from pathlib import Path

import h5netcdf

from sidescanpy.core.schema import (
    HDF5_CHUNK_PING,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_DVL,
    HDF5_GROUP_FREQUENCIES,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_SENSORS,
    make_empty_attitude_record,
    make_empty_dvl_record,
    make_empty_navigation_record,
)
from sidescanpy.core.survey import SideScanSurvey

logger = logging.getLogger(__name__)


_HDF5_ENGINE = "h5netcdf"
_COMPRESSED_VARS = ("backscatter_raw", "backscatter_intensity")


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
        encoding = _build_frequency_encoding(
            n_pings=channel.n_pings,
            n_channels=channel.n_channels,
            n_samples=channel.n_samples,
        )
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
    """Create the file (mode='w') and set root-level survey metadata."""
    with h5netcdf.File(str(path), "w") as f:
        f.attrs["survey_name"]       = survey.survey_name
        f.attrs["source_format"]     = survey.source_format
        f.attrs["source_files_json"] = json.dumps(
            [str(sf) for sf in survey.source_files]
        )


def _build_frequency_encoding(
    *,
    n_pings: int,
    n_channels: int,
    n_samples: int,
) -> dict[str, dict]:
    """Encoding dict for backscatter compression. Aux records get no encoding."""
    chunk_ping = min(HDF5_CHUNK_PING, n_pings) if n_pings > 0 else 1
    encoding: dict[str, dict] = {}
    for var in _COMPRESSED_VARS:
        encoding[var] = {
            "compression":      HDF5_COMPRESSION,
            "compression_opts": HDF5_COMPRESSION_LEVEL,
            "chunksizes":       (chunk_ping, n_channels, n_samples),
        }
    return encoding
