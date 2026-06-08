"""
Write a SideScanFrequencyChannel to a NetCDF (.nc) file.

Uses h5netcdf engine. Compression (gzip level 4) is applied to the large
backscatter arrays only; small per-ping scalars are stored uncompressed.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.core.schema import (
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,
    HDF5_CHUNK_PING,
)
from sidescanpy.core.survey import SideScanFrequencyChannel

logger = logging.getLogger(__name__)


_NETCDF_ENGINE = "h5netcdf"
_COMPRESSED_VARS = ("backscatter_raw", "backscatter_intensity")


def write_frequency_channel(
    channel: SideScanFrequencyChannel,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """
    Write a single SideScanFrequencyChannel to a .nc file.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        The channel to serialize. Its underlying dataset must already pass
        SideScanAccessor.validate().
    path : Path or str
        Destination file path.
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
                f"NetCDF file already exists: {p}. Pass overwrite=True to replace."
            )
        p.unlink()

    encoding = _build_encoding(channel)

    channel.dataset.to_netcdf(
        path=p,
        engine=_NETCDF_ENGINE,
        encoding=encoding,
    )


def _build_encoding(channel: SideScanFrequencyChannel) -> dict[str, dict]:
    """Compression and chunking encoding for backscatter variables."""
    n_pings    = channel.n_pings
    n_channels = channel.n_channels
    n_samples  = channel.n_samples

    chunk_ping = min(HDF5_CHUNK_PING, n_pings) if n_pings > 0 else 1

    encoding: dict[str, dict] = {}
    for var in _COMPRESSED_VARS:
        if var in channel.dataset.data_vars:
            encoding[var] = {
                "compression":       HDF5_COMPRESSION,
                "compression_opts":  HDF5_COMPRESSION_LEVEL,
                "chunksizes":        (chunk_ping, n_channels, n_samples),
            }
    return encoding