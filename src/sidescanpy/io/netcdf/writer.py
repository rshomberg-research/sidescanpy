"""
Write a SideScanFrequencyChannel to a NetCDF (.nc) file.

Uses h5netcdf engine. Encoding follows the shared policy in
``io.hdf5.writer._encoding_for`` (gzip + shuffle on every ndim >= 2
variable; small per-ping scalars are stored uncompressed).

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.core.survey import SideScanFrequencyChannel
from sidescanpy.io.hdf5.writer import _build_frequency_encoding

logger = logging.getLogger(__name__)


_NETCDF_ENGINE = "h5netcdf"


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
    """Encoding for the channel dataset: the same policy as the .ss.h5 writer."""
    return _build_frequency_encoding(channel.dataset)
