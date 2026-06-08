"""
Read a SideScanFrequencyChannel from a NetCDF (.nc) file.

Uses h5netcdf engine. Pass ``chunks=`` for lazy/dask-backed loading.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanFrequencyChannel

logger = logging.getLogger(__name__)


_NETCDF_ENGINE = "h5netcdf"


def read_frequency_channel(
    path: Path | str,
    *,
    chunks: dict[str, int] | None = None,
) -> SideScanFrequencyChannel:
    """
    Read a NetCDF file back into a SideScanFrequencyChannel.

    Parameters
    ----------
    path : Path or str
        Source file path.
    chunks : dict, optional
        Pass to ``xr.open_dataset`` for lazy loading via dask. e.g.
        ``chunks={"ping": 256}``. None (default) loads eagerly.

    Returns
    -------
    SideScanFrequencyChannel
        Validated channel wrapper around the loaded dataset.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the loaded dataset fails schema validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NetCDF file not found: {p}")

    ds = xr.open_dataset(
        p,
        engine=_NETCDF_ENGINE,
        chunks=chunks,
    )

    ds = _normalize_string_coords(ds)
    return SideScanFrequencyChannel(ds)


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
        if values.dtype.kind == "S":   # byte strings
            decoded = np.array([v.decode("utf-8") for v in values])
            ds = ds.assign_coords({name: (ds.coords[name].dims, decoded)})
    return ds