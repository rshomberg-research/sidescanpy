"""
Read frequency channel datasets from standard .ss.h5 HDF5 files

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional

import xarray as xr

from sidescanpy.core.schema import (
    HDF5_GROUP_FREQUENCIES,
    HDF5_FREQ_PREFIX,
)

logger = logging.getLogger(__name__)


def read_ss_hdf5(
    file_path: Path,
    freq_hz: int,
    chunks: Optional[dict[str, int]] = None,    # ???: is that dict type hint accurate
) -> xr.Dataset:
    """
    Read frequency channel dataset from standard .ss.h5 HDF5 file.

    Validates loaded dataset. Optionally opens with dask for lazy loading.

    Parameters
    ----------
    file_path : Path
        Input HDF5 file path.
    freq_hz : int
        Frequency in Hz to read.
    chunks : dict, optional
        Dask chunk specification, e.g., {"ping": 256}. If provided, dataset
        is opened lazily with dask.

    Returns
    -------
    xr.Dataset
        Validated frequency channel dataset, optionally chunked with dask.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If frequency group not found or dataset fails schema validation.

    Examples
    --------
    >>> ds = read_ss_hdf5("survey_001.ss.h5", freq_hz=1495000)
    >>> ds.sidescan.validate()
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")    
    
    freq_key = f"{HDF5_FREQ_PREFIX}{freq_hz}"   # TODO: just pull using freq_key as input
    group = f"{HDF5_GROUP_FREQUENCIES}/{freq_key}"

    logger.info(f"Reading {group} from {file_path}")

    # Open dataset w/ optional dask chunks
    ds = xr.open_dataset(
        file_path, 
        engine='h5netcdf',
        group=group,
        chunks=chunks,
    )

    logger.debug(f"Validating loaded dataset")
    ds.sidescan.validate()

    logger.info(f"Successfully loaded {file_path} frequency {freq_hz}")

    return ds