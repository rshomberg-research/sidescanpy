"""
Write frequency channel datasets to standard .ss.h5 HDF5 files

2026 rshomberg@uri.edu
"""

import logging 
from pathlib import Path 
from typing import Optional 

import xarray as xr

from sidescanpy.core.schema import (
    HDF5_GROUP_FREQUENCIES,
    HDF5_FREQ_PREFIX,
    HDF5_CHUNK_PING,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,
)

logger = logging.getLogger(__name__)


def write_ss_hdf5(
        ds: xr.Dataset,
        output_path : Path,
        freq_hz     : Optional[int] = None, 
        overwrite   : bool = False,
) -> None:
    """
    Write frequency channel dataset to standard .ss.h5 HDF5 file.

    Validates dataset before writing. Applies chunking and compression for dask
    compatibility and efficient storage.

    Parameters
    ----------
    ds : xr.Dataset
        Validated frequency channel dataset.
    output_path : Path
        Output HDF5 file path.
    freq_hz : int, optional
        Frequency in Hz. If None, extracted from ds.attrs["freq_hz_start"].
    overwrite : bool
        If False (default), raise if file exists. If True, overwrite.

    Raises
    ------
    ValueError
        If dataset fails schema validation.
    FileExistsError
        If file exists and overwrite=False.

    Examples
    --------
    >>> ds.sidescan.validate()
    >>> write_ss_hdf5(ds, "survey_001.ss.h5")
    """

    # Validate dataset
    logger.debug(f"Validating dataset before writing to {output_path}")
    ds.sidescan.validate() 

    # Check if file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {output_path}. Use overwrite=True to replace")
    
    # Determine frequency
    if freq_hz is None:
        freq_hz = int(ds.attrs["freq_hz_start"])
    freq_key = f"{HDF5_FREQ_PREFIX}{freq_hz}"
    # FIXME: I think I want the input option to be freq_key

    logger.info(f"Writing dataset to {output_path} in group 'frequencies/{freq_key}")

    # Prepare encoding: chunk along ping dimension, apply compression
    n_samples = ds.sizes["sample"]
    encoding = {}
    for var in ds.data_vars:
        if var in ["backscatter_raw", "backscatter_intensity"]:
            encoding[var] = {
                "chunksizes": (HDF5_CHUNK_PING, 2, n_samples),
                "compression": HDF5_COMPRESSION,
                "compression_opts": HDF5_COMPRESSION_LEVEL,
            }

    logger.debug(f"Encoding configuration: {encoding}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to HDF5 using h5netcdf engine
    group = f"{HDF5_GROUP_FREQUENCIES}/{freq_key}"
    mode = 'w' if not output_path.exists() else 'a'

    ds.to_netcdf(
        output_path,
        mode=mode,
        engine='h5netcdf',
        group=group,
        encoding=encoding
    )

    logger.info(f"Successfully wrote {output_path}")