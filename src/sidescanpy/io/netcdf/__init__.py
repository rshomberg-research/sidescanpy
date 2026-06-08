"""
NetCDF IO for single-frequency channels.

This module reads and writes individual :class:`SideScanFrequencyChannel`
datasets to standalone NetCDF (``.nc``) files. For full multi-frequency
surveys with auxiliary records, use :mod:`sidescanpy.io.hdf5` instead.

Public entry points
-------------------
- :func:`read_frequency_channel` — read a NetCDF file back into a
  validated channel. Accepts ``chunks=`` for lazy/dask-backed loading.
- :func:`write_frequency_channel` — write a channel to disk with
  per-variable encoding (gzip on backscatter, raw on per-ping scalars).

Conventions
-----------
- Engine: ``h5netcdf``.
- Compression: gzip level 4 on ``backscatter_raw`` and
  ``backscatter_intensity``. Per-ping scalars uncompressed.
- Chunking: ``HDF5_CHUNK_PING`` along the ping axis on disk.
  Override on read via ``chunks={"ping": N}``.
- Round-trip is bit-exact for variable values, attributes, and
  ``processing_history``.

See ``docs/io_netcdf.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.netcdf.reader import read_frequency_channel
from sidescanpy.io.netcdf.writer import write_frequency_channel

__all__ = ["read_frequency_channel", "write_frequency_channel"]