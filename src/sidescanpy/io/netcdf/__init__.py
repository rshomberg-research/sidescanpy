"""NetCDF IO for individual SideScanFrequencyChannel datasets."""

from sidescanpy.io.netcdf.reader import read_frequency_channel
from sidescanpy.io.netcdf.writer import write_frequency_channel

__all__ = ["read_frequency_channel", "write_frequency_channel"]