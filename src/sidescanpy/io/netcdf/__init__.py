"""netCDF I/O for sidescan surveys."""

from .reader import from_netcdf
from .writer import to_netcdf

__all__ = ['from_netcdf', 'to_netcdf']