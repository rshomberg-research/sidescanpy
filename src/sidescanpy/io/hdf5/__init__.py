"""HDF5 I/O for sidescan surveys."""

from .reader import from_hdf5
from .writer import to_hdf5

__all__ = ['from_hdf5', 'to_hdf5']