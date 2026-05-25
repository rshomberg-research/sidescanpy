"""Standard HDF5 I/O for .ss.h5 files"""

from sidescanpy.io.standard.reader import read_ss_hdf5
from sidescanpy.io.standard.writer import write_ss_hdf5

__all__ = ["read_ss_hdf5", "write_ss_hdf5"]