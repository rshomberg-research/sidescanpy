"""Standard HDF5 I/O for .ss.h5 files"""


from sidescanpy.io.standard.reader import read_ss_hdf5
from sidescanpy.io.standard.writer import write_ss_hdf5
from sidescanpy.io.standard.aux_streams import (
    write_navigation_stream,
    read_navigation_stream,
    write_attitude_stream,
    read_attitude_stream,
    write_environmental_stream,
    read_environmental_stream,
    interpolate_to_ping_time,
)

__all__ = [
    "read_ss_hdf5",
    "write_ss_hdf5",
    "write_navigation_stream",
    "read_navigation_stream",
    "write_attitude_stream",
    "read_attitude_stream",
    "write_environmental_stream",
    "read_environmental_stream",
    "interpolate_to_ping_time",
]