"""
JSF (Xylem Jet Stream Fish) I/O module.

Provides utilities for reading JSF binary files, archiving to intermediate
HDF5 format, and converting to standard sonar schema for analysis.

Main workflow:
1. read_jsf() - Parse JSF binary → Messages
2. jsf_to_native_hdf5() - Archive Messages → Native HDF5
3. native_hdf5_to_standard() - Convert → Standard schema

2026 rshomberg@uri.edu
"""

from sidescanpy.io.jsf.reader import (
    read_jsf,
    read_jsf_iter,
    jsf_to_native_hdf5,
    save_temp_file,
    load_temp_file,
)

from sidescanpy.io.jsf.converter import (
    native_hdf5_to_standard,
    list_frequencies_in_native_hdf5,
)

__all__ = [
    # Reader API
    "read_jsf",
    "read_jsf_iter",
    "jsf_to_native_hdf5",
    "save_temp_file",
    "load_temp_file",
    # Converter API
    "native_hdf5_to_standard",
    "list_frequencies_in_native_hdf5",
]