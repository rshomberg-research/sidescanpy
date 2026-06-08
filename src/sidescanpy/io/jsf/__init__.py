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
    )



__all__ = [
    # Reader API
    "read_jsf",
    "read_jsf_iter",
    # Converter API
]


    