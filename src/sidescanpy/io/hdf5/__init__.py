"""HDF5 IO for SideScanSurvey objects (multi-frequency + aux records)."""

from sidescanpy.io.hdf5.reader import read_survey
from sidescanpy.io.hdf5.writer import write_survey

__all__ = ["read_survey", "write_survey"]