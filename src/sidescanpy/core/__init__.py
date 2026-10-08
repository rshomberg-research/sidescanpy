"""
Core schema, validation, and survey container types.

This module defines the contract that every dataset in SideScanPy must
satisfy and provides the top-level objects that user code interacts with.

Public types
------------
- :class:`SideScanFrequencyChannel` — single-frequency wrapper around a
  validated ``xr.Dataset``.
- :class:`SideScanSurvey` — multi-frequency container with optional
  navigation, attitude, DVL, and sensor auxiliary records.
- :class:`Target` — format-agnostic sonar contact (see
  :mod:`sidescanpy.core.target`).

Validation is provided through xarray accessors:

- ``ds.sidescan`` — validates a single-frequency sonar dataset.
- ``ds.navigation_record``, ``ds.attitude_record``, ``ds.dvl_record``,
  ``ds.sensor_record`` — validate the corresponding aux stream datasets.

Conventions
-----------
- All time coordinates are ``datetime64[ns]``.
- The ``channel`` dimension is always size 2 (port + starboard).
- ``ds.attrs["processing_history"]`` is a JSON-encoded list of step records
  (see :mod:`sidescanpy.core.history`); the processing level is derived from
  it by :func:`processing_level`.
- Empty-but-valid aux records (zero observations) are first-class — they
  validate cleanly and report ``is_empty() == True``.

See ``docs/core.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.core.history import (
    BUILTIN_STEPS,
    PROCESSING_LEVELS,
    RADIOMETRIC_STEPS,
    applied_steps,
    has_step,
    processing_level,
    read_history,
)
from sidescanpy.core.schema import (
    HDF5_CHUNK_PING,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,
    HDF5_FREQ_PREFIX,
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_AUX_STREAMS,
    HDF5_GROUP_DVL,
    HDF5_GROUP_FREQUENCIES,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_SENSORS,
    KNOWN_PROCESSING_LEVELS,
    REQUIRED_ATTRS,
    REQUIRED_COORDS,
    # Schema constants (exposed for IO modules and external users).
    REQUIRED_DIMS,
    REQUIRED_VARS,
    AttitudeRecordAccessor,
    DVLRecordAccessor,
    NavigationRecordAccessor,
    SensorRecordAccessor,
    # Accessors — registered for side effects on import.
    SideScanAccessor,
    make_empty_attitude_record,
    make_empty_dvl_record,
    # Empty-record builders.
    make_empty_navigation_record,
    make_empty_sensor_record,
)
from sidescanpy.core.survey import (
    SideScanFrequencyChannel,
    SideScanSurvey,
)
from sidescanpy.core.target import (
    CONFIDENCE_RANK,
    PositionConfidence,
    Target,
    bbox_from_center,
    clip_bbox,
)

__all__ = [
    "BUILTIN_STEPS",
    "PROCESSING_LEVELS",
    "RADIOMETRIC_STEPS",
    "applied_steps",
    "has_step",
    "processing_level",
    "read_history",
    "HDF5_CHUNK_PING",
    "HDF5_COMPRESSION",
    "HDF5_COMPRESSION_LEVEL",
    "HDF5_FREQ_PREFIX",
    "HDF5_GROUP_ATTITUDE",
    "HDF5_GROUP_AUX_STREAMS",
    "HDF5_GROUP_DVL",
    "HDF5_GROUP_FREQUENCIES",
    "HDF5_GROUP_NAVIGATION",
    "HDF5_GROUP_SENSORS",
    "KNOWN_PROCESSING_LEVELS",
    "REQUIRED_ATTRS",
    "REQUIRED_COORDS",
    "REQUIRED_DIMS",
    "REQUIRED_VARS",
    "AttitudeRecordAccessor",
    "DVLRecordAccessor",
    "NavigationRecordAccessor",
    "SensorRecordAccessor",
    "SideScanAccessor",
    "SideScanFrequencyChannel",
    "SideScanSurvey",
    "make_empty_navigation_record",
    "make_empty_attitude_record",
    "make_empty_dvl_record",
    "make_empty_sensor_record",
    "CONFIDENCE_RANK",
    "PositionConfidence",
    "Target",
    "bbox_from_center",
    "clip_bbox",
]
