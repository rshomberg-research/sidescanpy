"""
HDF5 IO for full SideScanSurvey objects.

This module reads and writes complete :class:`SideScanSurvey` objects —
multiple frequency channels plus optional navigation, attitude, DVL, and
sensor auxiliary records — as a single `.ss.h5` file with a grouped
layout.

For single-frequency channels in standalone NetCDF, see
:mod:`sidescanpy.io.netcdf`.

Public entry points
-------------------
- :func:`read_survey` — read a `.ss.h5` file back into a validated
  survey. Accepts ``chunks=`` for lazy/dask-backed loading of frequency
  channel data.
- :func:`write_survey` — write a survey to disk with grouped layout
  and per-variable encoding.

File layout
-----------
::

    /                           (root: survey_name, source_format,
    │                            source_files_json)
    /frequencies/
        hz_<freq>/              (per SideScanFrequencyChannel)
        ...
    /aux_streams/
        navigation/             (always written, may be empty)
        attitude/
        dvl/
        sensors/
            <sensor_type>/      (only if any sensors present)

Conventions
-----------
- Engine: ``h5netcdf``.
- Aux records always written, even when empty (zero ``obs``). The
  reader returns them as empty-but-valid datasets;
  ``has_navigation``/``has_attitude``/``has_dvl`` correctly report
  ``False`` in that case.
- Compression: gzip level 4 on backscatter only. Aux records are tiny
  and stored uncompressed.
- ``chunks=`` applies only to frequency channels. Aux records always
  load eagerly regardless.

See ``docs/io_hdf5.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.hdf5.reader import read_survey
from sidescanpy.io.hdf5.writer import write_survey

__all__ = ["read_survey", "write_survey"]