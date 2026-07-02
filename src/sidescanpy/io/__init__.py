"""
IO for SideScan sonar surveys.

Format-specific readers/converters live in submodules
(``sidescanpy.io.jsf``, ``sidescanpy.io.sdf``, ``sidescanpy.io.hdf5``,
``sidescanpy.io.netcdf``). This top-level package additionally exposes
format-dispatching helpers for callers that have file paths and don't
want to branch on extension themselves.

Public entry points
--------------------
- :func:`detect_format` — extension-based format dispatch (``.jsf`` /
  ``.sdf``).
- :func:`read_and_convert` — read + convert a list of same-format
  sonar files into a validated :class:`~sidescanpy.core.survey.SideScanSurvey`.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.dispatch import detect_format, read_and_convert

__all__ = [
    "detect_format",
    "read_and_convert",
]
