"""
IO for SideScan sonar surveys.

Format-specific readers/converters live in submodules
(``sidescanpy.io.jsf``, ``sidescanpy.io.sdf``, ``sidescanpy.io.xtf``,
``sidescanpy.io.hdf5``,
``sidescanpy.io.netcdf``). This top-level package additionally exposes
format-dispatching helpers for callers that have file paths and don't
want to branch on extension themselves.

Public entry points
--------------------
- :func:`detect_format` — extension-based format dispatch (``.jsf`` /
  ``.sdf`` / ``.xtf``).
- :func:`read_and_convert` — read + convert a list of same-format
  sonar files into a validated :class:`~sidescanpy.core.survey.SideScanSurvey`.
- :func:`find_sonar_files` — raw sonar files in a directory.
- :func:`load_survey` — a ``.ss.h5`` or raw file, whichever it is.
- :func:`detect_catalog_format` / :func:`read_target_catalog` — target
  catalogs (EdgeTech Discover or Klein SonarWiz), by directory contents.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.dispatch import (
    CATALOG_JSF_DISCOVER,
    CATALOG_SDF_KLEIN,
    detect_catalog_format,
    detect_format,
    find_sonar_files,
    load_survey,
    read_and_convert,
    read_target_catalog,
)

__all__ = [
    "CATALOG_JSF_DISCOVER",
    "CATALOG_SDF_KLEIN",
    "detect_catalog_format",
    "detect_format",
    "find_sonar_files",
    "load_survey",
    "read_and_convert",
    "read_target_catalog",
]
