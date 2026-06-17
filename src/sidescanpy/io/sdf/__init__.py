"""
Klein Marine Systems SDF (Sonar Data File) reader.

Parses Edgetech-style SDF files: a sequence of fixed-format pages, each
with a 4-byte sync marker followed by a 512-byte header and variable-
length sonar data.

This module provides:

- :class:`SDFSonarPage` — typed dataclass for sonar data pages
  (versions 3501 and 3502; low- and high-frequency channels).
- :func:`parse_page_header` — parse 516 bytes of header into a typed
  page, dispatching by page version.
- :func:`read_sdf` / :func:`read_sdf_iter` — read pages from a file.
- :func:`convert` — convert pages into a SideScanSurvey.

Klein conventions captured here:

- Position fields (latitude/longitude) are stored as float64 in
  *radians*; this module converts to degrees on parse.
- Attitude fields (heading/pitch/roll) are stored as float32 in
  *degrees*; no unit conversion needed.
- Page sizes in the header *exclude* the 4-byte marker; total page
  on disk is page_size_field + 4.
- Year is normally 4-digit (e.g. 2026); some firmware stores
  offset-from-1900. Auto-detected via < 200 check.

See ``docs/io_sdf.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.sdf.pages import (
    SDFPageHeader,
    SDFSonarPage,
    SDFUnknownPage,
    MARKER,
    MARKER_BYTES,
    PAGE_HEADER_BYTES,
    HEADER_TOTAL_BYTES,
    PAGE_VERSIONS_SONAR,
)
from sidescanpy.io.sdf.reader import (
    read_sdf,
    read_sdf_iter,
    read_sdf_header,
    SDFPage,
)

# converter (still uses old API; will need updating)
from sidescanpy.io.sdf.converter import convert

__all__ = [
    "SDFPageHeader",
    "SDFSonarPage",
    "SDFUnknownPage",
    "SDFPage",
    "read_sdf",
    "read_sdf_iter",
    "read_sdf_header",
    "MARKER",
    "MARKER_BYTES",
    "PAGE_HEADER_BYTES",
    "HEADER_TOTAL_BYTES",
    "PAGE_VERSIONS_SONAR",
]