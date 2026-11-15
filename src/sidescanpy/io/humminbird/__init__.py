"""Humminbird side imaging (``.DAT`` + ``.SON``) reading.

- :func:`read_dat`, :func:`read_son`, :func:`read_humminbird` -- the
  recording header and each beam's ping frames
  (:class:`HumRecording`).
- :func:`convert` -- one or more recordings into a
  :class:`~sidescanpy.core.survey.SideScanSurvey` (side imaging only).
- :func:`write_subset` -- a run of pings cut out byte for byte, for
  test fixtures.

See the "Humminbird" source format page (``docs/sources/humminbird.md``)
for format notes.
"""

from sidescanpy.io.humminbird.converter import HUMMINBIRD_SAMPLE_RATE_HZ, convert
from sidescanpy.io.humminbird.reader import (
    BEAM_NAMES,
    BEAM_PORT,
    BEAM_STARBOARD,
    HumBeam,
    HumDat,
    HumRecording,
    humminbird_to_lonlat,
    lonlat_to_humminbird,
    read_dat,
    read_humminbird,
    read_son,
    son_directory,
)
from sidescanpy.io.humminbird.subset import write_subset

__all__ = [
    "BEAM_NAMES",
    "BEAM_PORT",
    "BEAM_STARBOARD",
    "HUMMINBIRD_SAMPLE_RATE_HZ",
    "HumBeam",
    "HumDat",
    "HumRecording",
    "convert",
    "humminbird_to_lonlat",
    "lonlat_to_humminbird",
    "read_dat",
    "read_humminbird",
    "read_son",
    "son_directory",
    "write_subset",
]
