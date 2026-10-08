"""XTF (eXtended Triton Format) reading.

- :func:`read_xtf_header`, :func:`iter_packets`, :func:`count_packets` --
  file header and packet framing.
- :func:`read_xtf_pings` / :func:`read_xtf` -- sonar pings of one file
  into per-channel sample arrays (:class:`XtfPingData`).
- :func:`convert` -- one or more files' pings into a
  :class:`~sidescanpy.core.survey.SideScanSurvey`.

See ``docs/_internal/xtf-reader.md`` for the design.
"""

from sidescanpy.io.xtf.converter import convert
from sidescanpy.io.xtf.reader import (
    ChannelBlockStats,
    PacketStats,
    XtfChannel,
    XtfFileHeader,
    XtfFrequencyGroup,
    XtfPacket,
    XtfPingData,
    count_packets,
    iter_packets,
    ping_times,
    read_xtf,
    read_xtf_header,
    read_xtf_pings,
)

__all__ = [
    "ChannelBlockStats",
    "PacketStats",
    "XtfChannel",
    "XtfFileHeader",
    "XtfFrequencyGroup",
    "XtfPacket",
    "XtfPingData",
    "convert",
    "count_packets",
    "iter_packets",
    "ping_times",
    "read_xtf",
    "read_xtf_header",
    "read_xtf_pings",
]
