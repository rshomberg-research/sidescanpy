"""XTF (eXtended Triton Format) reading.

Low-level reading is implemented: :func:`read_xtf_header`,
:func:`iter_packets` and :func:`read_xtf_pings` (sonar pings into
per-channel sample arrays). Conversion to a survey is planned;
:func:`read_xtf` raises ``NotImplementedError`` until then. See
``docs/_internal/xtf-reader.md`` for the design.
"""

from sidescanpy.io.xtf.reader import (
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
    "PacketStats",
    "XtfChannel",
    "XtfFileHeader",
    "XtfFrequencyGroup",
    "XtfPacket",
    "XtfPingData",
    "count_packets",
    "iter_packets",
    "ping_times",
    "read_xtf",
    "read_xtf_header",
    "read_xtf_pings",
]
