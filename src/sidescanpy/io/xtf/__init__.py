"""XTF (eXtended Triton Format) reading.

- :func:`read_xtf_header`, :func:`iter_packets`, :func:`count_packets` --
  file header and packet framing.
- :func:`read_xtf_pings` / :func:`read_xtf` -- sonar pings of one file
  into per-channel sample arrays (:class:`XtfPingData`).
- :func:`split_by_sample_rate` -- one file's pings cut into runs of
  constant sample rate (range-setting changes), one survey per run.
- :func:`convert` -- one or more files' pings into a
  :class:`~sidescanpy.core.survey.SideScanSurvey`.

See the "XTF" source format page (``docs/sources/xtf.md``) for format notes.
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
    XtfSegmentInfo,
    count_packets,
    iter_packets,
    ping_times,
    read_xtf,
    read_xtf_header,
    read_xtf_pings,
)
from sidescanpy.io.xtf.split import sample_rate_runs, split_by_sample_rate
from sidescanpy.io.xtf.subset import write_subset

__all__ = [
    "ChannelBlockStats",
    "PacketStats",
    "XtfChannel",
    "XtfFileHeader",
    "XtfFrequencyGroup",
    "XtfPacket",
    "XtfPingData",
    "XtfSegmentInfo",
    "convert",
    "count_packets",
    "iter_packets",
    "ping_times",
    "read_xtf",
    "read_xtf_header",
    "read_xtf_pings",
    "sample_rate_runs",
    "split_by_sample_rate",
    "write_subset",
]
