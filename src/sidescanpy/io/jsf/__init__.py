"""
JSF (Edgetech Survey Format) IO.

This module parses Edgetech .jsf binary files into typed message instances and
converts those messages into a validated :class:`SideScanSurvey`.

Public entry points
-------------------
- :func:`read_jsf` — eager read, returns ``list[JSFMessage]``.
- :func:`read_jsf_iter` — generator over messages (use for large files).
- :func:`convert` — dispatches a message list into a :class:`SideScanSurvey`.

Conventions
-----------
- Sonar messages expose ``timestamp_ns: int``. Auxiliary messages expose
  ``timestamp: float`` (epoch seconds).
- Per-ping fields with invalid validity flags currently appear as ``None``; the
  converter coerces them to ``np.nan``.
- All output time coordinates are ``datetime64[ns]``.

See ``docs/io_jsf.md`` for full reference and known issues.

2026 rshomberg@uri.edu
"""

from sidescanpy.io.jsf.reader import read_jsf, read_jsf_iter
from sidescanpy.io.jsf.converter import convert
from sidescanpy.io.jsf.messages import (
    JSFMessage,
    JSFMessageHeader,
    SonarDataMessage,
    PitchRollDataMessage,
    DopplerVelocityLogDataMessage,
    PressureSensorReadingMessage,
    NMEAStringMessage,
    NMEARMCMessage,
    NMEAGLLMessage,
    NMEAGGAMessage,
    NMEAVTGMessage,
    NMEAZDAMessage,
    NMEADBTMessage,
    NMEADPTMessage,
)

__all__ = [
    "read_jsf",
    "read_jsf_iter",
    "convert",
    "JSFMessage",
    "JSFMessageHeader",
    "SonarDataMessage",
    "PitchRollDataMessage",
    "DopplerVelocityLogDataMessage",
    "PressureSensorReadingMessage",
    "NMEAStringMessage",
    "NMEARMCMessage",
    "NMEAGLLMessage",
    "NMEAGGAMessage",
    "NMEAVTGMessage",
    "NMEAZDAMessage",
    "NMEADBTMessage",
    "NMEADPTMessage",
]