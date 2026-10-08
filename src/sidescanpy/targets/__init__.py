"""
Format-agnostic sonar target/contact model.

A ``Target`` describes one sonar contact/pick, which may carry position
information in either (or both) of two address spaces: *native*
(ping/channel/sample, lat/lon — how a survey's own target catalog
describes it) and *pixel* (image id + bbox — how ML interchange formats
like COCO/YOLO describe it).

The data model itself lives in :mod:`sidescanpy.core.target` (re-exported
here). Format-specific catalog readers live next to that format's other IO
code (e.g. ``sidescanpy.io.jsf.targets``); this package holds target JSON/CSV
IO, window decoding and display helpers, and position correlation against
converted surveys. ML annotation writers (COCO/YOLO) are in
:mod:`sidescanpy.export.ml`.

2026 rshomberg@uri.edu
"""

from sidescanpy.core.target import PositionConfidence, Target, bbox_from_center, clip_bbox
from sidescanpy.targets.io import (
    read_targets_csv,
    read_targets_json,
    write_targets_csv,
    write_targets_json,
)

__all__ = [
    "Target",
    "PositionConfidence",
    "bbox_from_center",
    "clip_bbox",
    "read_targets_json",
    "write_targets_json",
    "read_targets_csv",
    "write_targets_csv",
]
