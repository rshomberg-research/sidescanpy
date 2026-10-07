"""
Format-agnostic sonar target/contact model.

A ``Target`` describes one sonar contact/pick, which may carry position
information in either (or both) of two address spaces: *native*
(ping/channel/sample, lat/lon — how a survey's own target catalog
describes it) and *pixel* (image id + bbox — how ML interchange formats
like COCO/YOLO describe it).

Format-specific catalog readers live next to that format's other IO code
(e.g. ``sidescanpy.io.jsf.targets``), not here — this package holds only
the shared data model, JSON/CSV IO, and export interchange formats.

2026 rshomberg@uri.edu
"""

from sidescanpy.targets.core import PositionConfidence, Target, bbox_from_center, clip_bbox
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
