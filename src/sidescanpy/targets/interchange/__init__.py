"""
Export converters between :class:`sidescanpy.targets.Target` and external
ML interchange formats.

2026 rshomberg@uri.edu
"""

from sidescanpy.targets.interchange.coco import CocoImageEntry, write_coco

__all__ = ["CocoImageEntry", "write_coco"]
