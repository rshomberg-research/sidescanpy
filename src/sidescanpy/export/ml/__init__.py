"""
ML annotation writers: :class:`sidescanpy.core.target.Target` lists to COCO
and YOLO label files.

2026 rshomberg@uri.edu
"""

from sidescanpy.export.ml.coco import CocoImageEntry, resolve_categories, write_coco
from sidescanpy.export.ml.yolo import write_yolo

__all__ = ["CocoImageEntry", "resolve_categories", "write_coco", "write_yolo"]
