"""
ML datasets: :class:`sidescanpy.core.target.Target` lists to COCO and YOLO
label files, and :func:`build_target_dataset`, which renders a whole
annotated image dataset from converted surveys and a target catalog.

2026 rshomberg@uri.edu
"""

from sidescanpy.export.ml.coco import CocoImageEntry, resolve_categories, write_coco
from sidescanpy.export.ml.dataset import (
    LAYOUTS,
    DatasetResult,
    DatasetSpec,
    SkippedTarget,
    StyleResult,
    build_target_dataset,
)
from sidescanpy.export.ml.yolo import write_yolo

__all__ = [
    "CocoImageEntry",
    "resolve_categories",
    "write_coco",
    "write_yolo",
    "LAYOUTS",
    "DatasetResult",
    "DatasetSpec",
    "SkippedTarget",
    "StyleResult",
    "build_target_dataset",
]
