"""
ML datasets: :class:`sidescanpy.core.target.Target` lists to COCO and YOLO
label files; :func:`build_target_dataset`, which renders a whole
annotated image dataset from converted surveys and a target catalog; and
swath tiles for annotation or detection, with the boxes drawn on them
turned back into targets (:mod:`sidescanpy.export.ml.tiles`).

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
from sidescanpy.export.ml.tiles import (
    TileBox,
    TileRecord,
    boxes_to_targets,
    read_coco_boxes,
    read_tile_index,
    write_swath_tiles,
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
    "TileBox",
    "TileRecord",
    "boxes_to_targets",
    "read_coco_boxes",
    "read_tile_index",
    "write_swath_tiles",
]
