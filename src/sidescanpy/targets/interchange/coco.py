"""
COCO-format export for sonar targets.

Per ``docs/planning/v0.10.0-target-catalog-and-coco-export.md`` Phase 5,
scoped to JSF for now: one canonical crop image + one bounding-box
annotation per target, single category (no real class labels exist in
any dataset seen so far — every ``Description``/``Tag`` checked was
empty), defaulting to ``min_confidence="verified"`` so a dataset export
can't silently ship unlocalized or low-confidence boxes.

The bbox itself is a fixed placeholder size (``box_size``), not a
measured target dimension — no source catalog seen so far has real
``LengthMeters``/``WidthMeters`` data. This is recorded in the written
JSON's own ``info.description`` so the caveat travels with the file, not
just with whatever presentation wraps it.

2026 rshomberg@uri.edu
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sidescanpy.targets.core import Target, bbox_from_center

_CONFIDENCE_RANK = {"unknown": 0, "approximate": 1, "verified": 2}


@dataclass
class CocoImageEntry:
    """One target's canonical crop, ready to become a COCO image+annotation."""

    target: Target
    file_name: str
    width: int
    height: int


def _clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[tuple[float, float, float, float], bool]:
    x, y, w, h = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    clipped = (x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))
    return clipped, clipped != bbox


def write_coco(
    entries: list[CocoImageEntry],
    output_path: str | Path,
    *,
    category_name: str = "target",
    box_size: tuple[float, float] = (40.0, 40.0),
    min_confidence: str = "verified",
) -> dict:
    """Write ``entries`` to ``output_path`` as a COCO-format JSON dataset.

    Only targets with ``position_confidence`` at or above
    ``min_confidence`` (ordered ``unknown < approximate < verified``) are
    included — pass ``min_confidence="approximate"`` or ``"unknown"``
    explicitly to include lower-confidence targets; this must be an
    opt-in, not the default, per the design doc's own rationale (a prior
    export in a sibling project shipped an unverified/wrong bounding-box
    hypothesis by not gating on this).

    Returns the dict that was written, for inline inspection/testing.
    """
    min_rank = _CONFIDENCE_RANK[min_confidence]
    included = [e for e in entries if _CONFIDENCE_RANK[e.target.position_confidence] >= min_rank]

    categories = [{"id": 1, "name": category_name}]
    images = []
    annotations = []

    for image_id, entry in enumerate(included, start=1):
        t = entry.target
        images.append(
            {
                "id": image_id,
                "file_name": entry.file_name,
                "width": entry.width,
                "height": entry.height,
            }
        )

        row = t.extra_attributes.get("window_row")
        col = t.sample
        bbox, was_clipped = _clip_bbox(bbox_from_center(row, col, box_size), entry.width, entry.height)

        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": list(bbox),
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [],
                "attributes": {
                    "target_id": t.target_id,
                    "position_confidence": t.position_confidence,
                    "was_clipped": was_clipped,
                },
            }
        )

    coco = {
        "info": {
            "description": (
                "Single-category ('target') sonar contact dataset. No real class "
                "labels exist in the source catalog (Description/Tag empty on "
                "every target). Bounding boxes are a fixed placeholder size "
                f"{tuple(box_size)}, not a measured target dimension — no source "
                "catalog seen so far has real LengthMeters/WidthMeters data."
            ),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    Path(output_path).write_text(json.dumps(coco, indent=2))
    return coco
