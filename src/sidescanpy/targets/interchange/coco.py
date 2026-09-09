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

from sidescanpy.targets.core import CONFIDENCE_RANK, Target, bbox_from_center, clip_bbox


@dataclass
class CocoImageEntry:
    """One target's canonical crop, ready to become a COCO image+annotation.

    Multiple entries may share the same ``file_name`` — e.g. every
    cataloged target that falls within one exported window (see
    ``sidescanpy.export.windows.targets_within_window``). They become one
    shared COCO image record with one annotation per entry, rather than
    one image per target. Every entry sharing a ``file_name`` must agree
    on ``width``/``height`` (the same physical image).

    ``bbox_size``, if given, overrides ``write_coco``/``write_yolo``'s
    global ``box_size`` for this entry only — for a real, measured target
    extent instead of the shared placeholder.
    """

    target: Target
    file_name: str
    width: int
    height: int
    bbox_size: tuple[float, float] | None = None


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

    Entries sharing a ``file_name`` become one image with multiple
    annotations (see ``CocoImageEntry``'s own docstring) — each entry's
    own ``bbox_size`` (if set) is used for its box, else the shared
    ``box_size``.

    Returns the dict that was written, for inline inspection/testing.

    Raises
    ------
    ValueError
        If two entries share a ``file_name`` but disagree on
        ``width``/``height``.
    """
    min_rank = CONFIDENCE_RANK[min_confidence]
    included = [e for e in entries if CONFIDENCE_RANK[e.target.position_confidence] >= min_rank]

    categories = [{"id": 1, "name": category_name}]
    images = []
    annotations = []

    image_ids: dict[str, int] = {}
    annotation_id = 1
    for entry in included:
        image_id = image_ids.get(entry.file_name)
        if image_id is None:
            image_id = len(image_ids) + 1
            image_ids[entry.file_name] = image_id
            images.append(
                {"id": image_id, "file_name": entry.file_name, "width": entry.width, "height": entry.height}
            )
        else:
            existing = images[image_id - 1]
            if (existing["width"], existing["height"]) != (entry.width, entry.height):
                raise ValueError(
                    f"conflicting width/height for file_name={entry.file_name!r}: "
                    f"{(existing['width'], existing['height'])} vs {(entry.width, entry.height)}"
                )

        t = entry.target
        row = t.extra_attributes.get("window_row")
        col = t.sample
        entry_box_size = entry.bbox_size if entry.bbox_size is not None else box_size
        bbox, was_clipped = clip_bbox(bbox_from_center(row, col, entry_box_size), entry.width, entry.height)

        annotations.append(
            {
                "id": annotation_id,
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
        annotation_id += 1

    coco = {
        "info": {
            "description": (
                "Single-category ('target') sonar contact dataset. No real class "
                "labels exist in the source catalog (Description/Tag empty on "
                "every target). Bounding boxes default to a fixed placeholder size "
                f"{tuple(box_size)}, not a measured target dimension, unless a "
                "given CocoImageEntry.bbox_size overrides it with a real extent."
            ),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    Path(output_path).write_text(json.dumps(coco, indent=2))
    return coco
