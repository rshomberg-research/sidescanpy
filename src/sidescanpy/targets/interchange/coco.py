"""
COCO-format export for sonar targets.

Scoped to JSF for now: one canonical crop image + one bounding-box
annotation per target, single category (no real class labels are assumed — ``Description``/``Tag`` are
commonly empty), defaulting to ``min_confidence="verified"`` so a dataset export
can't silently ship unlocalized or low-confidence boxes.

The bbox itself is a fixed placeholder size (``box_size``), not a
measured target dimension — no source catalog seen so far has real
``LengthMeters``/``WidthMeters`` data. This is recorded in the written
JSON's own ``info.description`` so the caveat travels with the file, not
just with whatever presentation wraps it.

2026 rshomberg@uri.edu
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sidescanpy.targets.core import CONFIDENCE_RANK, Target, bbox_from_center, clip_bbox

logger = logging.getLogger(__name__)


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


def resolve_categories(
    entries: list[CocoImageEntry],
    category_name: str = "target",
    categories: list[str] | None = None,
) -> list[str]:
    """Ordered class names for ``entries`` (index + 1 is the COCO id,
    index is the YOLO class).

    Uncategorized targets (``Target.category is None``) belong to
    ``category_name``. With no explicit ``categories``: the sorted set of
    names present (just ``[category_name]`` when nothing is categorized).
    With an explicit ordered ``categories`` list, that list is returned
    as-is so ids stay stable across separately-exported datasets.

    Raises
    ------
    ValueError
        If ``categories`` has duplicates, or a target's category is not in it.
    """
    present = {e.target.category if e.target.category is not None else category_name for e in entries}
    if categories is None:
        return sorted(present) if present else [category_name]
    names = list(categories)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate names in categories: {names}")
    unknown = sorted(present - set(names))
    if unknown:
        raise ValueError(f"target categories {unknown} are not in the explicit categories list {names}")
    return names


def write_coco(
    entries: list[CocoImageEntry],
    output_path: str | Path,
    *,
    category_name: str = "target",
    box_size: tuple[float, float] = (40.0, 40.0),
    min_confidence: str = "verified",
    categories: list[str] | None = None,
) -> dict:
    """Write ``entries`` to ``output_path`` as a COCO-format JSON dataset.

    Per-annotation class comes from ``Target.category`` (``None`` ->
    ``category_name``). The ``categories`` list is the sorted set of
    names present with ids 1..N, or, if ``categories`` is given, that
    explicit ordered list (see :func:`resolve_categories`). With no
    categorized targets the output is the single ``category_name``
    category, id 1.

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

    names = resolve_categories(included, category_name, categories)
    cat_ids = {n: i + 1 for i, n in enumerate(names)}
    categories_out = [{"id": i, "name": n} for n, i in cat_ids.items()]
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
                "category_id": cat_ids[t.category if t.category is not None else category_name],
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

    multi = names != [category_name]
    coco = {
        "info": {
            "description": (
                f"Multi-category sonar contact dataset ({', '.join(names)}). Bounding boxes "
                f"default to a fixed placeholder size {tuple(box_size)}, not a measured "
                "target dimension, unless a given CocoImageEntry.bbox_size overrides it "
                "with a real extent."
            ) if multi else (
                "Single-category ('target') sonar contact dataset. No real class "
                "labels exist in the source catalog (Description/Tag empty on "
                "every target). Bounding boxes default to a fixed placeholder size "
                f"{tuple(box_size)}, not a measured target dimension, unless a "
                "given CocoImageEntry.bbox_size overrides it with a real extent."
            ),
        },
        "licenses": [],
        "categories": categories_out,
        "images": images,
        "annotations": annotations,
    }
    Path(output_path).write_text(json.dumps(coco, indent=2))
    return coco
