"""
YOLO-format export for sonar targets.

Mirrors ``export/ml/coco.py``'s ``write_coco`` — same input
(``CocoImageEntry`` list, despite the name it's just "one target's
canonical crop", not COCO-specific), same ``min_confidence`` gating, same
placeholder ``box_size`` caveat (no measured target dimension exists in
any source catalog seen so far). Kept as a separate writer rather than a
shared abstraction over both formats because the two output shapes
(one JSON tree vs. one label file per image + a class list) don't share
much beyond "filter, then compute a box".

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path

from sidescanpy.core.target import bbox_from_center, clip_bbox
from sidescanpy.export.ml.coco import CocoImageEntry, _filter_by_confidence, resolve_categories

logger = logging.getLogger(__name__)


def write_yolo(
    entries: list[CocoImageEntry],
    output_dir: str | Path,
    *,
    category_name: str = "target",
    box_size: tuple[float, float] = (40.0, 40.0),
    min_confidence: str = "verified",
    categories: list[str] | None = None,
) -> dict:
    """Write ``entries`` to ``output_dir`` as a YOLO-format label set.

    Same inclusion filtering as ``write_coco`` (``position_confidence``
    at or above ``min_confidence``, opt-in only for lower confidence),
    and the same multi-entries-per-``file_name`` grouping (one label
    file gets one line per entry sharing that ``file_name``, instead of
    the last one silently overwriting the rest) — see
    ``CocoImageEntry``'s own docstring. Writes one label file per image
    at ``output_dir/labels/<Path(file_name).stem>.txt`` — one
    ``class_id cx cy w h`` line per box, normalized to [0, 1] by that
    image's own width/height (the YOLO convention) — plus a
    ``output_dir/data.yaml`` naming the single class. Each entry's own
    ``bbox_size`` (if set) is used for its box, else the shared
    ``box_size``. An entry with ``target=None`` contributes an image with
    an empty label file (a background image).

    Returns a dict of what was written (``{"names": [...], "images": [...]}``,
    one entry per included image with its label path), for inline
    inspection/testing — YOLO has no single canonical in-memory
    representation the way COCO's JSON dict does.

    Class index is the position in the sorted set of ``Target.category``
    names present (uncategorized -> ``category_name``), or in the explicit
    ordered ``categories`` list if given; ``data.yaml`` lists the same
    order. See ``resolve_categories``.

    Raises
    ------
    ValueError
        If two entries share a ``file_name`` but disagree on
        ``width``/``height``.
    """
    included = _filter_by_confidence(entries, min_confidence)

    names = resolve_categories(included, category_name, categories)
    class_ids = {n: i for i, n in enumerate(names)}

    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    lines_by_file: dict[str, list[str]] = {}
    dims_by_file: dict[str, tuple[int, int]] = {}
    order: list[str] = []
    for entry in included:
        dims = dims_by_file.get(entry.file_name)
        if dims is None:
            lines_by_file[entry.file_name] = []
            dims_by_file[entry.file_name] = (entry.width, entry.height)
            order.append(entry.file_name)
        elif dims != (entry.width, entry.height):
            raise ValueError(
                f"conflicting width/height for file_name={entry.file_name!r}: "
                f"{dims} vs {(entry.width, entry.height)}"
            )

        t = entry.target
        if t is None:
            continue
        row = t.extra_attributes.get("window_row")
        col = t.sample
        entry_box_size = entry.bbox_size if entry.bbox_size is not None else box_size
        bbox, _was_clipped = clip_bbox(bbox_from_center(row, col, entry_box_size), entry.width, entry.height)

        x, y, w, h = bbox
        cx = (x + w / 2) / entry.width
        cy = (y + h / 2) / entry.height
        norm_w = w / entry.width
        norm_h = h / entry.height
        lines_by_file[entry.file_name].append(f"{class_ids[t.category if t.category is not None else category_name]} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    written_images = []
    for file_name in order:
        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("".join(lines_by_file[file_name]))
        written_images.append({"file_name": file_name, "label_file": str(label_path)})

    data_yaml_path = Path(output_dir) / "data.yaml"
    names_block = "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    data_yaml_path.write_text(f"names:\n{names_block}nc: {len(names)}\n")

    return {"names": names, "images": written_images}
