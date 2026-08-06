"""
YOLO-format export for sonar targets.

Mirrors ``interchange/coco.py``'s ``write_coco`` — same input
(``CocoImageEntry`` list, despite the name it's just "one target's
canonical crop", not COCO-specific), same ``min_confidence`` gating, same
placeholder ``box_size`` caveat (no measured target dimension exists in
any source catalog seen so far). Kept as a separate writer rather than a
shared abstraction over both formats because the two output shapes
(one JSON tree vs. one label file per image + a class list) don't share
much beyond "filter, then compute a box".

2026 rshomberg@uri.edu
"""

from pathlib import Path

from sidescanpy.targets.core import CONFIDENCE_RANK, bbox_from_center, clip_bbox
from sidescanpy.targets.interchange.coco import CocoImageEntry


def write_yolo(
    entries: list[CocoImageEntry],
    output_dir: str | Path,
    *,
    category_name: str = "target",
    box_size: tuple[float, float] = (40.0, 40.0),
    min_confidence: str = "verified",
) -> dict:
    """Write ``entries`` to ``output_dir`` as a YOLO-format label set.

    Same inclusion filtering as ``write_coco`` (``position_confidence``
    at or above ``min_confidence``, opt-in only for lower confidence).
    Writes one label file per image at
    ``output_dir/labels/<Path(entry.file_name).stem>.txt`` — one
    ``class_id cx cy w h`` line per box, normalized to [0, 1] by that
    image's own width/height (the YOLO convention) — plus a
    ``output_dir/data.yaml`` naming the single class.

    Returns a dict of what was written (``{"names": [...], "images": [...]}``,
    one entry per included image with its label path), for inline
    inspection/testing — YOLO has no single canonical in-memory
    representation the way COCO's JSON dict does.
    """
    min_rank = CONFIDENCE_RANK[min_confidence]
    included = [e for e in entries if CONFIDENCE_RANK[e.target.position_confidence] >= min_rank]

    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    written_images = []
    for entry in included:
        t = entry.target
        row = t.extra_attributes.get("window_row")
        col = t.sample
        bbox, _was_clipped = clip_bbox(bbox_from_center(row, col, box_size), entry.width, entry.height)

        x, y, w, h = bbox
        cx = (x + w / 2) / entry.width
        cy = (y + h / 2) / entry.height
        norm_w = w / entry.width
        norm_h = h / entry.height

        label_path = labels_dir / f"{Path(entry.file_name).stem}.txt"
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        written_images.append({"file_name": entry.file_name, "label_file": str(label_path)})

    data_yaml_path = Path(output_dir) / "data.yaml"
    data_yaml_path.write_text(f"names:\n  0: {category_name}\nnc: 1\n")

    return {"names": [category_name], "images": written_images}
