"""
JSON/CSV IO for lists of ``Target`` records.

JSON is the source-of-truth format (round-trips every field losslessly).
CSV is a secondary, flatten-to-rows view for manual/spreadsheet review —
``extra_attributes`` is JSON-encoded into a single CSV column rather than
expanded, since its shape is format-specific and not fixed across rows.

2026 rshomberg@uri.edu
"""

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from sidescanpy.core.target import Target

logger = logging.getLogger(__name__)


def write_targets_json(targets: list[Target], path: str | Path) -> None:
    """Write ``targets`` to ``path`` as a JSON list of records."""
    records = [asdict(t) for t in targets]
    for r in records:
        # Omit an unset category so output for uncategorized catalogs is
        # unchanged from before the field existed.
        if r.get("category") is None:
            r.pop("category", None)
    Path(path).write_text(json.dumps(records, indent=2))


def read_targets_json(path: str | Path) -> list[Target]:
    """Read a list of ``Target`` records previously written by ``write_targets_json``."""
    records = json.loads(Path(path).read_text())
    targets = []
    for r in records:
        bbox = r.get("bbox")
        if bbox is not None:
            r = {**r, "bbox": tuple(bbox)}
        targets.append(Target(**r))
    return targets


_CSV_FIELDS = [
    "target_id",
    "source_format",
    "catalog_source",
    "pick_timestamp",
    "ping",
    "channel",
    "sample",
    "latitude",
    "longitude",
    "image_id",
    "bbox",
    "position_confidence",
    "label",
    "extra_attributes",
]


def write_targets_csv(targets: list[Target], path: str | Path) -> None:
    """Write ``targets`` to ``path`` as a flattened CSV (lossy: ``bbox``/
    ``extra_attributes`` are JSON-encoded into single cells). A
    ``category`` column is written only if at least one target has one."""
    fields = list(_CSV_FIELDS)
    if any(t.category is not None for t in targets):
        fields.insert(fields.index("label") + 1, "category")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in targets:
            row = asdict(t)
            row["bbox"] = json.dumps(row["bbox"]) if row["bbox"] is not None else ""
            row["extra_attributes"] = json.dumps(row["extra_attributes"])
            writer.writerow(row)


def read_targets_csv(path: str | Path, *, category_column: str | None = None) -> list[Target]:
    """Read a CSV previously written by ``write_targets_csv``.

    ``category_column`` names the column to use as ``Target.category``
    (default: the ``category`` column if present, else none). Empty cells
    become ``None``.
    """
    category_column = category_column or "category"
    targets = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            bbox = json.loads(row["bbox"]) if row["bbox"] else None
            extra_attributes = json.loads(row["extra_attributes"]) if row["extra_attributes"] else {}
            targets.append(
                Target(
                    target_id=row["target_id"],
                    source_format=row["source_format"],
                    catalog_source=row["catalog_source"],
                    pick_timestamp=float(row["pick_timestamp"]) if row["pick_timestamp"] else None,
                    ping=int(row["ping"]) if row["ping"] else None,
                    channel=int(row["channel"]) if row["channel"] else None,
                    sample=int(row["sample"]) if row["sample"] else None,
                    latitude=float(row["latitude"]) if row["latitude"] else None,
                    longitude=float(row["longitude"]) if row["longitude"] else None,
                    image_id=row["image_id"] or None,
                    bbox=tuple(bbox) if bbox is not None else None,
                    position_confidence=row["position_confidence"],
                    label=row["label"] or None,
                    category=(row.get(category_column) or None),
                    extra_attributes=extra_attributes,
                )
            )
    return targets
