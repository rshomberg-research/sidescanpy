"""
2026 rshomberg@uri.edu
"""

import json

import pytest

from sidescanpy.core.target import Target
from sidescanpy.export.ml.coco import CocoImageEntry, write_coco


def _entry(
    target_id, confidence, row=50, col=60, width=120, height=100, file_name=None, bbox_size=None
) -> CocoImageEntry:
    t = Target(
        target_id=target_id,
        source_format="jsf_discover",
        catalog_source=f"Target-{target_id}.target",
        ping=1000,
        channel=0,
        sample=col,
        position_confidence=confidence,
        extra_attributes={"window_row": row},
    )
    return CocoImageEntry(
        target=t,
        file_name=file_name or f"target_{target_id}.png",
        width=width,
        height=height,
        bbox_size=bbox_size,
    )


def test_default_min_confidence_excludes_unverified(tmp_path):
    entries = [_entry("0001", "verified"), _entry("0002", "approximate"), _entry("0003", "unknown")]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out)
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "target_0001.png"


def test_explicit_min_confidence_includes_approximate(tmp_path):
    entries = [_entry("0001", "verified"), _entry("0002", "approximate"), _entry("0003", "unknown")]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out, min_confidence="approximate")
    assert {im["file_name"] for im in coco["images"]} == {"target_0001.png", "target_0002.png"}


def test_schema_shape_and_bbox_position(tmp_path):
    entries = [_entry("0001", "verified", row=50, col=60, width=120, height=100)]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out, box_size=(20.0, 20.0))

    assert coco["categories"] == [{"id": 1, "name": "target"}]
    ann = coco["annotations"][0]
    assert ann["image_id"] == coco["images"][0]["id"]
    assert ann["bbox"] == [50.0, 40.0, 20.0, 20.0]  # centered at (col=60,row=50), 20x20
    assert ann["area"] == 400.0
    assert ann["attributes"]["target_id"] == "0001"
    assert ann["attributes"]["was_clipped"] is False

    # file round-trips as valid JSON
    assert json.loads(out.read_text()) == coco


def test_bbox_clipped_near_image_edge(tmp_path):
    # target near the top-left corner: a centered box would run off-image
    entries = [_entry("0001", "verified", row=2, col=2, width=120, height=100)]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out, box_size=(20.0, 20.0))
    ann = coco["annotations"][0]
    x, y, w, h = ann["bbox"]
    assert x >= 0 and y >= 0
    assert ann["attributes"]["was_clipped"] is True


def test_per_entry_bbox_size_overrides_global_box_size(tmp_path):
    entries = [_entry("0001", "verified", row=50, col=60, width=120, height=100, bbox_size=(10.0, 8.0))]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out, box_size=(40.0, 40.0))
    ann = coco["annotations"][0]
    assert ann["bbox"][2:] == [10.0, 8.0]


def test_multiple_targets_sharing_file_name_become_one_image_two_annotations(tmp_path):
    entries = [
        _entry("0001", "verified", row=10, col=10, file_name="window_0.png"),
        _entry("0002", "verified", row=90, col=90, file_name="window_0.png"),
    ]
    out = tmp_path / "targets.coco.json"
    coco = write_coco(entries, out)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert {a["image_id"] for a in coco["annotations"]} == {coco["images"][0]["id"]}
    assert {a["id"] for a in coco["annotations"]} == {1, 2}
    assert {a["attributes"]["target_id"] for a in coco["annotations"]} == {"0001", "0002"}


def test_conflicting_dims_for_same_file_name_raises(tmp_path):
    entries = [
        _entry("0001", "verified", width=120, height=100, file_name="window_0.png"),
        _entry("0002", "verified", width=200, height=200, file_name="window_0.png"),
    ]
    out = tmp_path / "targets.coco.json"
    with pytest.raises(ValueError):
        write_coco(entries, out)


def test_image_without_annotations_is_an_image_record_only(tmp_path):
    entries = [_entry("1", "verified"), CocoImageEntry(target=None, file_name="bg_0.png", width=64, height=32)]
    coco = write_coco(entries, tmp_path / "c.json")
    assert [im["file_name"] for im in coco["images"]] == ["target_1.png", "bg_0.png"]
    assert coco["images"][1] == {"id": 2, "file_name": "bg_0.png", "width": 64, "height": 32}
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["image_id"] == 1
    assert json.loads((tmp_path / "c.json").read_text()) == coco


def test_only_annotation_free_images_keep_default_category(tmp_path):
    coco = write_coco([CocoImageEntry(target=None, file_name="bg_0.png", width=8, height=8)], tmp_path / "c.json")
    assert coco["categories"] == [{"id": 1, "name": "target"}]
    assert coco["annotations"] == []
