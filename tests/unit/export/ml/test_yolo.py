"""
2026 rshomberg@uri.edu
"""

import pytest

from sidescanpy.core.target import Target
from sidescanpy.export.ml.coco import CocoImageEntry
from sidescanpy.export.ml.yolo import write_yolo


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
    out_dir = tmp_path / "yolo"
    result = write_yolo(entries, out_dir)
    assert len(result["images"]) == 1
    assert result["images"][0]["file_name"] == "target_0001.png"
    assert (out_dir / "labels" / "target_0001.txt").exists()
    assert not (out_dir / "labels" / "target_0002.txt").exists()


def test_explicit_min_confidence_includes_approximate(tmp_path):
    entries = [_entry("0001", "verified"), _entry("0002", "approximate"), _entry("0003", "unknown")]
    out_dir = tmp_path / "yolo"
    result = write_yolo(entries, out_dir, min_confidence="approximate")
    assert {im["file_name"] for im in result["images"]} == {"target_0001.png", "target_0002.png"}


def test_label_file_has_normalized_bbox(tmp_path):
    entries = [_entry("0001", "verified", row=50, col=60, width=120, height=100)]
    out_dir = tmp_path / "yolo"
    write_yolo(entries, out_dir, box_size=(20.0, 20.0))

    label_path = out_dir / "labels" / "target_0001.txt"
    class_id, cx, cy, w, h = label_path.read_text().split()
    assert class_id == "0"
    # box centered at (col=60, row=50), 20x20, in a 120x100 image
    assert float(cx) == pytest.approx(60.0 / 120.0, abs=1e-5)
    assert float(cy) == pytest.approx(50.0 / 100.0, abs=1e-5)
    assert float(w) == pytest.approx(20.0 / 120.0, abs=1e-5)
    assert float(h) == pytest.approx(20.0 / 100.0, abs=1e-5)


def test_bbox_clipped_near_image_edge(tmp_path):
    # target near the top-left corner: a centered box would run off-image
    entries = [_entry("0001", "verified", row=2, col=2, width=120, height=100)]
    out_dir = tmp_path / "yolo"
    write_yolo(entries, out_dir, box_size=(20.0, 20.0))

    label_path = out_dir / "labels" / "target_0001.txt"
    _class_id, cx, cy, w, h = (float(v) if i > 0 else v for i, v in enumerate(label_path.read_text().split()))
    # clipped box stays within [0, 1] normalized bounds
    assert 0.0 <= cx - w / 2
    assert 0.0 <= cy - h / 2


def test_data_yaml_names_single_category(tmp_path):
    entries = [_entry("0001", "verified")]
    out_dir = tmp_path / "yolo"
    write_yolo(entries, out_dir, category_name="target")
    data_yaml = (out_dir / "data.yaml").read_text()
    assert "target" in data_yaml
    assert "nc: 1" in data_yaml


def test_per_entry_bbox_size_overrides_global_box_size(tmp_path):
    entries = [_entry("0001", "verified", row=50, col=60, width=100, height=100, bbox_size=(10.0, 10.0))]
    out_dir = tmp_path / "yolo"
    write_yolo(entries, out_dir, box_size=(40.0, 40.0))
    label_path = out_dir / "labels" / "target_0001.txt"
    _class_id, _cx, _cy, w, h = label_path.read_text().split()
    assert float(w) == pytest.approx(10.0 / 100.0)
    assert float(h) == pytest.approx(10.0 / 100.0)


def test_multiple_targets_sharing_file_name_produce_multiple_lines(tmp_path):
    entries = [
        _entry("0001", "verified", row=10, col=10, file_name="window_0.png"),
        _entry("0002", "verified", row=90, col=90, file_name="window_0.png"),
    ]
    out_dir = tmp_path / "yolo"
    result = write_yolo(entries, out_dir)
    assert len(result["images"]) == 1
    label_path = out_dir / "labels" / "window_0.txt"
    lines = label_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_conflicting_dims_for_same_file_name_raises(tmp_path):
    entries = [
        _entry("0001", "verified", width=120, height=100, file_name="window_0.png"),
        _entry("0002", "verified", width=200, height=200, file_name="window_0.png"),
    ]
    out_dir = tmp_path / "yolo"
    with pytest.raises(ValueError):
        write_yolo(entries, out_dir)
