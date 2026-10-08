"""
Multi-category COCO/YOLO annotation tests.

2026 rshomberg@uri.edu
"""

import json

import pytest

from sidescanpy.core.target import Target
from sidescanpy.export.ml.coco import CocoImageEntry, write_coco
from sidescanpy.export.ml.yolo import write_yolo
from sidescanpy.targets.io import (
    read_targets_csv,
    read_targets_json,
    write_targets_csv,
    write_targets_json,
)


def _entry(tid, category=None, file_name=None):
    t = Target(
        target_id=tid, source_format="x", catalog_source="c", sample=60,
        position_confidence="verified", category=category, extra_attributes={"window_row": 50},
    )
    return CocoImageEntry(target=t, file_name=file_name or f"t_{tid}.png", width=120, height=100)


def test_default_single_category_unchanged(tmp_path):
    coco = write_coco([_entry("1"), _entry("2")], tmp_path / "c.json")
    assert coco["categories"] == [{"id": 1, "name": "target"}]
    assert {a["category_id"] for a in coco["annotations"]} == {1}
    assert "Single-category" in coco["info"]["description"]
    res = write_yolo([_entry("1")], tmp_path / "y")
    assert (tmp_path / "y" / "data.yaml").read_text() == "names:\n  0: target\nnc: 1\n"
    assert (tmp_path / "y" / "labels" / "t_1.txt").read_text().startswith("0 ")
    assert res["names"] == ["target"]


def test_two_classes_coco_and_yolo(tmp_path):
    entries = [_entry("1", "class_b"), _entry("2", "class_a"), _entry("3", "class_b", file_name="t_1.png")]
    coco = write_coco(entries, tmp_path / "c.json")
    assert coco["categories"] == [{"id": 1, "name": "class_a"}, {"id": 2, "name": "class_b"}]
    by_target = {a["attributes"]["target_id"]: a["category_id"] for a in coco["annotations"]}
    assert by_target == {"1": 2, "2": 1, "3": 2}
    res = write_yolo(entries, tmp_path / "y")
    assert res["names"] == ["class_a", "class_b"]
    assert (tmp_path / "y" / "data.yaml").read_text() == "names:\n  0: class_a\n  1: class_b\nnc: 2\n"
    assert [ln.split()[0] for ln in (tmp_path / "y" / "labels" / "t_1.txt").read_text().splitlines()] == ["1", "1"]
    assert (tmp_path / "y" / "labels" / "t_2.txt").read_text().split()[0] == "0"


def test_explicit_ordering_keeps_unused_classes(tmp_path):
    cats = ["class_b", "class_c", "class_a"]
    coco = write_coco([_entry("1", "class_a")], tmp_path / "c.json", categories=cats)
    assert [c["name"] for c in coco["categories"]] == cats
    assert coco["annotations"][0]["category_id"] == 3
    write_yolo([_entry("1", "class_a")], tmp_path / "y", categories=cats)
    assert (tmp_path / "y" / "labels" / "t_1.txt").read_text().split()[0] == "2"
    assert "  1: class_c\n" in (tmp_path / "y" / "data.yaml").read_text()


def test_unknown_category_is_error(tmp_path):
    with pytest.raises(ValueError, match="class_z"):
        write_coco([_entry("1", "class_z")], tmp_path / "c.json", categories=["class_a"])
    with pytest.raises(ValueError, match="class_z"):
        write_yolo([_entry("1", "class_z")], tmp_path / "y", categories=["class_a"])


def test_json_omits_unset_category_and_round_trips(tmp_path):
    plain = [Target(target_id="1", source_format="x", catalog_source="c")]
    write_targets_json(plain, tmp_path / "p.json")
    assert "category" not in json.loads((tmp_path / "p.json").read_text())[0]
    cat = [Target(target_id="1", source_format="x", catalog_source="c", category="class_a")]
    write_targets_json(cat, tmp_path / "c.json")
    assert read_targets_json(tmp_path / "c.json")[0].category == "class_a"


def test_csv_category_column(tmp_path):
    plain = [Target(target_id="1", source_format="x", catalog_source="c")]
    write_targets_csv(plain, tmp_path / "p.csv")
    assert "category" not in (tmp_path / "p.csv").read_text().splitlines()[0]
    assert read_targets_csv(tmp_path / "p.csv")[0].category is None
    cat = [Target(target_id="1", source_format="x", catalog_source="c", category="class_a", label="lbl")]
    write_targets_csv(cat, tmp_path / "c.csv")
    assert read_targets_csv(tmp_path / "c.csv")[0].category == "class_a"
    # map another column to the category
    assert read_targets_csv(tmp_path / "c.csv", category_column="label")[0].category == "lbl"
