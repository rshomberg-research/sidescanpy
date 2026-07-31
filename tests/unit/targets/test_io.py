"""
2026 rshomberg@uri.edu
"""

from sidescanpy.targets.core import Target
from sidescanpy.targets.io import (
    read_targets_csv,
    read_targets_json,
    write_targets_csv,
    write_targets_json,
)


def _sample_targets() -> list[Target]:
    return [
        Target(
            target_id="1",
            source_format="jsf_discover",
            catalog_source="Target-0001.target",
            pick_timestamp=1234.5,
            ping=5300,
            channel=0,
            sample=1854,
            latitude=41.60784,
            longitude=-71.39521,
            image_id="target_0001.png",
            bbox=(1834.0, 80.0, 40.0, 40.0),
            position_confidence="verified",
            label=None,
            extra_attributes={"note": "clean window"},
        ),
        Target(
            target_id="2",
            source_format="jsf_discover",
            catalog_source="Target-0002.target",
        ),
    ]


def test_json_round_trip(tmp_path):
    targets = _sample_targets()
    path = tmp_path / "targets.json"
    write_targets_json(targets, path)
    loaded = read_targets_json(path)
    assert loaded == targets


def test_json_round_trip_all_none_fields(tmp_path):
    targets = [Target(target_id="3", source_format="jsf_discover", catalog_source="x")]
    path = tmp_path / "targets.json"
    write_targets_json(targets, path)
    loaded = read_targets_json(path)
    assert loaded == targets


def test_csv_round_trip(tmp_path):
    targets = _sample_targets()
    path = tmp_path / "targets.csv"
    write_targets_csv(targets, path)
    loaded = read_targets_csv(path)
    assert loaded == targets
