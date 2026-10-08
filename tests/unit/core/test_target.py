"""
Unit tests for sidescanpy.core.target.

2026 rshomberg@uri.edu
"""

from sidescanpy.core.target import Target


def test_defaults_are_none_or_empty():
    t = Target(target_id="1", source_format="jsf_discover", catalog_source="Target-0001.target")
    assert t.ping is None
    assert t.bbox is None
    assert t.position_confidence == "unknown"
    assert t.label is None
    assert t.extra_attributes == {}


def test_full_native_and_pixel_fields():
    t = Target(
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
    )
    assert t.bbox == (1834.0, 80.0, 40.0, 40.0)
    assert t.position_confidence == "verified"
