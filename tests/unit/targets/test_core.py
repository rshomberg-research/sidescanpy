"""
2026 rshomberg@uri.edu
"""

import numpy as np

from sidescanpy.targets.core import Target, detrend_by_range, orient_for_display


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


def _make_target(channel, sample=1854, window_row=1, **kwargs):
    return Target(
        target_id="1", source_format="jsf_discover", catalog_source="Target-0001.target",
        channel=channel, sample=sample,
        extra_attributes={} if window_row is None else {"window_row": window_row},
        **kwargs,
    )


class TestOrientForDisplay:
    """Vertical (ping-axis) flip always applies, matching Discover's
    display order. Horizontal (sample-axis) mirror applies only to port
    channel (0), matching viz/waterfall.py's existing port/starboard sign
    convention — starboard (and anything else) skips only that part."""

    def test_vertical_flip_always_applied(self):
        window = np.arange(30).reshape(3, 10)  # 3 pings x 10 samples
        target = _make_target(channel=1, window_row=0)  # starboard: no horizontal flip

        oriented_window, oriented_target = orient_for_display(window, target)

        np.testing.assert_array_equal(oriented_window, window[::-1, :])

    def test_vertical_flip_remaps_window_row(self):
        window = np.arange(30).reshape(3, 10)
        target = _make_target(channel=1, window_row=0)

        _, oriented_target = orient_for_display(window, target)

        assert oriented_target.extra_attributes["window_row"] == 3 - 1 - 0  # == 2

    def test_missing_window_row_leaves_extra_attributes_unchanged(self):
        window = np.arange(30).reshape(3, 10)
        target = _make_target(channel=1, window_row=None)

        _, oriented_target = orient_for_display(window, target)

        assert "window_row" not in oriented_target.extra_attributes

    def test_port_channel_mirrors_window_and_remaps_sample(self):
        window = np.arange(20).reshape(2, 10)  # 2 pings x 10 samples
        target = _make_target(channel=0, sample=3, window_row=0)

        oriented_window, oriented_target = orient_for_display(window, target)

        np.testing.assert_array_equal(oriented_window, window[::-1, ::-1])
        assert oriented_target.sample == 10 - 1 - 3  # == 6

    def test_port_channel_sample_zero_maps_to_last_column(self):
        window = np.arange(20).reshape(2, 10)
        target = _make_target(channel=0, sample=0, window_row=0)

        _, oriented_target = orient_for_display(window, target)

        assert oriented_target.sample == 9

    def test_port_channel_none_sample_stays_none(self):
        window = np.arange(20).reshape(2, 10)
        target = _make_target(channel=0, sample=None, window_row=0)

        _, oriented_target = orient_for_display(window, target)

        assert oriented_target.sample is None

    def test_starboard_channel_skips_horizontal_flip_only(self):
        window = np.arange(20).reshape(2, 10)
        target = _make_target(channel=1, sample=3, window_row=0)

        oriented_window, oriented_target = orient_for_display(window, target)

        np.testing.assert_array_equal(oriented_window, window[::-1, :])
        assert oriented_target.sample == 3  # unchanged — no horizontal mirror

    def test_none_channel_skips_horizontal_flip_only(self):
        window = np.arange(20).reshape(2, 10)
        target = _make_target(channel=None, sample=3, window_row=0)

        oriented_window, oriented_target = orient_for_display(window, target)

        np.testing.assert_array_equal(oriented_window, window[::-1, :])
        assert oriented_target.sample == 3

    def test_other_fields_unchanged_for_port(self):
        target = Target(
            target_id="7", source_format="jsf_discover", catalog_source="Target-0007.target",
            channel=0, sample=3, latitude=41.6, longitude=-71.4, position_confidence="verified",
            extra_attributes={"window_row": 0},
        )
        window = np.arange(20).reshape(2, 10)

        _, oriented_target = orient_for_display(window, target)

        assert oriented_target.target_id == "7"
        assert oriented_target.latitude == 41.6
        assert oriented_target.longitude == -71.4
        assert oriented_target.position_confidence == "verified"


class TestDetrendByRange:
    """Flattens a per-column (range-dependent) brightness gradient by
    subtracting each column's median across pings."""

    def test_flattens_known_linear_gradient(self):
        n_pings, n_samples = 20, 50
        col_trend = np.linspace(0.0, 100.0, n_samples)
        window = np.tile(col_trend, (n_pings, 1))  # every ping identical

        result = detrend_by_range(window)

        np.testing.assert_array_almost_equal(result, np.zeros_like(window))

    def test_preserves_local_deviation_from_column_median(self):
        n_pings, n_samples = 10, 5
        window = np.zeros((n_pings, n_samples))
        window[3, 2] = 50.0  # one real, local anomaly

        result = detrend_by_range(window)

        assert result[3, 2] == 50.0
        assert np.all(result[np.arange(n_pings) != 3, 2] == 0.0)

    def test_output_shape_matches_input(self):
        window = np.random.rand(7, 13)
        result = detrend_by_range(window)
        assert result.shape == window.shape
