"""
Unit tests for sidescanpy.targets.waterfall.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.processing.navigation import along_track_distance_m
from sidescanpy.targets.waterfall import (
    WaterfallBox,
    mean_sample_spacing_m,
    median_ping_spacing_m,
    target_waterfall_position,
    waterfall_boxes,
)
from tests.unit.export.surveys import geo_channel_ds


class TestTargetWaterfallPosition:
    def test_port_channel_is_negative_x(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=300)
        x, y, x_scale, y_scale = target_waterfall_position(t, None, 57, 0, "sample", "ping")
        assert x == -300.0
        assert y == 57.0
        assert (x_scale, y_scale) == (1.0, 1.0)

    def test_starboard_channel_is_positive_x(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=1, sample=300)
        x, y, x_scale, y_scale = target_waterfall_position(t, None, 57, 0, "sample", "ping")
        assert x == 300.0
        assert y == 57.0
        assert (x_scale, y_scale) == (1.0, 1.0)

    def test_missing_channel_defaults_to_port(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=None, sample=300)
        x, _y, _xs, _ys = target_waterfall_position(t, None, 0, 0, "sample", "ping")
        assert x == -300.0

    def test_y_is_the_global_ping_index_unmodified(self):
        """The whole point of this helper -- plot_waterfall's own
        ping_offset already bakes the segment's start into the y-axis,
        so a target's file-global ping_idx must NOT be re-based to the
        segment here (that was a real bug caught before this shipped)."""
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=0)
        _x, y, _xs, _ys = target_waterfall_position(t, None, 4321, 0, "sample", "ping")
        assert y == 4321.0

    def test_ground_mode_reads_across_track_at_targets_own_position(self):
        """x_axis="across_track"/y_axis="along_track" (--range-mode
        ground): x comes from the real across_track_m value at the
        target's own (local ping, channel, sample) -- not derivable from
        sample index alone the way slant mode's x is."""
        across_track = xr.DataArray(
            np.arange(2 * 2 * 10, dtype=np.float64).reshape(2, 2, 10),
            dims=("ping", "channel", "sample"),
        )
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=1, sample=3)
        # ping_offset=100, target's global ping_idx=101 -> local ping 1
        x, y, _xs, _ys = target_waterfall_position(t, ds, 101, 100, "across_track", "along_track", along_track=np.array([0.0, 12.5]))
        assert x == float(across_track.isel(ping=1, channel=1, sample=3).values)
        assert y == 12.5

    def test_ground_mode_scale_factors_are_real_meters_not_one(self):
        """The bug this whole helper's scale-factor return exists to fix:
        a shared --box-size (a sample/ping pixel count) must NOT be
        applied as literal meters in ground mode -- x_scale/y_scale carry
        the real meters-per-sample/meters-per-ping conversion so the
        caller can size the box correctly."""
        # across_track steps by 2.0m per sample (0, 2, 4, ...) on this ping/channel.
        across_track = xr.DataArray(
            (np.arange(10, dtype=np.float64) * 2.0).reshape(1, 1, 10),
            dims=("ping", "channel", "sample"),
        )
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        # along_track steps by 5.0m per ping.
        along_track = np.array([0.0, 5.0, 10.0])
        _x, _y, x_scale, y_scale = target_waterfall_position(t, ds, 100, 100, "across_track", "along_track", along_track=along_track)
        assert x_scale == pytest.approx(2.0)
        assert y_scale == pytest.approx(5.0)

    def test_ground_mode_out_of_window_ping_returns_none(self):
        ds = xr.Dataset({"across_track_m": xr.DataArray(np.zeros((2, 2, 10)), dims=("ping", "channel", "sample"))})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert target_waterfall_position(t, ds, 999, 100, "across_track", "along_track", along_track=np.array([0.0, 1.0])) is None

    def test_ground_mode_nan_across_track_returns_none(self):
        """A target still in the water column at its own position has no
        real across-track value -- skip its box rather than drawing one
        at a garbage/NaN location."""
        across_track = xr.DataArray(np.full((1, 2, 10), np.nan), dims=("ping", "channel", "sample"))
        ds = xr.Dataset({"across_track_m": across_track})
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert target_waterfall_position(t, ds, 100, 100, "across_track", "along_track", along_track=np.array([0.0])) is None

    def test_unsupported_axis_combo_returns_none(self):
        t = Target(target_id="1", source_format="x", catalog_source="x", channel=0, sample=3)
        assert target_waterfall_position(t, None, 0, 0, "ground_range", "time") is None


class TestWaterfallBoxes:
    def _t(self, tid, channel, sample):
        return Target(target_id=tid, source_format="x", catalog_source="x", channel=channel, sample=sample)

    def test_pixel_boxes_are_centered_and_labeled(self):
        boxes = waterfall_boxes(None, [(self._t("a", 0, 10), 5), (self._t("b", 1, 3), 7)], 0, "sample", "ping", 4.0)
        assert boxes == [WaterfallBox("a", -12.0, 3.0, 4.0, 4.0), WaterfallBox("b", 1.0, 5.0, 4.0, 4.0)]

    def test_unplaceable_targets_are_dropped(self):
        assert waterfall_boxes(None, [(self._t("a", 0, None), 5)], 0, "sample", "ping", 4.0) == []

    def test_ground_boxes_scale_to_meters(self):
        ds = geo_channel_ds()
        sample = int(np.argwhere(np.isfinite(ds["across_track_m"].isel(ping=5, channel=1).values))[0][0]) + 10
        (box,) = waterfall_boxes(ds, [(self._t("a", 1, sample), 105)], 100, "across_track", "along_track", 10.0)
        along = along_track_distance_m(ds)
        assert box.width == pytest.approx(10.0 * mean_sample_spacing_m(ds, 5, 1))
        assert box.height == pytest.approx(10.0 * median_ping_spacing_m(along))


def test_median_ping_spacing_ignores_a_turn_jump():
    assert median_ping_spacing_m(np.array([0.0, 1.0, 2.0, 50.0, 51.0])) == 1.0
    assert median_ping_spacing_m(None) == 1.0
