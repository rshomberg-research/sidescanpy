"""
Unit tests for sidescanpy.export.orientation.

2026 rshomberg@uri.edu
"""

import numpy as np

from sidescanpy.core.target import Target
from sidescanpy.export.orientation import (
    detrend_by_range,
    flip_geo_vertical,
    orient_for_display,
    orient_geo_window,
    orient_pixel_window,
)
from tests.unit.export.surveys import geo_channel_ds


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


class TestOrientPixelWindow:
    """Multi-target counterpart of orient_for_display for export windows."""

    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_vertical_flip_always_applies(self):
        window = np.array([[1, 2], [3, 4], [5, 6]])  # 3 rows, 2 cols
        ct = self._co_target(sample=0, window_row=0)
        flipped, (out,) = orient_pixel_window(window, [ct], channel_idx=1, style="single-swath")
        np.testing.assert_array_equal(flipped, window[::-1, :])
        assert out.extra_attributes["window_row"] == 2

    def test_port_channel_flips_horizontally_for_single_channel_style(self):
        window = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows, 3 cols
        ct = self._co_target(sample=0, window_row=0)
        _, (out,) = orient_pixel_window(window, [ct], channel_idx=0, style="chip")
        assert out.sample == 2

    def test_starboard_channel_does_not_flip_horizontally(self):
        window = np.array([[1, 2, 3], [4, 5, 6]])
        ct = self._co_target(sample=0, window_row=0)
        flipped, (out,) = orient_pixel_window(window, [ct], channel_idx=1, style="chip")
        assert out.sample == 0
        np.testing.assert_array_equal(flipped, window[::-1, :])

    def test_dual_swath_never_flips_horizontally_even_for_channel_0(self):
        """dual-swath windows are already port-mirrored by
        _combine_channels_mirrored at combination time -- flipping again
        here would un-mirror them."""
        window = np.array([[1, 2, 3], [4, 5, 6]])
        ct = self._co_target(sample=0, window_row=0)
        _, (out,) = orient_pixel_window(window, [ct], channel_idx=0, style="dual-swath")
        assert out.sample == 0

    def test_multiple_co_targets_all_remapped(self):
        window = np.zeros((4, 5))
        cts = [self._co_target(sample=1, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = orient_pixel_window(window, cts, channel_idx=0, style="chip")
        assert {(o.sample, o.extra_attributes["window_row"]) for o in out} == {(3, 3), (0, 0)}


class TestFlipGeoVertical:
    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_flips_grid_vertically(self):
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        flipped, _ = flip_geo_vertical(grid, [], along_size_px=3)
        np.testing.assert_array_equal(flipped, grid[::-1, :])

    def test_remaps_every_target_window_row(self):
        grid = np.zeros((10, 10))
        targets = [self._co_target(sample=1, window_row=0), self._co_target(sample=4, window_row=9)]
        _, out = flip_geo_vertical(grid, targets, along_size_px=10)
        assert {t.extra_attributes["window_row"] for t in out} == {10, 1}

    def test_uses_along_size_px_not_column_count_when_non_square(self):
        # Real bug regression, 2026-09-29: along-track pixel resolution
        # can now differ from across-track, so a grid's row count and
        # column count can genuinely differ -- the flip must mirror
        # against the actual ROW count, not whatever size_px happens to
        # be passed for the (unrelated) column axis.
        grid = np.arange(12).reshape(4, 3)  # 4 rows, 3 cols -- non-square
        targets = [self._co_target(sample=1, window_row=0), self._co_target(sample=1, window_row=4)]
        flipped, out = flip_geo_vertical(grid, targets, along_size_px=4)
        np.testing.assert_array_equal(flipped, grid[::-1, :])
        assert {t.extra_attributes["window_row"] for t in out} == {4, 0}


class TestOrientGeoWindow:
    """Columns are built in ascending across_track_m, which puts port's
    near range on the right and starboard's on the left; port windows are
    mirrored so both put near range at column 0."""

    def _co_target(self, sample, window_row):
        return Target(target_id="1", source_format="x", catalog_source="x", sample=sample, extra_attributes={"window_row": window_row})

    def test_port_gets_horizontal_flip_on_top_of_vertical(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = orient_geo_window(grid, [], size_px=2, combine=False, channel_idx=0)
        # vertical flip first (rows reversed), then horizontal (cols reversed)
        expected = grid[::-1, :][:, ::-1]
        np.testing.assert_array_equal(oriented, expected)

    def test_starboard_gets_only_vertical_flip(self):
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = orient_geo_window(grid, [], size_px=2, combine=False, channel_idx=1)
        np.testing.assert_array_equal(oriented, grid[::-1, :])

    def test_uses_along_size_px_for_vertical_flip_when_non_square(self):
        # Real bug regression, 2026-09-29: a non-square grid (anisotropic
        # along-track pixel resolution) must flip against its own real
        # row count, not size_px (the across-track/column count).
        grid = np.arange(12).reshape(4, 3)  # 4 rows, 3 cols
        targets = [self._co_target(sample=1, window_row=0)]
        oriented, out = orient_geo_window(grid, targets, size_px=3, combine=False, channel_idx=1, along_size_px=4)
        np.testing.assert_array_equal(oriented, grid[::-1, :])
        assert out[0].extra_attributes["window_row"] == 4

    def test_dual_swath_never_gets_horizontal_flip_even_with_channel_idx_zero(self):
        # combine=True always passes channel_idx=None in the real caller
        # (_extract_geo_style_window's geom_channel_idx), but this
        # function is tested directly against combine as the deciding
        # flag, not channel_idx alone -- confirms combine wins.
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        oriented, _ = orient_geo_window(grid, [], size_px=2, combine=True, channel_idx=None)
        np.testing.assert_array_equal(oriented, grid[::-1, :])

    def test_port_flip_remaps_target_column(self):
        grid = np.zeros((4, 5))
        targets = [self._co_target(sample=0, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = orient_geo_window(grid, targets, size_px=4, combine=False, channel_idx=0)
        assert {o.sample for o in out} == {4, 0}

    def test_starboard_does_not_remap_target_column(self):
        grid = np.zeros((4, 5))
        targets = [self._co_target(sample=0, window_row=0), self._co_target(sample=4, window_row=3)]
        _, out = orient_geo_window(grid, targets, size_px=4, combine=False, channel_idx=1)
        assert {o.sample for o in out} == {0, 4}

    def test_port_near_range_moves_to_column_zero_after_orientation(self):
        """The actual point of this fix, verified directly from real
        window geometry (not just the flip mechanics tested above): a
        real port window's near-range (nadir-ward) edge is at the LAST
        column before orientation (ascending across_track_m puts port's
        least-negative/nearest-zero value last) and must be at column 0
        after -- matching starboard's own, already-correct column-0-is-
        near-range convention (starboard's smallest/nearest-zero value
        is ascending-order-first already, so it needs no flip)."""
        from sidescanpy.export.geometric_window import extract_geometric_window
        from sidescanpy.processing.navigation import along_track_distance_m

        ds = geo_channel_ds()
        along = along_track_distance_m(ds)
        across_port = ds["across_track_m"].isel(channel=0).values
        finite_port = np.isfinite(across_port)
        p_ping, p_sample = np.argwhere(finite_port)[len(np.argwhere(finite_port)) // 2]
        port_across_m = float(across_port[p_ping, p_sample])
        size_px = 16
        resolution_m = 0.5
        grid_port, meta_port = extract_geometric_window(
            ds, size_px, resolution_m=resolution_m, center_along_m=float(along[p_ping]),
            center_across_m=port_across_m, channel_idx=0,
        )
        # Pre-flip: column 0's real across_track_m is meta["across_min"]
        # (most negative = far-range for port); the last column is
        # across_min + (size_px-1)*resolution_m (least negative = near-range).
        across_min = meta_port["across_min"]
        pre_flip_col0_across_m = across_min
        pre_flip_last_col_across_m = across_min + (size_px - 1) * resolution_m
        assert abs(pre_flip_last_col_across_m) < abs(pre_flip_col0_across_m), (
            "test setup assumption broken: expected port's near-range (closer to 0) "
            "to be the LAST column before orientation"
        )

        oriented, _ = orient_geo_window(grid_port, [], size_px=size_px, combine=False, channel_idx=0)
        # After the horizontal flip, oriented column 0 == pre-flip (vertically-flipped) last column.
        np.testing.assert_array_equal(oriented[:, 0], grid_port[::-1, :][:, -1])
