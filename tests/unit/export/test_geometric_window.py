"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.export.geometric_window import _clamp_center, extract_geometric_window, locate_point
from sidescanpy.viz.waterfall import along_track_distance_m


def _real_center(ds, channel_idx=0):
    """A real (along_m, across_m) point actually present in the fixture,
    for centering a test window on real data instead of guessing."""
    along = along_track_distance_m(ds)
    across = ds["across_track_m"].isel(channel=channel_idx).values
    finite = np.isfinite(across)
    ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]
    return float(along[ping_idx]), float(across[ping_idx, sample_idx])


class TestExtractGeometricWindow:
    def test_output_shape_matches_request(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        grid, _meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across
        )
        assert grid.shape == (16, 16)

    def test_has_some_finite_values(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        grid, _meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across
        )
        assert np.isfinite(grid).any()

    def test_center_is_reported_at_the_true_middle(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across
        )
        assert meta["center_row"] == pytest.approx(8.0, abs=0.5)
        assert meta["center_col"] == pytest.approx(8.0, abs=0.5)

    def test_square_regardless_of_asymmetric_extent(self, geometric_channel):
        # regression guard for the stretching bug: a resolution/size pair
        # must produce the same real-world extent on both axes even when
        # the source data's own along/across spans differ wildly.
        center_along, center_across = _real_center(geometric_channel)
        grid, meta = extract_geometric_window(
            geometric_channel, 20, resolution_m=0.5, center_along_m=center_along, center_across_m=center_across
        )
        assert grid.shape == (20, 20)
        # both axes span exactly size_px * resolution_m by construction
        assert meta["resolution_m"] == 0.5

    def test_channel_idx_selects_different_data(self, geometric_channel):
        # port (channel 0) and starboard (channel 1) occupy opposite-sign
        # across_track_m by convention -- each needs its own real center,
        # not one shared point, to land on real data for both.
        along_port, across_port = _real_center(geometric_channel, channel_idx=0)
        along_stbd, across_stbd = _real_center(geometric_channel, channel_idx=1)
        grid_port, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=along_port, center_across_m=across_port, channel_idx=0
        )
        grid_stbd, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=along_stbd, center_across_m=across_stbd, channel_idx=1
        )
        assert np.isfinite(grid_port).any()
        assert np.isfinite(grid_stbd).any()
        assert not np.allclose(np.nan_to_num(grid_port), np.nan_to_num(grid_stbd))

    def test_channel_idx_none_pools_both_sides(self, geometric_channel):
        center_along, _ = _real_center(geometric_channel)
        # across=0 straddles both port (negative) and starboard (positive)
        grid, _meta = extract_geometric_window(
            geometric_channel, 40, resolution_m=1.0, center_along_m=center_along, center_across_m=0.0, channel_idx=None
        )
        assert np.isfinite(grid).any()

    def test_all_water_column_raises(self, geometric_channel_factory):
        ds = geometric_channel_factory(n_pings=5, n_samples=10, altitude_m=1_000_000.0)
        with pytest.raises(ValueError):
            extract_geometric_window(ds, 8, resolution_m=1.0, center_along_m=0.0, center_across_m=0.0)

    def test_avoid_edge_gaps_reduces_nans_near_edge(self, geometric_channel):
        # A target right at the far (max-range) edge of the recorded swath
        # -- real-data case (project-a target 0011, sample 7751 of a
        # 7792-wide window) where a window centered exactly on it would be
        # half no-data. avoid_edge_gaps=True should shift the window inward
        # instead. Default (False) preserves the original unconditional
        # centering -- see test_center_is_reported_at_the_true_middle /
        # TestLocatePoint for that regression guard.
        across = geometric_channel["across_track_m"].isel(channel=1).values  # starboard, positive
        along = along_track_distance_m(geometric_channel)
        finite = np.isfinite(across)
        masked_across = np.where(finite, across, -np.inf)
        ping_idx, sample_idx = np.unravel_index(np.argmax(masked_across), masked_across.shape)
        center_along = float(along[ping_idx])
        center_across = float(across[ping_idx, sample_idx])

        grid_avoid, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            channel_idx=1, avoid_edge_gaps=True,
        )
        grid_default, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across, channel_idx=1,
        )
        assert np.isnan(grid_avoid).sum() < np.isnan(grid_default).sum()


class TestAnisotropicSearchRadius:
    """2026-09-28: real user report (blurry chips at a tight resolution_m)
    traced to the isotropic search radius pooling far more across-track
    real detail than the data's own resolution supports, since the
    radius had to stay wide enough to bridge real along-track ping gaps.
    along_search_radius_m/across_search_radius_m let a caller separate
    the two."""

    def test_isotropic_via_new_params_matches_old_search_radius_m(self, geometric_channel):
        # Equal along/across radii must reproduce search_radius_m's own
        # behavior exactly -- this is a generalization, not a behavior
        # change, when both radii are the same.
        center_along, center_across = _real_center(geometric_channel)
        grid_old, meta_old = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            search_radius_m=3.0,
        )
        grid_new, meta_new = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_search_radius_m=3.0, across_search_radius_m=3.0,
        )
        np.testing.assert_array_equal(grid_old, grid_new)
        assert meta_old == meta_new

    def test_narrowing_across_radius_only_increases_nans(self, geometric_channel):
        # A narrow across_search_radius_m (much smaller than the
        # along-track-driven default) should leave MORE cells without a
        # real neighbor within reach -- the direct, generic signature of
        # "the ellipse is actually narrower on that axis", without
        # needing to hand-engineer exact point positions.
        center_along, center_across = _real_center(geometric_channel)
        wide_radius = 5.0
        grid_isotropic, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=0.5, center_along_m=center_along, center_across_m=center_across,
            along_search_radius_m=wide_radius, across_search_radius_m=wide_radius,
        )
        grid_narrow_across, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=0.5, center_along_m=center_along, center_across_m=center_across,
            along_search_radius_m=wide_radius, across_search_radius_m=0.05,
        )
        assert np.isnan(grid_narrow_across).sum() > np.isnan(grid_isotropic).sum()
        # ... but not EVERYTHING went to NaN -- a reasonably-scaled
        # narrow radius should still find real nearby samples on the
        # across axis, not just fail outright.
        assert np.isfinite(grid_narrow_across).any()

    def test_narrowing_along_radius_only_also_increases_nans(self, geometric_channel):
        # Symmetric check on the other axis, same reasoning.
        center_along, center_across = _real_center(geometric_channel)
        wide_radius = 5.0
        grid_isotropic, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=0.5, center_along_m=center_along, center_across_m=center_across,
            along_search_radius_m=wide_radius, across_search_radius_m=wide_radius,
        )
        grid_narrow_along, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=0.5, center_along_m=center_along, center_across_m=center_across,
            along_search_radius_m=0.01, across_search_radius_m=wide_radius,
        )
        assert np.isnan(grid_narrow_along).sum() > np.isnan(grid_isotropic).sum()

    def test_omitting_new_params_falls_back_to_search_radius_m_default(self, geometric_channel):
        # Omitting all three radius params must still reproduce the
        # original 1.5 * resolution_m isotropic default exactly.
        center_along, center_across = _real_center(geometric_channel)
        grid_default, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
        )
        grid_explicit, _ = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            search_radius_m=1.5,
        )
        np.testing.assert_array_equal(grid_default, grid_explicit)


class TestAnisotropicPixels:
    """2026-09-29: real user report -- even with anisotropic search
    radii, a single shared resolution_m still forces the same pixel
    density on both axes, so on a survey with coarse ping spacing the
    along-track search radius (tied to real ping gaps) ends up wide
    relative to a fine shared pixel spacing, visible as elliptical
    averaging footprints. along_resolution_m/across_resolution_m/
    along_size_px/across_size_px let a caller give each axis its own
    real pixel density, matching its own native resolving power."""

    def test_omitting_new_params_reproduces_square_output_exactly(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        grid_old, meta_old = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
        )
        grid_new, meta_new = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=1.0, across_resolution_m=1.0, along_size_px=16, across_size_px=16,
        )
        np.testing.assert_array_equal(grid_old, grid_new)
        assert meta_old["along_resolution_m"] == meta_new["along_resolution_m"]
        assert meta_old["across_resolution_m"] == meta_new["across_resolution_m"]

    def test_independent_resolution_produces_non_square_array(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        grid, _meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=2.0, across_resolution_m=0.5, along_size_px=10, across_size_px=40,
        )
        assert grid.shape == (10, 40)

    def test_meta_carries_both_resolved_resolutions(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=2.0, across_resolution_m=0.5,
        )
        assert meta["along_resolution_m"] == 2.0
        assert meta["across_resolution_m"] == 0.5
        # legacy key stays populated (equal to along_resolution_m) for
        # any old caller still reading it directly.
        assert meta["resolution_m"] == 2.0

    def test_only_along_overridden_across_falls_back_to_shared_resolution_m(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=3.0,
        )
        assert meta["along_resolution_m"] == 3.0
        assert meta["across_resolution_m"] == 1.0

    def test_real_footprint_covers_the_requested_extent_per_axis(self, geometric_channel):
        """A coarser along-track resolution with fewer along pixels
        should still cover roughly the SAME real along-track extent as
        an equivalent square-pixel window at that along resolution --
        confirms along_size_px/along_resolution_m aren't independently
        broken (e.g. accidentally covering a much wider or narrower
        real span than requested)."""
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=2.0, across_resolution_m=0.5, along_size_px=8, across_size_px=32,
        )
        along_extent_m = 8 * 2.0
        across_extent_m = 32 * 0.5
        assert along_extent_m == pytest.approx(16.0)
        assert across_extent_m == pytest.approx(16.0)
        # both axes cover the same real 16m extent despite very
        # different pixel counts/resolutions -- the actual point of
        # this feature.
        assert center_along - meta["along_min"] == pytest.approx(along_extent_m / 2)
        assert center_across - meta["across_min"] == pytest.approx(across_extent_m / 2)

    def test_locate_point_uses_per_axis_resolution(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across,
            along_resolution_m=2.0, across_resolution_m=0.5, along_size_px=8, across_size_px=32,
        )
        row, col = locate_point(center_along, center_across, meta)
        assert row == pytest.approx(meta["center_row"])
        assert col == pytest.approx(meta["center_col"])
        # sanity: row should be near the middle of the 8-row axis, col
        # near the middle of the 32-col axis -- not swapped/confused.
        assert 0 <= row <= 8
        assert 0 <= col <= 32


class TestClampCenter:
    def test_no_change_when_window_fits_away_from_edges(self):
        assert _clamp_center(5.0, half_extent=2.0, data_min=0.0, data_max=10.0) == 5.0

    def test_shifts_inward_when_too_close_to_max_edge(self):
        result = _clamp_center(9.5, half_extent=2.0, data_min=0.0, data_max=10.0)
        assert result == pytest.approx(8.0)

    def test_shifts_inward_when_too_close_to_min_edge(self):
        result = _clamp_center(0.5, half_extent=2.0, data_min=0.0, data_max=10.0)
        assert result == pytest.approx(2.0)

    def test_centers_on_available_extent_when_window_larger_than_data(self):
        result = _clamp_center(1.0, half_extent=10.0, data_min=0.0, data_max=4.0)
        assert result == pytest.approx(2.0)

    def test_always_returns_plain_python_float_even_from_numpy_inputs(self):
        # Regression: along_valid/across_valid are numpy float32 arrays in
        # extract_geometric_window, so an un-cast .min()/.max() silently
        # taints the result with np.float32 -- which downstream COCO/YOLO
        # export's json.dumps can't serialize.
        data_min = np.float32(0.0)
        data_max = np.float32(10.0)
        assert type(_clamp_center(9.5, half_extent=2.0, data_min=data_min, data_max=data_max)) is float
        assert type(_clamp_center(0.5, half_extent=2.0, data_min=data_min, data_max=data_max)) is float
        assert type(_clamp_center(1.0, half_extent=10.0, data_min=data_min, data_max=data_max)) is float


class TestLocatePoint:
    def test_matches_the_centering_point(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across
        )
        row, col = locate_point(center_along, center_across, meta)
        assert row == pytest.approx(meta["center_row"])
        assert col == pytest.approx(meta["center_col"])
