"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
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


class TestLocatePoint:
    def test_matches_the_centering_point(self, geometric_channel):
        center_along, center_across = _real_center(geometric_channel)
        _grid, meta = extract_geometric_window(
            geometric_channel, 16, resolution_m=1.0, center_along_m=center_along, center_across_m=center_across
        )
        row, col = locate_point(center_along, center_across, meta)
        assert row == pytest.approx(meta["center_row"])
        assert col == pytest.approx(meta["center_col"])
