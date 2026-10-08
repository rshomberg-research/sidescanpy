"""
Unit tests for sidescanpy.export.windows.geo.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.core.target import Target
from sidescanpy.export.windows.geo import geo_targets_within_window, swath_fill_extent
from tests.unit.export.surveys import geo_channel_ds


class TestGeoTargetsWithinWindow:
    def test_finds_a_co_located_target_and_places_it_locally(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.processing.navigation import along_track_distance_m

        ds = geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        co_target = Target(target_id="co", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        candidates = [(co_target, int(ping_idx))]
        found = geo_targets_within_window(candidates, ds, 0, along, 16, meta, locate_point)
        assert len(found) == 1
        assert found[0].target_id == "co"
        assert 0 <= found[0].sample < 16
        assert 0 <= found[0].extra_attributes["window_row"] < 16

    def test_target_outside_window_is_excluded(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.processing.navigation import along_track_distance_m

        ds = geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        # a target at ping 0 -- far outside a tight 16*0.5m=8m window centered mid-survey
        far_target = Target(target_id="far", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        found = geo_targets_within_window([(far_target, 0)], ds, 0, along, 16, meta, locate_point)
        assert found == []

    def test_out_of_range_ping_idx_skipped_not_crashed(self):
        from sidescanpy.export.geometric_window import extract_geometric_window, locate_point
        from sidescanpy.processing.navigation import along_track_distance_m

        ds = geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]
        grid, meta = extract_geometric_window(
            ds, 16, resolution_m=0.5, center_along_m=float(along[ping_idx]),
            center_across_m=float(across[ping_idx, sample_idx]), channel_idx=0,
        )
        bogus = Target(target_id="bogus", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        # ping_idx way outside this masked slice's own ping axis (lo=0, n_pings=20)
        found = geo_targets_within_window([(bogus, 9999)], ds, 0, along, 16, meta, locate_point)
        assert found == []

    def test_along_size_px_bounds_row_independent_of_size_px(self):
        # Real bug regression, 2026-09-29: the row bound used to reuse
        # size_px (the column count) for both axes -- a co-target whose
        # locate_fn-reported row falls inside size_px but outside a
        # smaller real along_size_px must now be excluded. A stub
        # locate_fn pins the row deterministically (row=8), avoiding any
        # dependence on this fixture's own real geometry.
        from sidescanpy.processing.navigation import along_track_distance_m

        ds = geo_channel_ds()
        along = along_track_distance_m(ds)
        across = ds["across_track_m"].isel(channel=0).values
        finite = np.isfinite(across)
        ping_idx, sample_idx = np.argwhere(finite)[len(np.argwhere(finite)) // 2]

        def stub_locate(*_args, **_kwargs):
            return 8, 2  # row=8: inside size_px=16, outside along_size_px=4

        co_target = Target(target_id="co", source_format="x", catalog_source="x", channel=0, sample=int(sample_idx))
        candidates = [(co_target, int(ping_idx))]
        found_no_limit = geo_targets_within_window(candidates, ds, 0, along, 16, {}, stub_locate)
        found_with_limit = geo_targets_within_window(candidates, ds, 0, along, 16, {}, stub_locate, along_size_px=4)
        assert len(found_no_limit) == 1  # demonstrates the stale size_px bound over-accepts
        assert found_with_limit == []


class TestSwathFillExtent:
    """Swath frame sized to where the data is."""

    def _arrays(self, n_pings=50, n_samples=100, port_reach=40.0, stbd_reach=60.0, near=2.0):
        # across: port negative, starboard positive; reach in meters
        frac = np.linspace(0.0, 1.0, n_samples)
        across = np.empty((n_pings, 2, n_samples))
        across[:, 0, :] = -(near + frac * (port_reach - near))
        across[:, 1, :] = near + frac * (stbd_reach - near)
        values = np.ones_like(across)
        return across, values

    def test_dual_swath_spans_port_far_to_starboard_far(self):
        across, values = self._arrays()
        lo, hi = swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert lo == pytest.approx(-40.0) and hi == pytest.approx(60.0)

    def test_short_pings_crop_long_ones_not_pad(self):
        across, values = self._arrays()
        values[:45, 1, 80:] = np.nan  # 90% of pings stop short on starboard
        _lo, hi = swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert hi < 60.0 * 0.82

    def test_one_sided_recording_mirrors_other_side(self):
        across, values = self._arrays()
        values[:, 0, :] = np.nan
        lo, hi = swath_fill_extent(across, values, combine=True, channel_idx=None)
        assert (lo, hi) == pytest.approx((-60.0, 60.0))

    def test_single_swath_near_to_far_per_channel(self):
        across, values = self._arrays()
        assert swath_fill_extent(across, values, combine=False, channel_idx=1) == pytest.approx((2.0, 60.0))
        assert swath_fill_extent(across, values, combine=False, channel_idx=0) == pytest.approx((-40.0, -2.0))

    def test_no_data_returns_none(self):
        across, values = self._arrays()
        values[:] = np.nan
        assert swath_fill_extent(across, values, combine=True, channel_idx=None) is None
