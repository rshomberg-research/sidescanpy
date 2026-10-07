"""Tests for processing.navigation.

2026 rshomberg@uri.edu
"""

import numpy as np


class TestFillNavStalls:
    """Navigation that stalls for a few pings while the vessel keeps
    moving; geometric resampling needs those pings at their true
    spacing, not stacked at one along-track point."""

    def test_short_stall_gets_local_step(self):
        from sidescanpy.processing.navigation import fill_nav_stalls

        steps = np.array([0.07] * 30 + [0.011, 0, 0, 0, 0, 0, 0] + [0.07] * 30)
        along = np.concatenate([[0.0], np.cumsum(steps)])
        out = fill_nav_stalls(along)
        np.testing.assert_allclose(np.diff(out), 0.07)

    def test_no_stall_is_unchanged(self):
        from sidescanpy.processing.navigation import fill_nav_stalls

        along = np.cumsum(np.full(40, 0.07))
        np.testing.assert_allclose(fill_nav_stalls(along), along)

    def test_long_stop_is_left_alone(self):
        from sidescanpy.processing.navigation import fill_nav_stalls

        steps = np.array([0.07] * 30 + [0.0] * 40 + [0.07] * 30)
        along = np.concatenate([[0.0], np.cumsum(steps)])
        np.testing.assert_allclose(fill_nav_stalls(along), along)

    def test_flag_is_opt_in(self):
        import xarray as xr

        from sidescanpy.processing.navigation import along_track_distance_m

        lat = 40.0 + np.concatenate([np.arange(30), np.full(5, 29.0), 29 + np.arange(1, 31)]) * 1e-6
        ds = xr.Dataset({"latitude": ("ping", lat), "longitude": ("ping", np.full(lat.size, -70.0))})
        raw = along_track_distance_m(ds)
        filled = along_track_distance_m(ds, fill_nav_stalls=True)
        assert np.diff(raw)[30] == 0
        assert np.diff(filled).min() > 0
