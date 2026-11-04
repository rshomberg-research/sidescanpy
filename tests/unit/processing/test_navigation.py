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


def _track(lat, lon):
    from tests.unit.synthetic import make_raw_dataset

    ds = make_raw_dataset(n_pings=len(lat))
    return ds.assign(latitude=("ping", np.asarray(lat, dtype=np.float64)),
                     longitude=("ping", np.asarray(lon, dtype=np.float64)))


class TestCourseOverGround:
    """GPS course as a stand-in heading for an unreliable towfish compass."""

    def test_northbound_track_is_zero(self):
        from sidescanpy.processing.navigation import course_over_ground

        ds = _track(41.5 + np.arange(60) * 1e-5, np.full(60, -71.4))
        course = course_over_ground(ds)
        assert np.all(np.minimum(course, 360 - course) < 0.5)

    def test_eastbound_track_is_ninety(self):
        from sidescanpy.processing.navigation import course_over_ground

        ds = _track(np.full(60, 41.5), -71.4 + np.arange(60) * 1e-5)
        np.testing.assert_allclose(course_over_ground(ds), 90.0, atol=0.5)

    def test_turn_through_north_has_no_jump(self):
        from sidescanpy.processing.navigation import course_over_ground

        # heading 350 deg, then 10 deg: the course passes through north, not 180
        steps = np.r_[np.full(30, -10.0), np.full(30, 10.0)]
        lat = 41.5 + np.cumsum(np.cos(np.radians(steps))) * 1e-5
        lon = -71.4 + np.cumsum(np.sin(np.radians(steps))) * 1e-5
        course = course_over_ground(_track(lat, lon), half_window_s=2.0)
        assert np.all(np.minimum(course, 360 - course) < 12)

    def test_nan_positions_are_interpolated(self):
        from sidescanpy.processing.navigation import course_over_ground

        lat = 41.5 + np.arange(60) * 1e-5
        lat[20:25] = np.nan
        course = course_over_ground(_track(lat, np.full(60, -71.4)))
        assert np.all(np.isfinite(course))

    def test_stationary_track_is_nan(self, caplog):
        from sidescanpy.processing.navigation import course_over_ground

        course = course_over_ground(_track(np.full(20, 41.5), np.full(20, -71.4)))
        assert np.all(np.isnan(course))
        assert "never change" in caplog.text
