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


def _track_ds(n_pings=600, ping_s=0.11, speed=2.0, error_m=None):
    """Straight north-east track at ``speed`` m/s, pings every ``ping_s``;
    ``error_m(t)`` adds an along-track position error (m)."""
    import pandas as pd

    from tests.unit.synthetic import make_raw_dataset

    ds = make_raw_dataset(n_pings=n_pings, n_samples=400)
    t = np.arange(n_pings) * ping_s
    along = speed * t + (error_m(t) if error_m is not None else 0.0)
    k = 111_320.0
    lat = 41.5 + along * np.cos(np.radians(45)) / k
    lon = -71.5 + along * np.sin(np.radians(45)) / (k * np.cos(np.radians(41.5)))
    times = pd.Timestamp("2026-01-01").to_datetime64() + (t * 1e9).astype("timedelta64[ns]")
    return ds.assign_coords(time=("ping", times)).assign(
        latitude=("ping", lat), longitude=("ping", lon)
    )


def _along_m(ds):
    k = 111_320.0
    north = (ds["latitude"].values - 41.5) * k
    east = (ds["longitude"].values + 71.5) * k * np.cos(np.radians(41.5))
    return (north + east) / np.sqrt(2)


class TestSmoothNavigation:
    """Positions with a periodic error between fixes (a recorder that runs
    ahead of each GPS fix and jumps back when the next arrives)."""

    def test_sawtooth_removed_when_window_spans_whole_periods(self):
        from sidescanpy.processing.navigation import smooth_navigation

        sawtooth = lambda t: 1.5 * (t % 1.0)  # noqa: E731
        ds = _track_ds(error_m=sawtooth)
        out = smooth_navigation(ds, window_s=5.0)
        along = _along_m(out)[60:-60]
        t = np.arange(ds.sizes["ping"])[60:-60] * 0.11
        residual = along - np.polyval(np.polyfit(t, along, 1), t)
        assert np.abs(residual).max() < 0.1
        raw = _along_m(ds)[60:-60]
        assert np.abs(raw - np.polyval(np.polyfit(t, raw, 1), t)).max() > 0.6

    def test_clean_straight_track_unchanged(self):
        from sidescanpy.processing.navigation import smooth_navigation

        ds = _track_ds()
        out = smooth_navigation(ds, window_s=5.0)
        np.testing.assert_allclose(_along_m(out), _along_m(ds), atol=1e-3)

    def test_ends_keep_their_position(self):
        from sidescanpy.processing.navigation import smooth_navigation

        ds = _track_ds(error_m=lambda t: 1.5 * (t % 1.0))
        out = smooth_navigation(ds, window_s=5.0)
        for i in (0, -1):
            assert out["latitude"].values[i] == ds["latitude"].values[i]
            assert out["longitude"].values[i] == ds["longitude"].values[i]

    def test_nan_positions_stay_nan_and_are_ignored(self):
        from sidescanpy.processing.navigation import smooth_navigation

        ds = _track_ds()
        lat = ds["latitude"].values.copy()
        lat[100:103] = np.nan
        ds = ds.assign(latitude=("ping", lat))
        out = smooth_navigation(ds, window_s=5.0)
        assert np.isnan(out["latitude"].values[100:103]).all()
        assert np.isfinite(out["latitude"].values[np.r_[0:100, 103:600]]).all()
        keep = np.r_[0:100, 103:600]
        np.testing.assert_allclose(_along_m(out)[keep], _along_m(ds)[keep], atol=1e-3)

    def test_records_history_and_keeps_input(self):
        from sidescanpy.core.history import read_history
        from sidescanpy.processing.navigation import smooth_navigation

        ds = _track_ds(error_m=lambda t: 1.5 * (t % 1.0))
        before = ds["latitude"].values.copy()
        out = smooth_navigation(ds, window_s=3.0)
        np.testing.assert_array_equal(ds["latitude"].values, before)
        record = read_history(out)[-1]
        assert record["step"] == "smooth_navigation"
        assert record["params"]["window_s"] == 3.0
        assert out["latitude"].dtype == np.float64

    def test_rejects_georeferenced_dataset(self):
        import pytest

        from sidescanpy.core.history import STEP_GEOREF, append_history
        from sidescanpy.processing.navigation import smooth_navigation

        ds = _track_ds()
        ds = ds.assign(latitude_per_sample=(("ping", "channel", "sample"),
                                            np.zeros((ds.sizes["ping"], 2, ds.sizes["sample"]))))
        ds = append_history(ds, STEP_GEOREF, {}, function="f", version="0")
        with pytest.raises(ValueError, match="georef"):
            smooth_navigation(ds)

    def test_rejects_non_positive_window(self):
        import pytest

        from sidescanpy.processing.navigation import smooth_navigation

        with pytest.raises(ValueError, match="window_s"):
            smooth_navigation(_track_ds(), window_s=0.0)

    def test_survey_smoothed_per_frequency(self):
        from sidescanpy.core import SideScanSurvey
        from sidescanpy.processing.navigation import smooth_navigation

        survey = SideScanSurvey(source_files=["a.xtf"], source_format="XTF", survey_name="a")
        survey.add_frequency("hz_400000", _track_ds(error_m=lambda t: 1.5 * (t % 1.0)))
        out = smooth_navigation(survey)
        assert isinstance(out, SideScanSurvey) and out is not survey
        shift_m = 111_320.0 * np.abs(out.get_frequency("hz_400000").dataset["latitude"].values
                                     - survey.get_frequency("hz_400000").dataset["latitude"].values)
        assert shift_m.max() > 0.2
