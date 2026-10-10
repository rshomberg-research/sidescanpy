"""
Unit tests for sidescanpy.targets.correlate (position resolution and
clock-offset calibration against synthetic ``.ss.h5`` files).

2026 rshomberg@uri.edu
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from sidescanpy.core.target import Target
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.io.sdf.targets import read_klein_targets
from sidescanpy.targets.correlate import (
    calibrate_clock_offset,
    correlate_target,
    project_target_to_frequency,
    resolve_target_position,
    resolve_targets_by_catalog_ping,
    resolve_targets_nearest_ping,
    resolve_targets_offset_corrected,
)
from tests.unit.io.sdf.test_targets import _make_window, _write_target


class TestResolveTargetPosition:
    def _make_survey(self):
        from sidescanpy.core.survey import SideScanSurvey
        from tests.conftest import make_minimal_sidescan_dataset

        ds = make_minimal_sidescan_dataset(n_pings=10)
        survey = SideScanSurvey(source_files=["a.sdf"], source_format="SDF", survey_name="a")
        survey.add_frequency("hz_900000", ds)
        return survey, ds

    def test_sets_position_from_nearest_ping(self, tmp_path):
        window = _make_window()
        _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=4, range_m=10.0, window=window)
        target = read_klein_targets(tmp_path)[0]

        survey, ds = self._make_survey()
        times_s = ds["time"].values.astype("datetime64[s]").astype(np.int64)
        target = replace(target, pick_timestamp=float(times_s[3]))

        resolved = resolve_target_position(target, survey, "hz_900000")
        assert resolved.latitude == pytest.approx(float(ds["latitude"].values[3]))
        assert resolved.longitude == pytest.approx(float(ds["longitude"].values[3]))
        assert resolved.position_confidence == "unknown"  # never upgraded

    def test_leaves_position_unset_beyond_max_time_delta(self, tmp_path):
        window = _make_window()
        _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=4, range_m=10.0, window=window)
        target = read_klein_targets(tmp_path)[0]

        survey, ds = self._make_survey()
        times_s = ds["time"].values.astype("datetime64[s]").astype(np.int64)
        far_timestamp = float(times_s[0]) - 10_000
        target = replace(target, pick_timestamp=far_timestamp)

        resolved = resolve_target_position(target, survey, "hz_900000", max_time_delta_s=300.0)
        assert resolved.latitude is None
        assert resolved.longitude is None

    def test_no_pick_timestamp_returns_unchanged(self, tmp_path):
        window = _make_window()
        _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=4, range_m=10.0, window=window)
        target = read_klein_targets(tmp_path)[0]
        target = replace(target, pick_timestamp=None)

        survey, _ds = self._make_survey()
        resolved = resolve_target_position(target, survey, "hz_900000")
        assert resolved.latitude is None


# ============================================================
# Clock-offset calibration -- real-world offsets are on the order of
# ~2798 days, but the math only cares about the relative shift, so a
# small synthetic offset exercises the same code path.
# ============================================================

def _write_ss_h5_with_corrupted_time(
    directory: Path,
    true_local_start: datetime,
    corrupted_utc_start,
    *,
    n_pings: int = 10,
    altitude_m: float = 5.0,
    filename_prefix: str = "sonar_data",
) -> Path:
    from sidescanpy.core.survey import SideScanSurvey
    from tests.conftest import make_minimal_sidescan_dataset

    ds = make_minimal_sidescan_dataset(n_pings=n_pings)
    corrupted_times = pd.date_range(corrupted_utc_start, periods=n_pings, freq="1s").values.astype("datetime64[ns]")
    ds = ds.assign_coords(time=("ping", corrupted_times))
    ds["altitude_m"] = ("ping", np.full(n_pings, altitude_m, dtype=np.float32))

    survey = SideScanSurvey(source_files=["a.sdf"], source_format="SDF", survey_name="a")
    survey.add_frequency("hz_900000", ds)

    path = directory / f"{filename_prefix}{true_local_start:%y%m%d%H%M%S}.ss.h5"
    write_survey(survey, path)
    return path


class TestCalibrateClockOffset:
    _LOCAL_UTC_OFFSET_HOURS = 4.0

    def test_matches_known_offset(self, tmp_path):
        true_local_start = datetime(2018, 9, 14, 7, 20, 0)
        true_utc = true_local_start.replace(tzinfo=UTC) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
        known_offset_s = 500.0
        corrupted_utc_start = pd.Timestamp(true_utc) + pd.Timedelta(seconds=known_offset_s)
        _write_ss_h5_with_corrupted_time(tmp_path, true_local_start, corrupted_utc_start)

        offset_s = calibrate_clock_offset(tmp_path, local_utc_offset_hours=self._LOCAL_UTC_OFFSET_HOURS)
        assert offset_s == pytest.approx(known_offset_s, abs=1.0)

    def test_median_across_multiple_files(self, tmp_path):
        base_local = datetime(2018, 9, 14, 7, 20, 0)
        _true_utc = base_local.replace(tzinfo=UTC) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
        for i, offset in enumerate([400.0, 500.0, 600.0]):
            local_start = base_local + timedelta(minutes=i)
            this_true_utc = local_start.replace(tzinfo=UTC) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
            corrupted_start = pd.Timestamp(this_true_utc) + pd.Timedelta(seconds=offset)
            _write_ss_h5_with_corrupted_time(tmp_path, local_start, corrupted_start, filename_prefix=f"sonar_data_{i}_")

        offset_s = calibrate_clock_offset(tmp_path, local_utc_offset_hours=self._LOCAL_UTC_OFFSET_HOURS)
        assert offset_s == pytest.approx(500.0, abs=1.0)  # median of 400/500/600

    def test_raises_when_no_filename_matches_pattern(self, tmp_path):
        (tmp_path / "not_a_matching_name.ss.h5").write_bytes(b"")
        with pytest.raises(RuntimeError):
            calibrate_clock_offset(tmp_path, local_utc_offset_hours=4.0)

    def test_local_utc_offset_is_required(self, tmp_path):
        with pytest.raises(TypeError):
            calibrate_clock_offset(tmp_path)


class TestResolveTargetsOffsetCorrected:
    _LOCAL_UTC_OFFSET_HOURS = 4.0

    def _make_calibrated_survey(self, tmp_path, known_offset_s=500.0):
        true_local_start = datetime(2018, 9, 14, 7, 20, 0)
        true_utc = true_local_start.replace(tzinfo=UTC) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
        corrupted_utc_start = pd.Timestamp(true_utc) + pd.Timedelta(seconds=known_offset_s)
        _write_ss_h5_with_corrupted_time(tmp_path, true_local_start, corrupted_utc_start, n_pings=20, altitude_m=8.0)
        return true_utc, known_offset_s

    def test_enriches_position_altitude_and_ground_range(self, tmp_path):
        true_utc, offset_s = self._make_calibrated_survey(tmp_path)
        # A pick 5s into the true-clock survey -- shifted by offset_s
        # should land near ping index 5 in the corrupted .ss.h5 domain.
        pick_timestamp = true_utc.timestamp() + 5.0
        target = Target(
            target_id="1", source_format="sdf_klein", catalog_source="x.tgt",
            pick_timestamp=pick_timestamp, channel=0, sample=10,
            position_confidence="unknown", extra_attributes={"range_m": 10.0},
        )

        resolved = resolve_targets_offset_corrected([target], tmp_path, offset_s, max_time_delta_s=300.0)[0]
        assert resolved.latitude is not None
        assert resolved.extra_attributes["altitude_m"] == pytest.approx(8.0)
        assert resolved.extra_attributes["position_source"] == "offset_corrected_nearest_ping_timestamp"
        # ground_range = sqrt(10^2 - 8^2) = 6.0
        assert resolved.extra_attributes["ground_range_m"] == pytest.approx(6.0, abs=1e-6)
        assert resolved.position_confidence == "unknown"  # never upgraded

    def test_unmatched_target_returned_unchanged(self, tmp_path):
        true_utc, offset_s = self._make_calibrated_survey(tmp_path)
        far_pick_timestamp = true_utc.timestamp() - 100_000  # way outside any file's window
        target = Target(
            target_id="2", source_format="sdf_klein", catalog_source="x.tgt",
            pick_timestamp=far_pick_timestamp, channel=0, sample=10,
            position_confidence="unknown", extra_attributes={},
        )

        resolved = resolve_targets_offset_corrected([target], tmp_path, offset_s, max_time_delta_s=1.0)[0]
        assert resolved.latitude is None

    def test_no_pick_timestamp_returned_unchanged(self, tmp_path):
        true_utc, offset_s = self._make_calibrated_survey(tmp_path)
        target = Target(
            target_id="3", source_format="sdf_klein", catalog_source="x.tgt",
            pick_timestamp=None, channel=0, sample=10,
            position_confidence="unknown", extra_attributes={},
        )
        resolved = resolve_targets_offset_corrected([target], tmp_path, offset_s)[0]
        assert resolved.latitude is None


class TestResolveTargetsNearestPing:
    def test_matches_closest_of_several_files(self, tmp_path):
        from sidescanpy.core.survey import SideScanSurvey
        from tests.conftest import make_minimal_sidescan_dataset

        def _write_plain(name, start_local):
            ds = make_minimal_sidescan_dataset(n_pings=10)
            times = pd.date_range(start_local, periods=10, freq="1s").values.astype("datetime64[ns]")
            ds = ds.assign_coords(time=("ping", times))
            survey = SideScanSurvey(source_files=["a.sdf"], source_format="SDF", survey_name=name)
            survey.add_frequency("hz_900000", ds)
            path = tmp_path / f"{name}.ss.h5"
            write_survey(survey, path)
            return ds

        _write_plain("far_file", datetime(2020, 1, 1))
        near_ds = _write_plain("near_file", datetime(2021, 6, 1))

        near_times_s = near_ds["time"].values.astype("datetime64[s]").astype(np.int64)
        target = Target(
            target_id="1", source_format="sdf_klein", catalog_source="x.tgt",
            pick_timestamp=float(near_times_s[3]), channel=0, sample=5,
            position_confidence="unknown", extra_attributes={},
        )

        resolved = resolve_targets_nearest_ping([target], tmp_path, max_time_delta_s=300.0)[0]
        assert resolved.latitude == pytest.approx(float(near_ds["latitude"].values[3]))
        assert resolved.longitude == pytest.approx(float(near_ds["longitude"].values[3]))


class TestProjectTargetToFrequency:
    def _target(self, **overrides):
        defaults = dict(
            target_id="1", source_format="jsf_discover", catalog_source="x",
            ping=100, channel=1, sample=500, position_confidence="verified",
        )
        defaults.update(overrides)
        return Target(**defaults)

    def test_converts_via_physical_slant_range(self):
        # HF: 0.0137 m/sample: sample 500 -> 6.85m. LF: 0.0233 m/sample
        # -> 6.85 / 0.0233 = 294 (rounded).
        t = self._target(sample=500)
        result = project_target_to_frequency(t, source_mps=0.0137, target_mps=0.0233)
        assert result.sample == round(500 * 0.0137 / 0.0233)

    def test_ping_and_channel_unchanged(self):
        t = self._target(ping=4321, channel=0)
        result = project_target_to_frequency(t, source_mps=0.01, target_mps=0.02)
        assert result.ping == 4321
        assert result.channel == 0

    def test_records_source_mps_in_extra_attributes(self):
        t = self._target()
        result = project_target_to_frequency(t, source_mps=0.0137, target_mps=0.0233)
        assert result.extra_attributes["projected_from_mps"] == 0.0137

    def test_identity_when_source_equals_target_mps(self):
        t = self._target(sample=777)
        result = project_target_to_frequency(t, source_mps=0.02, target_mps=0.02)
        assert result.sample == 777

    def test_none_sample_returns_none(self):
        t = self._target(sample=None)
        assert project_target_to_frequency(t, source_mps=0.01, target_mps=0.02) is None

    def test_out_of_bounds_projection_returns_none_not_clamped(self):
        # A target near the source channel's own far range can project
        # past a shorter-range target channel's own recorded window --
        # must be reported as unavailable, not silently clamped in.
        t = self._target(sample=4000)
        result = project_target_to_frequency(t, source_mps=0.02, target_mps=0.02, n_samples=3896)
        assert result is None

    def test_within_bounds_projection_not_none(self):
        t = self._target(sample=100)
        result = project_target_to_frequency(t, source_mps=0.02, target_mps=0.02, n_samples=3896)
        assert result is not None

    def test_n_samples_none_skips_bounds_check(self):
        t = self._target(sample=999999)
        result = project_target_to_frequency(t, source_mps=1.0, target_mps=1.0, n_samples=None)
        assert result is not None


class TestCorrelateTarget:
    """Position match first, then fall back to the target's own ping."""

    def _index(self, *, lat, lon, n_pings, freq_key="hz_1495000", sample_interval_m=0.05, n_samples=1000, ping_start=0):
        return {
            Path("a.ss.h5"): {
                freq_key: {
                    "lat": np.asarray(lat, dtype=float),
                    "lon": np.asarray(lon, dtype=float),
                    "sample_interval_m": sample_interval_m,
                    "n_samples": n_samples,
                    "n_pings": n_pings,
                    "ping_start": ping_start,
                }
            }
        }

    def test_gps_match_within_range_unchanged(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=float(lat[10]), longitude=float(lon[10]))

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 10
        assert dist_m < 1.0

    def test_east_west_offset_uses_geodesic_distance(self):
        """A target 20 m east of a north-south track at 41N is 20 m away,
        not 20/cos(41) = 26.5 m (the old degrees-times-111 km error)."""
        lat = np.linspace(41.0, 41.002, 41)
        lon = np.full(41, -71.0)
        index = self._index(lat=lat, lon=lon, n_pings=41)
        geod = Geod(ellps="WGS84")
        t_lon, t_lat, _ = geod.fwd(-71.0, float(lat[20]), 90.0, 20.0)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=float(t_lat), longitude=float(t_lon))

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        assert result.ping_idx == 20
        assert result.dist_m == pytest.approx(20.0, abs=0.01)

    def test_pings_without_fix_are_skipped(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        lat[:5] = np.nan
        lon[:5] = np.nan
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=float(lat[10]), longitude=float(lon[10]))

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        assert result.ping_idx == 10

    def test_no_gps_falls_back_to_known_ping(self):
        """The exact reported bug: no recorded GPS position at all, but a
        real, already-known ping -- must now match instead of being
        silently dropped."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 7
        assert dist_m == 0.0

    def test_gps_too_far_falls_back_to_known_ping(self):
        """Recorded GPS position exists but doesn't land within
        max_dist_m of the survey's own track (e.g. a less precise/
        independent positioning system) -- ping fallback still rescues
        it rather than dropping it."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=41.0, longitude=-71.0, ping=7)

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        assert result[2] == 7
        assert result[3] == 0.0

    def test_ping_out_of_range_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=500)

        assert correlate_target(t, index, max_dist_m=25.0) is None

    def test_no_position_at_all_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=None)

        assert correlate_target(t, index, max_dist_m=25.0) is None

    def test_ambiguous_ping_across_multiple_files_returns_none_not_a_guess(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20)
        index[Path("b.ss.h5")] = {
            "hz_1495000": {
                "lat": lat.copy(), "lon": lon.copy(), "sample_interval_m": 0.05, "n_samples": 1000,
                "n_pings": 20, "ping_start": 0,
            },
        }
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        assert correlate_target(t, index, max_dist_m=25.0) is None

    def test_ping_fallback_uses_global_ping_start_not_local_zero_based_index(self):
        """A file's ping axis need not start at 0 (globally renumbered
        files), so a target's global ping must be offset by ``ping_start``."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        # This file covers mission-global pings [100, 120) -- NOT [0, 20).
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=100)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=107)

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert ping_idx == 7  # 107 - ping_start(100) -- the correct LOCAL array index
        assert dist_m == 0.0

    def test_ping_fallback_global_ping_before_this_files_range_returns_none(self):
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=100)
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=7)

        assert correlate_target(t, index, max_dist_m=25.0) is None

    def test_ping_fallback_across_two_globally_offset_files_picks_the_right_one(self):
        """The real multi-part scenario: two files, each internally
        zero-based-*looking* by n_pings alone, but covering disjoint
        global ping ranges -- exactly one must claim a given global
        ping, at the right LOCAL index within it."""
        lat = np.linspace(40.0, 40.001, 20)
        lon = np.linspace(-70.0, -70.001, 20)
        index = self._index(lat=lat, lon=lon, n_pings=20, ping_start=0)
        index[Path("b.ss.h5")] = {
            "hz_1495000": {
                "lat": lat.copy(), "lon": lon.copy(), "sample_interval_m": 0.05, "n_samples": 1000,
                "n_pings": 20, "ping_start": 20,
            },
        }
        t = Target(target_id="1", source_format="x", catalog_source="x", latitude=None, longitude=None, ping=25)

        result = correlate_target(t, index, max_dist_m=25.0)

        assert result is not None
        path, freq_key, ping_idx, dist_m = result
        assert path == Path("b.ss.h5")
        assert ping_idx == 5  # 25 - ping_start(20)


# ============================================================
# Catalog-ping resolution (Klein): match the catalog's own ping number and
# TPU time, which share the clock of the converted pings -- no offset.
# ============================================================

def _write_ping_clock_ss_h5(directory: Path, name: str, start, first_ping_number: int, *, n_pings: int = 100,
                            altitude_m: float = 8.0, lat0: float = 44.0) -> Path:
    from sidescanpy.core.survey import SideScanSurvey
    from tests.conftest import make_minimal_sidescan_dataset

    ds = make_minimal_sidescan_dataset(n_pings=n_pings)
    times = pd.date_range(start, periods=n_pings, freq="1s").values.astype("datetime64[ns]")
    ds = ds.assign_coords(time=("ping", times),
                          ping_number=("ping", np.arange(first_ping_number, first_ping_number + n_pings, dtype=np.int32)))
    ds["latitude"] = ("ping", lat0 + np.arange(n_pings) * 1e-5)
    ds["altitude_m"] = ("ping", np.full(n_pings, altitude_m, dtype=np.float32))
    survey = SideScanSurvey(source_files=[f"{name}.sdf"], source_format="SDF", survey_name=name)
    survey.add_frequency("hz_900000", ds)
    path = directory / f"{name}.ss.h5"
    write_survey(survey, path)
    return path


def _catalog_target(ping_time, ping_number, *, pick_timestamp=1536923484.0, range_m=10.0):
    return Target(
        target_id="t", source_format="sdf_klein", catalog_source="x.tgt", pick_timestamp=pick_timestamp,
        channel=0, sample=10, position_confidence="unknown",
        extra_attributes={"range_m": range_m, "catalog_ping_time": ping_time, "catalog_ping_number": ping_number},
    )


class TestResolveTargetsByCatalogPing:
    _START = pd.Timestamp("2026-05-13 14:50:00", tz="UTC")

    def test_matches_catalog_ping_exactly(self, tmp_path):
        _write_ping_clock_ss_h5(tmp_path, "sonar_data180914072000", self._START.tz_localize(None), 61650)
        t = _catalog_target((self._START + pd.Timedelta(seconds=65)).timestamp(), 61715)
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.latitude == pytest.approx(44.0 + 65 * 1e-5)
        ea = r.extra_attributes
        assert ea["position_source"] == "catalog_ping"
        assert ea["ping_index"] == 65
        assert ea["ss_h5"] == "sonar_data180914072000"
        assert ea["altitude_m"] == pytest.approx(8.0)
        assert ea["ground_range_m"] == pytest.approx(6.0, abs=1e-6)  # sqrt(10^2 - 8^2)

    def test_pick_timestamp_is_ignored(self, tmp_path):
        _write_ping_clock_ss_h5(tmp_path, "a", self._START.tz_localize(None), 61650)
        t = _catalog_target((self._START + pd.Timedelta(seconds=10)).timestamp(), 61660, pick_timestamp=0.0)
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.extra_attributes["ping_index"] == 10

    def test_repeated_ping_numbers_disambiguated_by_time(self, tmp_path):
        # The ping counter resets after a recording gap: both files contain ping 120.
        _write_ping_clock_ss_h5(tmp_path, "a", self._START.tz_localize(None), 100, lat0=44.0)
        later = self._START + pd.Timedelta(hours=1)
        _write_ping_clock_ss_h5(tmp_path, "b", later.tz_localize(None), 100, lat0=45.0)
        t = _catalog_target((later + pd.Timedelta(seconds=20)).timestamp(), 120)
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.extra_attributes["ss_h5"] == "b"
        assert r.latitude == pytest.approx(45.0 + 20 * 1e-5)

    def test_time_outside_every_file_returns_unchanged(self, tmp_path):
        # e.g. picked from the live display while not logging
        _write_ping_clock_ss_h5(tmp_path, "a", self._START.tz_localize(None), 61650)
        t = _catalog_target((self._START + pd.Timedelta(hours=2)).timestamp(), 61715)
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.latitude is None

    def test_ping_number_mismatch_within_time_window_uses_nearest_time(self, tmp_path):
        _write_ping_clock_ss_h5(tmp_path, "a", self._START.tz_localize(None), 61650)
        t = _catalog_target((self._START + pd.Timedelta(seconds=30)).timestamp(), 99999)
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.extra_attributes["ping_index"] == 30
        assert r.extra_attributes["position_source"] == "catalog_ping_time"

    def test_target_without_catalog_fields_returned_unchanged(self, tmp_path):
        _write_ping_clock_ss_h5(tmp_path, "a", self._START.tz_localize(None), 61650)
        t = replace(_catalog_target(None, None), extra_attributes={"range_m": 10.0})
        r = resolve_targets_by_catalog_ping([t], tmp_path)[0]
        assert r.latitude is None


class TestCorrelateTargetResolvedPing:
    """A target placed by resolve_targets_by_catalog_ping carries its exact file
    and ping; correlation uses them instead of the nearest vessel position
    (fixes repeat over several pings, so the nearest position is ambiguous)."""

    def _index(self, n_pings=20):
        lat = np.repeat(np.linspace(44.0, 44.001, n_pings // 10), 10)  # each fix held for 10 pings
        lon = np.full(n_pings, -68.9)
        return {Path("dir/f1.ss.h5"): {"hz_900000": {
            "lat": lat, "lon": lon, "sample_interval_m": 0.024, "n_samples": 2500, "n_pings": n_pings, "ping_start": 0}}}

    def _target(self, **ea):
        return Target(target_id="t", source_format="sdf_klein", catalog_source="x", channel=0, sample=100,
                      latitude=44.0, longitude=-68.9, position_confidence="approximate", extra_attributes=ea)

    def test_resolved_file_and_ping_used_exactly(self):
        m = correlate_target(self._target(ss_h5="f1", ping_index=7), self._index(), max_dist_m=25.0)
        assert (m.path.name, m.ping_idx, m.dist_m) == ("f1.ss.h5", 7, 0.0)

    def test_without_resolved_ping_nearest_position_picks_first_of_repeated_fix(self):
        m = correlate_target(self._target(), self._index(), max_dist_m=25.0)
        assert m.ping_idx == 0

    def test_resolved_file_absent_falls_back_to_position(self):
        m = correlate_target(self._target(ss_h5="other", ping_index=7), self._index(), max_dist_m=25.0)
        assert m.ping_idx == 0

    def test_resolved_ping_out_of_range_falls_back(self):
        m = correlate_target(self._target(ss_h5="f1", ping_index=999), self._index(), max_dist_m=25.0)
        assert m.ping_idx == 0


class TestKleinSampleSpacingPicksFrequency:
    def test_target_matched_on_frequency_with_its_own_sample_spacing(self):
        # 40 m range: 455 kHz at 0.024 m/sample, 900 kHz at 0.012 m/sample.
        index = {Path("a.ss.h5"): {
            "hz_455000": {"lat": np.array([44.0]), "lon": np.array([-68.9]), "sample_interval_m": 0.024,
                          "n_samples": 1666, "n_pings": 1, "ping_start": 0},
            "hz_900000": {"lat": np.array([44.0]), "lon": np.array([-68.9]), "sample_interval_m": 0.012,
                          "n_samples": 3334, "n_pings": 1, "ping_start": 0}}}
        t = Target(target_id="t", source_format="sdf_klein", catalog_source="x", channel=0, sample=1600,
                   latitude=44.0, longitude=-68.9, extra_attributes={"ping_meters_per_sample": 0.012})
        assert correlate_target(t, index, max_dist_m=25.0).freq_key == "hz_900000"
