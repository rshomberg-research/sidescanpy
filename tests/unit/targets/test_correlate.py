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

from sidescanpy.io.hdf5 import write_survey
from sidescanpy.io.sdf.targets import read_klein_targets
from sidescanpy.targets.core import Target
from sidescanpy.targets.correlate import (
    calibrate_clock_offset,
    resolve_target_position,
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
