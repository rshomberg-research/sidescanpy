"""
Unit tests for sidescanpy.io.sdf.targets, against small synthetic
.tgt/.tgd fixtures built here (not real data) — shaped after the
inferred structure documented in that module (see its docstring for the
"confirmed vs. inferred" breakdown this synthetic layout is based on).

2026 rshomberg@uri.edu
"""

import struct
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sidescanpy.io.hdf5 import write_survey
from sidescanpy.io.sdf.targets import (
    calibrate_clock_offset,
    decode_target_window,
    read_klein_targets,
    resolve_target_position,
    resolve_targets_nearest_ping,
    resolve_targets_offset_corrected,
)
from sidescanpy.targets.core import Target


def _make_tgt_bytes(
    *,
    timestamp: int,
    channel: int,
    raw_window_row: int,
    range_m: float,
    max_range_m: float = 60.0,
    strings: list[str] | None = None,
) -> bytes:
    """Build synthetic .tgt bytes. ``raw_window_row`` is the *raw*
    trailer value as it appears on disk -- the reader halves it (see
    sidescanpy.io.sdf.targets's module docstring for why), so callers
    checking the parsed result should expect ``raw_window_row // 2``."""
    if strings is None:
        strings = ["5/13/2026 14:51:05", "61704", "23.54", "VHW20304", "Some notes here."]

    header = bytearray(0x28)
    struct.pack_into("<i", header, 8, timestamp)

    string_section = bytearray()
    for s in strings:
        encoded = s.encode("utf-16-le")
        string_section += b"\xff\xfe\xff" + bytes([len(s)]) + encoded

    trailer = [0] * 20
    trailer_f = [0.0] * 20
    trailer[2] = channel
    trailer[5] = raw_window_row
    trailer_f[3] = range_m
    trailer_f[8] = max_range_m
    trailer_bytes = bytearray()
    for i in range(20):
        # store both interpretations at the same offset by packing whichever
        # this slot logically holds; other slots default to 0 either way.
        if i in (3, 8):
            trailer_bytes += struct.pack("<f", trailer_f[i])
        else:
            trailer_bytes += struct.pack("<i", trailer[i])

    return bytes(header) + bytes(string_section) + bytes(trailer_bytes)


def _make_tgd_bytes(window: np.ndarray) -> bytes:
    n_pings, n_channels, n_samples = window.shape
    header = struct.pack("<2i", n_channels, n_samples)
    return header + window.astype("<i2").tobytes()


def _write_target(tmp_path: Path, stem: str, *, channel: int, raw_window_row: int, range_m: float, window: np.ndarray) -> None:
    timestamp = int(stem.removeprefix("Target_"))
    (tmp_path / f"{stem}.tgt").write_bytes(
        _make_tgt_bytes(timestamp=timestamp, channel=channel, raw_window_row=raw_window_row, range_m=range_m)
    )
    (tmp_path / f"{stem}.tgd").write_bytes(_make_tgd_bytes(window))


def _make_window(n_pings=20, n_channels=2, n_samples=100, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5000, size=(n_pings, n_channels, n_samples))


def test_reads_target_with_expected_fields(tmp_path):
    window = _make_window()
    _write_target(tmp_path, "Target_1536923484", channel=1, raw_window_row=8, range_m=24.0, window=window)

    targets = read_klein_targets(tmp_path)
    assert len(targets) == 1
    t = targets[0]
    assert t.target_id == "1536923484"
    assert t.source_format == "sdf_klein"
    assert t.pick_timestamp == 1536923484.0
    assert t.channel == 1
    assert t.position_confidence == "unknown"
    # raw trailer value is 8; the reader halves it (see module docstring)
    assert t.extra_attributes["window_row"] == 4
    # max_range_m=60.0, n_samples=100 -> meters_per_sample=0.6, range_m=24.0 -> sample=40
    assert t.sample == 40


def test_window_row_is_halved_from_the_raw_trailer_value(tmp_path):
    # Regression test for a real bug found 2026-08-07 via user inspection
    # of real output: an earlier version of this reader used the raw
    # trailer field directly, placing every bounding box on empty
    # seafloor instead of the real target. Confirmed empirically against
    # six real targets that the raw field is 2x the true row -- see
    # sidescanpy.io.sdf.targets's module docstring for the evidence.
    window = _make_window()
    _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=142, range_m=10.0, window=window)
    t = read_klein_targets(tmp_path)[0]
    assert t.extra_attributes["window_row"] == 71


def test_multiple_targets_sorted_by_filename(tmp_path):
    window = _make_window()
    _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=4, range_m=10.0, window=window)
    _write_target(tmp_path, "Target_1536923511", channel=1, raw_window_row=6, range_m=20.0, window=window)
    targets = read_klein_targets(tmp_path)
    assert [t.target_id for t in targets] == ["1536923484", "1536923511"]


def test_unreadable_file_is_skipped_not_raised(tmp_path):
    bad = tmp_path / "Target_9999999999.tgt"
    bad.write_bytes(b"not a real target file")
    window = _make_window()
    _write_target(tmp_path, "Target_1536923484", channel=0, raw_window_row=4, range_m=10.0, window=window)
    targets = read_klein_targets(tmp_path)
    assert len(targets) == 1
    assert targets[0].target_id == "1536923484"


def test_decode_target_window_matches_source_channel(tmp_path):
    window = _make_window(n_pings=15, n_channels=2, n_samples=50)
    _write_target(tmp_path, "Target_1536923484", channel=1, raw_window_row=4, range_m=5.0, window=window)
    targets = read_klein_targets(tmp_path)
    decoded = decode_target_window(targets[0])
    assert decoded.shape == (15, 50)
    np.testing.assert_array_equal(decoded, window[:, 1, :])


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
        true_utc = true_local_start.replace(tzinfo=timezone.utc) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
        known_offset_s = 500.0
        corrupted_utc_start = pd.Timestamp(true_utc) + pd.Timedelta(seconds=known_offset_s)
        _write_ss_h5_with_corrupted_time(tmp_path, true_local_start, corrupted_utc_start)

        offset_s = calibrate_clock_offset(tmp_path, local_utc_offset_hours=self._LOCAL_UTC_OFFSET_HOURS)
        assert offset_s == pytest.approx(known_offset_s, abs=1.0)

    def test_median_across_multiple_files(self, tmp_path):
        base_local = datetime(2018, 9, 14, 7, 20, 0)
        true_utc = base_local.replace(tzinfo=timezone.utc) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
        for i, offset in enumerate([400.0, 500.0, 600.0]):
            local_start = base_local + timedelta(minutes=i)
            this_true_utc = local_start.replace(tzinfo=timezone.utc) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
            corrupted_start = pd.Timestamp(this_true_utc) + pd.Timedelta(seconds=offset)
            _write_ss_h5_with_corrupted_time(tmp_path, local_start, corrupted_start, filename_prefix=f"sonar_data_{i}_")

        offset_s = calibrate_clock_offset(tmp_path, local_utc_offset_hours=self._LOCAL_UTC_OFFSET_HOURS)
        assert offset_s == pytest.approx(500.0, abs=1.0)  # median of 400/500/600

    def test_raises_when_no_filename_matches_pattern(self, tmp_path):
        (tmp_path / "not_a_matching_name.ss.h5").write_bytes(b"")
        with pytest.raises(RuntimeError):
            calibrate_clock_offset(tmp_path)


class TestResolveTargetsOffsetCorrected:
    _LOCAL_UTC_OFFSET_HOURS = 4.0

    def _make_calibrated_survey(self, tmp_path, known_offset_s=500.0):
        true_local_start = datetime(2018, 9, 14, 7, 20, 0)
        true_utc = true_local_start.replace(tzinfo=timezone.utc) + timedelta(hours=self._LOCAL_UTC_OFFSET_HOURS)
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
