"""
Unit tests for sidescanpy.io.sdf.targets, against small synthetic
.tgt/.tgd fixtures built here (not real data) — shaped after the
inferred structure documented in that module (see its docstring for the
"confirmed vs. inferred" breakdown this synthetic layout is based on).

2026 rshomberg@uri.edu
"""

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.sdf.targets import (
    decode_target_window,
    read_klein_targets,
    resolve_target_position,
)


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
