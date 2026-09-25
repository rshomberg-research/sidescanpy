"""
Unit tests for sidescanpy.analysis.segmentation.

Synthetic fixtures with deliberately injected gaps/setting-changes/
garbage-runs/instability, per this project's TDD convention -- not real
survey data.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.analysis.segmentation import (
    Segment,
    detect_garbage_runs,
    detect_gaps,
    detect_setting_changes,
    detect_unstable_spans,
    find_segments,
    score_stability,
    write_segments,
    write_segments_report,
)
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey
from tests.conftest import make_minimal_sidescan_dataset


def _uniform_track_dataset(n_pings=200, n_samples=50):
    """A clean baseline: perfectly uniform lat/lon spacing (linspace ->
    constant along-track step), constant sound_vel_ms, nonzero random
    backscatter, constant altitude/pitch. Every detector should stay
    quiet against this on its own."""
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples)
    ds["latitude"] = ("ping", np.linspace(41.0000, 41.0100, n_pings))
    ds["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
    return ds


class TestDetectGaps:
    def test_no_gap_on_uniform_spacing(self):
        ds = _uniform_track_dataset()
        assert not detect_gaps(ds).any()

    def test_detects_injected_gap(self):
        ds = _uniform_track_dataset(n_pings=100)
        lat = ds["latitude"].values.copy()
        lat[50:] += 1.0  # ~111km jump, dwarfing the ~0.0001-degree/ping baseline step
        ds["latitude"] = ("ping", lat)
        boundary = detect_gaps(ds)
        assert boundary[50]
        assert boundary.sum() == 1

    def test_short_survey_returns_all_false(self):
        ds = _uniform_track_dataset(n_pings=2)
        assert not detect_gaps(ds).any()


class TestDetectSettingChanges:
    def test_no_change_no_boundary(self):
        ds = _uniform_track_dataset()
        assert not detect_setting_changes(ds).any()

    def test_sound_vel_step_detected(self):
        ds = _uniform_track_dataset(n_pings=60)
        sv = ds["sound_vel_ms"].values.copy()
        sv[30:] = sv[30:] * 1.5
        ds["sound_vel_ms"] = ("ping", sv)
        boundary = detect_setting_changes(ds)
        assert boundary[30]

    def test_valid_samples_change_detected(self):
        ds = _uniform_track_dataset(n_pings=60)
        n_channels = ds.sizes["channel"]
        vs = np.full((60, n_channels), 50, dtype=np.int32)
        vs[40:, :] = 30
        ds["valid_samples"] = (("ping", "channel"), vs)
        boundary = detect_setting_changes(ds)
        assert boundary[40]

    def test_tiny_relative_change_not_flagged(self):
        ds = _uniform_track_dataset(n_pings=60)
        sv = ds["sound_vel_ms"].values.copy()
        sv[30:] = sv[30:] * 1.001  # well under the 5% default tolerance
        ds["sound_vel_ms"] = ("ping", sv)
        assert not detect_setting_changes(ds).any()


class TestDetectGarbageRuns:
    def test_no_garbage_in_random_backscatter(self):
        ds = _uniform_track_dataset()
        assert detect_garbage_runs(ds) == []

    def test_detects_sustained_zero_run(self):
        ds = _uniform_track_dataset(n_pings=100)
        arr = ds["backscatter_raw"].values.copy()
        arr[10:35] = 0.0
        ds["backscatter_raw"] = (("ping", "channel", "sample"), arr)
        assert detect_garbage_runs(ds, min_run_pings=20) == [(10, 35)]

    def test_short_run_not_flagged(self):
        ds = _uniform_track_dataset(n_pings=100)
        arr = ds["backscatter_raw"].values.copy()
        arr[10:15] = 0.0  # 5 pings, under the 20-ping default
        ds["backscatter_raw"] = (("ping", "channel", "sample"), arr)
        assert detect_garbage_runs(ds, min_run_pings=20) == []


class TestScoreStability:
    def test_low_score_for_calm_data(self):
        altitude = np.full(50, 5.0)
        pitch = np.zeros(50)
        assert score_stability(altitude, pitch, 0, 50) < 0.01

    def test_high_score_for_noisy_data(self):
        rng = np.random.default_rng(0)
        altitude = 5.0 + rng.normal(0, 3.0, 50)
        pitch = rng.normal(0, 15.0, 50)
        assert score_stability(altitude, pitch, 0, 50) > 1.0

    def test_all_nan_window_returns_inf(self):
        altitude = np.full(50, np.nan)
        pitch = np.full(50, np.nan)
        assert score_stability(altitude, pitch, 0, 50) == float("inf")


class TestDetectUnstableSpans:
    def test_calm_survey_no_unstable_spans(self):
        ds = _uniform_track_dataset(n_pings=300)
        assert detect_unstable_spans(ds, window_pings=100) == []

    def test_detects_injected_instability(self):
        ds = _uniform_track_dataset(n_pings=300)
        rng = np.random.default_rng(1)
        pitch = ds["pitch_deg"].values.copy()
        altitude = ds["altitude_m"].values.copy()
        pitch[0:100] = rng.normal(0, 20.0, 100)
        altitude[0:100] = 5.0 + rng.normal(0, 4.0, 100)
        ds["pitch_deg"] = ("ping", pitch)
        ds["altitude_m"] = ("ping", altitude)
        spans = detect_unstable_spans(ds, window_pings=100, stability_threshold=1.0)
        assert spans == [(0, 100)]

    def test_missing_attitude_vars_returns_empty(self):
        ds = _uniform_track_dataset(n_pings=300).drop_vars(["pitch_deg"])
        assert detect_unstable_spans(ds) == []


class TestFindSegments:
    def test_clean_survey_is_one_segment(self):
        ds = _uniform_track_dataset(n_pings=300)
        segments = find_segments(ds, min_segment_pings=50)
        assert len(segments) == 1
        seg = segments[0]
        assert seg.status == "clean"
        assert seg.start == 0
        assert seg.stop == 300
        assert seg.reasons == ["clean"]

    def test_excludes_garbage_run_and_splits_around_it(self):
        ds = _uniform_track_dataset(n_pings=300)
        arr = ds["backscatter_raw"].values.copy()
        arr[100:150] = 0.0
        ds["backscatter_raw"] = (("ping", "channel", "sample"), arr)

        segments = find_segments(ds, garbage_min_run_pings=20, min_segment_pings=50)
        statuses = [(s.start, s.stop, s.status) for s in segments]
        assert (100, 150, "excluded") in statuses
        assert any(s.status == "clean" and s.stop <= 100 for s in segments)
        assert any(s.status == "clean" and s.start >= 150 for s in segments)
        excluded = [s for s in segments if s.status == "excluded"][0]
        assert "garbage_run" in excluded.reasons

    def test_min_segment_pings_merges_short_slivers(self):
        ds = _uniform_track_dataset(n_pings=300)
        sv = ds["sound_vel_ms"].values.copy()
        # Two setting-change boundaries 10 pings apart -- a sliver far
        # short of any reasonable min_segment_pings.
        sv[100:] *= 1.5
        sv[110:] /= 1.5
        ds["sound_vel_ms"] = ("ping", sv)

        segments = find_segments(ds, min_segment_pings=200)
        # No segment shorter than min_segment_pings should survive as its
        # own entry (the short sliver must have merged into a neighbor).
        assert all((s.stop - s.start) >= 200 or len(segments) == 1 for s in segments)

    def test_empty_survey_returns_no_segments(self):
        ds = _uniform_track_dataset(n_pings=0)
        assert find_segments(ds) == []


class TestWriteSegments:
    def _survey(self, n_pings=200):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
        survey.add_frequency("hz_1495000", _uniform_track_dataset(n_pings=n_pings))
        return survey

    def test_writes_only_clean_segments(self, tmp_path):
        survey = self._survey()
        segments = [
            Segment(label="section1", start=0, stop=80, status="clean", reasons=["clean"]),
            Segment(label="section2", start=80, stop=120, status="excluded", reasons=["garbage_run"]),
            Segment(label="section3", start=120, stop=200, status="clean", reasons=["clean"]),
        ]
        written = write_segments(survey, segments, tmp_path / "sections")
        names = {p.name for p in written}
        assert names == {"section1.ss.h5", "section3.ss.h5"}
        assert not (tmp_path / "sections" / "section2.ss.h5").exists()

    def test_written_segment_has_correct_ping_range(self, tmp_path):
        survey = self._survey()
        segments = [Segment(label="section1", start=50, stop=150, status="clean", reasons=["clean"])]
        [out_path] = write_segments(survey, segments, tmp_path / "sections")
        reread = read_survey(out_path)
        assert reread.get_frequency("hz_1495000").dataset.sizes["ping"] == 100

    def test_every_frequency_channel_sliced(self, tmp_path):
        survey = self._survey()
        survey.add_frequency("hz_547500", _uniform_track_dataset(n_pings=200))
        segments = [Segment(label="section1", start=0, stop=40, status="clean", reasons=["clean"])]
        [out_path] = write_segments(survey, segments, tmp_path / "sections")
        reread = read_survey(out_path)
        assert reread.get_frequency("hz_1495000").dataset.sizes["ping"] == 40
        assert reread.get_frequency("hz_547500").dataset.sizes["ping"] == 40


class TestWriteSegmentsReport:
    def test_report_round_trips(self, tmp_path):
        import json

        segments = [
            Segment(label="section1", start=0, stop=100, status="clean", reasons=["clean"]),
            Segment(label="section2", start=100, stop=150, status="excluded", reasons=["gap", "garbage_run"]),
        ]
        path = tmp_path / "segments.json"
        write_segments_report(segments, path)
        data = json.loads(path.read_text())
        assert len(data["segments"]) == 2
        assert data["segments"][1]["status"] == "excluded"
        assert data["segments"][1]["reasons"] == ["gap", "garbage_run"]
