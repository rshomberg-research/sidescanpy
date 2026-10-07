"""
Unit tests for sidescanpy.processing.segments.

Synthetic fixtures with deliberately injected gaps/setting-changes/
garbage-runs/instability.
"""

import numpy as np

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.segments import (
    Segment,
    detect_gaps,
    detect_garbage_runs,
    detect_setting_changes,
    detect_unstable_spans,
    find_segments,
    score_stability,
    split_by_segments,
)
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

    def test_detects_injected_time_gap_without_a_position_jump(self):
        """A logging pause/resume can replay a stale-but-plausible
        position while the clock jumps -- along-track distance alone
        wouldn't catch it, so the time check is independent."""
        ds = _uniform_track_dataset(n_pings=100)
        t = ds["time"].values.copy()
        t[50:] = t[50:] + np.timedelta64(1, "h")
        ds["time"] = ("ping", t)
        boundary = detect_gaps(ds)
        assert boundary[50]

    def test_time_gap_factor_controls_time_sensitivity(self):
        ds = _uniform_track_dataset(n_pings=100)
        t = ds["time"].values.copy()
        t[50:] = t[50:] + np.timedelta64(1, "h")
        ds["time"] = ("ping", t)
        assert not detect_gaps(ds, gap_factor=1e9, time_gap_factor=1e9).any()

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
        arr = ds["backscatter_intensity"].values.copy()
        arr[10:35] = 0.0
        ds["backscatter_intensity"] = (("ping", "channel", "sample"), arr)
        assert detect_garbage_runs(ds, min_run_pings=20) == [(10, 35)]

    def test_short_run_not_flagged(self):
        ds = _uniform_track_dataset(n_pings=100)
        arr = ds["backscatter_intensity"].values.copy()
        arr[10:15] = 0.0  # 5 pings, under the 20-ping default
        ds["backscatter_intensity"] = (("ping", "channel", "sample"), arr)
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
        arr = ds["backscatter_intensity"].values.copy()
        arr[100:150] = 0.0
        ds["backscatter_intensity"] = (("ping", "channel", "sample"), arr)

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

    def test_short_excluded_span_does_not_taint_the_surrounding_clean_pings(self):
        """Real bug, found via real-data testing (see sidescanpy's own
        CHANGELOG): an earlier version merged raw intervals by order
        regardless of status, then computed status from the *merged*
        range's own overlap with the excluded spans -- a short excluded
        span (here, 100 pings, under min_segment_pings=200) merging into
        a much larger clean interval flipped the ENTIRE merged range to
        "excluded", not just its own 100 pings. Reproduces the exact
        real shape: a short garbage run sitting inside a long clean
        stretch."""
        ds = _uniform_track_dataset(n_pings=2000)
        arr = ds["backscatter_intensity"].values.copy()
        arr[1000:1100] = 0.0  # 100-ping garbage run, shorter than min_segment_pings
        ds["backscatter_intensity"] = (("ping", "channel", "sample"), arr)

        segments = find_segments(ds, garbage_min_run_pings=20, min_segment_pings=200)

        excluded = [s for s in segments if s.status == "excluded"]
        assert excluded and all(s.start >= 1000 and s.stop <= 1100 for s in excluded)
        # The bulk of the survey, on both sides of the short excluded
        # run, must still be reported clean -- not swallowed into it.
        clean = [s for s in segments if s.status == "clean"]
        assert any(s.start <= 0 and s.stop >= 900 for s in clean)
        assert any(s.start >= 1100 and s.stop >= 1900 for s in clean)


class TestSplitBySegments:
    def _survey(self, n_pings=200):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="survey_a")
        survey.add_frequency("hz_1495000", _uniform_track_dataset(n_pings=n_pings))
        survey.add_frequency("hz_547500", _uniform_track_dataset(n_pings=n_pings))
        return survey

    def test_only_clean_segments_returned(self):
        segments = [
            Segment(label="section1", start=0, stop=80, status="clean", reasons=["clean"]),
            Segment(label="section2", start=80, stop=120, status="excluded", reasons=["garbage_run"]),
            Segment(label="section3", start=120, stop=200, status="clean", reasons=["clean"]),
        ]
        parts = split_by_segments(self._survey(), segments)
        assert list(parts) == ["section1", "section3"]
        assert parts["section1"].survey_name == "survey_a_section1"

    def test_every_frequency_sliced_to_segment(self):
        segments = [Segment(label="section1", start=50, stop=150, status="clean", reasons=["clean"])]
        sub = split_by_segments(self._survey(), segments)["section1"]
        for freq_key in ("hz_1495000", "hz_547500"):
            assert sub.get_frequency(freq_key).dataset.sizes["ping"] == 100

    def test_to_dict_is_json_serializable(self):
        import json

        seg = Segment(label="section1", start=0, stop=10, status="clean", reasons=["clean"])
        assert json.loads(json.dumps(seg.to_dict())) == {
            "label": "section1", "start": 0, "stop": 10, "status": "clean", "reasons": ["clean"],
        }
