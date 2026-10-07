"""
Unit tests for sidescanpy.analysis.investigation.

Small synthetic fixtures with known, hand-constructed signal shapes
(gap at nadir, exponential decay, injected physics-model curves) —
per this project's TDD convention, not real survey data.

2026 rshomberg@uri.edu
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.analysis.investigation import (
    StageResult,
    _auto_near_edge_m,
    blank_nadir_noise,
    detect_first_bottom_return,
    detect_nadir_noise_extent,
    mask_nadir_gap,
    reject_range_outliers,
    smooth_boundary_across_pings,
    stage1_5_outlier_comparison,
    stage1_characterize,
    stage2_5_beam_geometry,
    stage2_nadir,
    stage3_usable_range,
    stage4_decay,
    stage5_units,
    uniform_exclusion_boundary,
)
from sidescanpy.processing import invert
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, absorption_for_frequency, tvg_gain_db


def _make_raw_dataset(
    *,
    n_pings: int = 20,
    n_samples: int = 200,
    sample_rate_hz: float = 1000.0,
    backscatter: np.ndarray | None = None,
    freq_hz: float = 1_495_000,
) -> xr.Dataset:
    """Minimal valid raw sidescan dataset — mirrors
    test_tvg_diagnostics.py's `_make_test_dataset` helper."""
    if backscatter is None:
        backscatter = np.ones((n_pings, 2, n_samples), dtype=np.float32)
    else:
        n_pings, _, n_samples = backscatter.shape

    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], backscatter.astype(np.float32)),
        "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
        "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
        "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
        "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
    }
    attrs = {
        "freq_hz_start": freq_hz,
        "freq_hz_end": freq_hz,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }
    return xr.Dataset(data_vars, coords, attrs)


class TestStage1Characterize:
    def test_basic_metrics(self):
        bs = np.zeros((5, 2, 10), dtype=np.float32)
        bs[..., 5:] = 4.0  # half zeros, half value=4
        ds = _make_raw_dataset(n_pings=5, n_samples=10, backscatter=bs)

        result = stage1_characterize(ds)

        assert result.metrics["dtype"] == "float32"
        assert result.metrics["min"] == 0.0
        assert result.metrics["max"] == 4.0
        assert np.isclose(result.metrics["mean"], 2.0)
        assert np.isclose(result.metrics["pct_zero"], 50.0)
        assert result.metrics["pct_negative"] == 0.0

    def test_negative_values_detected(self):
        bs = np.full((3, 2, 10), -5.0, dtype=np.float32)
        ds = _make_raw_dataset(n_pings=3, n_samples=10, backscatter=bs)
        result = stage1_characterize(ds)
        assert result.metrics["pct_negative"] == 100.0


class TestStage1_5OutlierComparison:
    def test_before_after_metrics_reflect_rejection(self):
        n_pings, n_samples = 30, 300
        rng = np.random.default_rng(4)
        # Small baseline noise is required: with a perfectly constant
        # baseline, non-spike bins have MAD=0 and reject_range_outliers
        # deliberately leaves those bins unfiltered (see
        # TestRejectRangeOutliers.test_degenerate_zero_mad_bin_left_unfiltered).
        bs = (5.0 + rng.normal(0, 0.1, size=(n_pings, 2, n_samples))).astype(np.float32)
        spike_idx = rng.choice(bs.size, size=50, replace=False)
        bs.flat[spike_idx] += 500.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage1_5_outlier_comparison(ds, bin_width_m=5.0)

        assert result.metrics["outlier_n_rejected"] > 0
        # Not every spike is guaranteed rejected (edge bins, spike
        # clustering) -- check the bulk of the population was cleaned
        # rather than requiring a specific residual max.
        assert result.metrics["before"]["max"] > 400.0
        assert result.metrics["before"]["pct_negative"] == result.metrics["after"]["pct_negative"] == 0.0
        assert result.metrics["outlier_n_rejected"] >= 40  # most of the 50 injected spikes
        assert result.data["filtered_array"].shape == bs.shape

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="ground_range_m"):
            stage1_5_outlier_comparison(ds)


class TestStage2Nadir:
    def test_detects_gap(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        bs[..., :50] = 0.0  # leading zero run on every ping/channel
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)

        result = stage2_nadir(ds, nadir_n_samples=50)

        assert result.classification == "gap"
        assert result.metrics["nadir_pct_zero"] == 100.0

    def test_detects_bright_spike(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        bs[..., :50] = 10.0  # bright near-range return, no zeros
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)

        result = stage2_nadir(ds, nadir_n_samples=50)

        assert result.classification == "bright_spike"

    def test_detects_clean(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)
        result = stage2_nadir(ds, nadir_n_samples=50)
        assert result.classification == "clean"

    def test_water_column_boundary_skipped_without_range_var(self):
        """Must still run and classify the nadir gap even when geometric
        ranges haven't been computed -- the altitude sample is a bonus,
        not a requirement, for backward compatibility."""
        ds = _make_raw_dataset(n_pings=5, n_samples=200)
        result = stage2_nadir(ds, nadir_n_samples=50)

        assert result.classification is not None
        assert "altitude_sample_median" not in result.metrics
        assert any("ground_range_m" in n for n in result.notes)

    def test_altitude_sample_matches_geometry(self):
        """altitude_m=10.0, sound_vel_ms=1500.0, sample_rate_hz=1000.0
        (fixture defaults) -> slant_range = sample_idx * 0.75m ->
        slant_range crosses altitude once sample index > 13.33 -> first
        finite ground_range sample = 14 (not a real bottom-contact
        boundary -- just altitude re-expressed in sample units)."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        result = stage2_nadir(ds, nadir_n_samples=5)

        assert result.metrics["altitude_sample_median"] == pytest.approx(14.0, abs=1.0)
        altitude_arr = result.data["altitude_sample"]
        assert altitude_arr.shape == (20, 2)
        assert np.all(np.isfinite(altitude_arr))

    def test_strong_return_min_sample_buffer_skips_residual_noise_past_altitude(self):
        """Integration-level regression test for the real
        finding: residual noise right past the altitude
        floor won as first_bottom_return_sample instead of the real
        bottom return further out. strong_return_min_sample_buffer
        (wired through to detect_first_bottom_return's min_sample_buffer)
        fixes it at the stage2_nadir level, not just the unit level."""
        n_pings, n_samples = 20, 600
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 20:50] = 20.0   # residual noise just past altitude (~14)
        bs[..., 300:] = 15.0    # real sustained bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        without_buffer = stage2_nadir(ds, nadir_n_samples=5)
        assert without_buffer.metrics["first_bottom_return_sample_median"] < 100  # confirms the setup

        with_buffer = stage2_nadir(ds, nadir_n_samples=5, strong_return_min_sample_buffer=150.0)
        assert with_buffer.metrics["first_bottom_return_sample_median"] == pytest.approx(300, abs=15)

    def test_min_first_bottom_return_sample_floors_past_a_weak_altitude_floor(self):
        """min_first_bottom_return_sample is an explicit, user-supplied
        floor independent of the altitude-derived one -- here the
        altitude crossing (~14 samples, this fixture's default geometry)
        is far too weak on its own to keep the detector off residual
        near-nadir noise; the explicit floor does what the altitude
        floor alone can't."""
        n_pings, n_samples = 20, 600
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 20:50] = 20.0   # residual noise, past altitude but before the explicit floor
        bs[..., 300:] = 15.0    # real sustained bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        without_floor = stage2_nadir(ds, nadir_n_samples=5)
        assert without_floor.metrics["first_bottom_return_sample_median"] < 100  # confirms the setup

        with_floor = stage2_nadir(ds, nadir_n_samples=5, min_first_bottom_return_sample=200.0)
        assert with_floor.metrics["first_bottom_return_sample_median"] == pytest.approx(300, abs=15)

    def test_min_first_bottom_return_sample_works_without_range_var(self):
        """altitude_sample is None when range_var/geometric ranges
        aren't available -- the explicit floor must still work as a
        standalone floor in that case, not silently do nothing."""
        n_pings, n_samples = 10, 400
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 5:30] = 20.0    # near-start noise, no altitude info available to exclude it
        bs[..., 200:] = 15.0    # real sustained bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        # deliberately no compute_geometric_ranges(ds) -- range_var absent

        without_floor = stage2_nadir(ds, nadir_n_samples=5)
        assert without_floor.metrics["first_bottom_return_sample_median"] < 100  # confirms the setup

        with_floor = stage2_nadir(ds, nadir_n_samples=5, min_first_bottom_return_sample=150.0)
        assert with_floor.metrics["first_bottom_return_sample_median"] == pytest.approx(200, abs=15)

    def test_all_nan_ground_range_names_short_recorded_window_cause(self):
        """Regression test for a real finding: a channel
        recorded with too short a range window for the vehicle's actual
        altitude never geometrically reaches bottom anywhere -- the
        resulting all-NaN ground_range_m note should name that specific,
        diagnosable cause (slant_range never exceeds altitude) rather
        than a generic 'no finite values', so a hands-off pipeline run
        is self-diagnosing. n_samples=10 -> max slant_range = 9*0.75 =
        6.75m, well under the fixture's altitude=10.0m."""
        ds = _make_raw_dataset(n_pings=10, n_samples=10)
        ds = compute_geometric_ranges(ds)
        assert not np.isfinite(ds["ground_range_m"].values).any()  # confirms the setup

        result = stage2_nadir(ds, nadir_n_samples=2)

        assert any("never reaches the vehicle's own altitude" in n for n in result.notes)

    def test_strong_return_boundary_too_close_to_start_skips_step_check(self):
        """Constant backscatter everywhere: detect_first_bottom_return
        correctly reports the strong return starting essentially at
        sample 0 (there's no water column to distinguish it from) --
        too close to the ping start for a pre/post step check, which
        should be flagged as such rather than silently computing NaN."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        result = stage2_nadir(ds, nadir_n_samples=5)

        assert result.metrics["first_bottom_return_sample_median"] < 20
        assert "signal_step_ratio" not in result.metrics
        assert any("too close to the start" in n for n in result.notes)

    def test_signal_step_no_note_with_real_step(self):
        n_pings, n_samples = 20, 300
        bs = np.full((n_pings, 2, n_samples), 1.0, dtype=np.float32)
        bs[..., :100] = 0.01  # weak water column, long enough for a pre/post window
        bs[..., 100:] = 5.0   # strong bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage2_nadir(ds, nadir_n_samples=5)

        assert result.metrics["signal_step_ratio"] > 1.2
        assert not any("doesn't clearly step up" in n for n in result.notes)

    def test_altitude_and_bottom_boundaries_diverge_generates_note(self):
        """Weak-but-nonzero returns for a long stretch before the real
        first bottom return -- the altitude crossing (from range_var)
        lands early, the amplitude detector correctly lands much later,
        and the gap between them should be flagged."""
        n_pings, n_samples = 20, 300
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)  # un-illuminated nadir gap
        bs[..., 150:] = 5.0  # real first bottom return starts far past the altitude crossing (~14)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage2_nadir(ds, nadir_n_samples=5)

        assert result.metrics["altitude_sample_median"] < 20
        assert result.metrics["first_bottom_return_sample_median"] > 100
        assert result.metrics["altitude_to_bottom_gap_sample_median"] > 100
        assert any("disagree by a median of" in n for n in result.notes)

    def test_pre_altitude_spike_does_not_win_as_first_bottom_return(self):
        """Integration-level regression test for a real
        bug: stage2_nadir must pass its own altitude_sample through to
        detect_first_bottom_return as a floor, not just compute it and
        ignore it. altitude=10.0 (fixture default) -> altitude sample
        ~14 (0.75m/sample). A bright, sustained pre-altitude spike
        (residual nadir noise) must not win over a later, real bottom
        return."""
        n_pings, n_samples = 20, 200
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 5:12] = 20.0   # bright spike well before altitude (~14) -- physically impossible as bottom
        bs[..., 100:] = 15.0   # real bottom return, well past altitude
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage2_nadir(ds, nadir_n_samples=5, strong_return_smooth_window=3)

        assert result.metrics["first_bottom_return_sample_median"] >= 14
        assert result.metrics["first_bottom_return_sample_median"] > 50

    def test_smooths_isolated_per_ping_detection_outlier(self):
        """Real per-ping detection noise can make a single ping's raw
        boundary jump far from its neighbors, confirmed on real
        real data. Default boundary_smoothing_window should pull
        an isolated outlier ping back toward its neighbors;
        boundary_smoothing_window=1 disables that and leaves the raw
        detection as-is."""
        n_pings, n_samples = 41, 300
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 100:] = 15.0  # real bottom starts at sample 100 for every ping...
        bs[20, :, 100:180] = 0.05  # ...except ping 20, where noise pushes detection to ~180
        bs[20, :, 180:] = 15.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        smoothed = stage2_nadir(ds, nadir_n_samples=5)
        raw = stage2_nadir(ds, nadir_n_samples=5, boundary_smoothing_window=1)

        smoothed_boundary = smoothed.data["first_bottom_return_sample"]
        raw_boundary = raw.data["first_bottom_return_sample"]

        assert abs(raw_boundary[20, 0] - 180) <= 15
        assert abs(smoothed_boundary[20, 0] - 100) <= 15


class TestStage2_5BeamGeometry:
    def test_near_and_far_ray_match_known_geometry(self):
        """Isolates stage2_5's own angle math from
        detect_first_bottom_return's inherent smoothing imprecision
        (covered separately in TestDetectFirstBottomReturn) by feeding
        a hand-built StageResult with an exact, known boundary sample,
        rather than relying on detection to find it. altitude=10.0
        (fixture default); sample 27 with sound_vel=1500,
        sample_rate=1000 -> slant_range = 27*0.75 = 20.25m ->
        near_angle_from_vertical = arccos(10/20.25) ~= 60.4deg.
        usable_range_m=50.0 -> far_angle_from_vertical = arctan(50/10)
        ~= 78.7deg."""
        n_pings, n_samples = 20, 100
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        fake_boundary = np.full((n_pings, 2), 27.0)
        fake_stage2 = StageResult(
            stage="stage2_nadir", metrics={}, data={"first_bottom_return_sample": fake_boundary}
        )
        fake_stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 50.0})
        result = stage2_5_beam_geometry(ds, fake_stage2, fake_stage3)

        for entry in result.metrics["near_ray"].values():
            assert entry["observed_sample"] == 27
            assert entry["angle_from_vertical_deg"] == pytest.approx(60.4, abs=0.5)
        for label, entry in result.metrics["far_ray"].items():
            assert entry["angle_from_vertical_deg"] == pytest.approx(78.7, abs=0.5)
            expected_sign = -1.0 if label == "port" else 1.0
            assert entry["predicted_across_track_m"] == pytest.approx(expected_sign * 50.0)
            # slant range is a true distance (hypotenuse), never signed
            assert entry["predicted_slant_range_m"] == pytest.approx(np.hypot(50.0, 10.0))
        assert result.metrics["beam_width_deg"] == pytest.approx(78.7 - 60.4, abs=1.0)

    def test_far_ray_angle_at_45_degrees_when_usable_range_equals_altitude(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 10.0})  # == altitude -> 45deg

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        for label, entry in result.metrics["far_ray"].items():
            assert entry["angle_from_vertical_deg"] == pytest.approx(45.0, abs=1e-6)
            expected_sign = -1.0 if label == "port" else 1.0
            assert entry["predicted_across_track_m"] == pytest.approx(expected_sign * 10.0)

    def test_far_ray_port_and_starboard_are_oppositely_signed(self):
        """Regression test: usable_range_m is a single survey-aggregate
        magnitude, not per-channel -- an earlier version stored it
        unsigned for both "port" and "starboard" far_ray entries, which
        meant the beam-geometry diagram only ever drew a far ray toward
        the starboard side (both entries carried the same positive
        value), never port. across_track_m's convention (port negative,
        starboard positive) must be applied here the same way it's
        already inherent in near_ray's real per-channel values."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 25.0})

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.metrics["far_ray"]["port"]["predicted_across_track_m"] == pytest.approx(-25.0)
        assert result.metrics["far_ray"]["starboard"]["predicted_across_track_m"] == pytest.approx(25.0)

    def test_far_ray_none_when_usable_range_none(self):
        """A real, documented outcome on some hardware (see
        stage3_usable_range's docstring) -- the far edge, beam width,
        and center angle should come back None with an explanatory
        note, not raise or silently omit."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": None})

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.metrics["far_angle_deg"] is None
        assert result.metrics["beam_width_deg"] is None
        assert result.metrics["beam_center_angle_deg"] is None
        assert any("usable_range_m is None" in n for n in result.notes)

    def test_missing_required_var_raises(self):
        ds = _make_raw_dataset()  # no geometric ranges computed
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={})
        with pytest.raises(ValueError, match="across_track_m"):
            stage2_5_beam_geometry(ds, stage2, stage3)

    def test_impossible_geometry_flagged(self):
        """A boundary sample whose slant range is less than altitude
        (fixture default 10.0; sample 5 -> slant_range=3.75m) is
        geometrically impossible for any real beam angle -- should be
        flagged, not silently produce a bogus angle."""
        n_pings, n_samples = 20, 100
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        fake_boundary = np.full((n_pings, 2), 5.0)
        fake_stage2 = StageResult(
            stage="stage2_nadir", metrics={}, data={"first_bottom_return_sample": fake_boundary}
        )
        fake_stage3 = StageResult(stage="stage3_usable_range", metrics={})
        result = stage2_5_beam_geometry(ds, fake_stage2, fake_stage3)

        for entry in result.metrics["near_ray"].values():
            assert entry["angle_from_vertical_deg"] is None
        assert any("geometrically" in n for n in result.notes)

    @staticmethod
    def _make_angle_ds(angle_deg_per_ping: np.ndarray, *, altitude: float = 10.0) -> tuple[xr.Dataset, StageResult, StageResult]:
        """Directly construct slant_range_m so the near-edge angle at
        boundary sample 0 is exactly `angle_deg_per_ping[p]` for every
        ping/channel -- bypasses compute_geometric_ranges' coarse
        (0.75m/sample) discretization, which can't hit small target
        angles at an integer sample boundary. Returns a stage3 with no
        usable_range_m -- these tests only exercise the near-edge
        classification, which doesn't depend on the far edge."""
        n_pings = angle_deg_per_ping.shape[0]
        n_channels, n_samples = 2, 5
        slant_at_zero = altitude / np.cos(np.deg2rad(angle_deg_per_ping))
        slant = np.ones((n_pings, n_channels, n_samples), dtype=np.float64) * 1000.0
        slant[:, :, 0] = slant_at_zero[:, None]
        across = np.zeros((n_pings, n_channels, n_samples), dtype=np.float64)
        ds = xr.Dataset(
            {
                "slant_range_m": (("ping", "channel", "sample"), slant),
                "across_track_m": (("ping", "channel", "sample"), across),
                "altitude_m": (("ping",), np.full(n_pings, altitude)),
            }
        )
        boundary = np.zeros((n_pings, n_channels))
        stage2 = StageResult(stage="stage2_nadir", metrics={}, data={"first_bottom_return_sample": boundary})
        stage3 = StageResult(stage="stage3_usable_range", metrics={})
        return ds, stage2, stage3

    def test_classification_determined(self):
        angles = np.full(40, 30.0)  # consistent, clearly nonzero tilt
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.classification == "determined"
        assert result.metrics["near_angle_deg_median"] == pytest.approx(30.0, abs=0.1)
        assert result.metrics["near_angle_valid_fraction"] == pytest.approx(1.0)

    def test_classification_determined_even_at_near_zero_angle(self):
        """An earlier version of this stage treated a near-zero implied
        angle as a different physical category ("near-grazing weak
        backscatter" vs. "water column") -- that distinction was wrong:
        the near-nadir span is never illuminated regardless of how
        small the beam's near angle is. classification now reports
        measurement confidence only, not a choice between two physical
        stories -- a small but consistent angle is still `determined`."""
        angles = np.full(40, 2.0)
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.classification == "determined"
        assert result.metrics["near_angle_deg_median"] == pytest.approx(2.0, abs=0.1)

    def test_classification_ambiguous_when_inconsistent(self):
        rng = np.random.default_rng(11)
        angles = rng.uniform(5.0, 85.0, size=60)  # wide spread -> high std
        ds, stage2, stage3 = self._make_angle_ds(angles)

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.classification == "ambiguous"
        assert result.metrics["near_angle_deg_std"] > 15.0

    def test_classification_undetermined_when_mostly_invalid(self):
        n_pings, n_channels, n_samples = 20, 2, 5
        # slant_range_m < altitude everywhere -- every entry geometrically impossible.
        ds = xr.Dataset(
            {
                "slant_range_m": (("ping", "channel", "sample"), np.full((n_pings, n_channels, n_samples), 1.0)),
                "across_track_m": (("ping", "channel", "sample"), np.zeros((n_pings, n_channels, n_samples))),
                "altitude_m": (("ping",), np.full(n_pings, 10.0)),
            }
        )
        boundary = np.zeros((n_pings, n_channels))
        stage2 = StageResult(stage="stage2_nadir", metrics={}, data={"first_bottom_return_sample": boundary})
        stage3 = StageResult(stage="stage3_usable_range", metrics={})

        result = stage2_5_beam_geometry(ds, stage2, stage3)

        assert result.classification == "undetermined"
        assert result.metrics["near_angle_valid_fraction"] == pytest.approx(0.0)


class TestStage3UsableRange:
    def test_finds_usable_range_before_noise_floor(self):
        n_pings, n_samples = 50, 200
        rng = np.random.default_rng(0)
        sample_idx = np.arange(n_samples)
        # Strong signal 0-100 samples, near-zero noise floor beyond.
        decay = np.where(sample_idx < 100, 50.0 * np.exp(-sample_idx / 30.0), 0.0)
        bs = np.broadcast_to(decay, (n_pings, 2, n_samples)).copy()
        bs += rng.normal(0, 0.01, size=bs.shape)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)

        assert result.metrics["usable_range_m"] is not None
        # Signal decays to near-nothing well before the far range edge.
        assert result.metrics["usable_range_m"] < ds["ground_range_m"].max().item()

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="slant_range_m"):
            stage3_usable_range(ds)

    def _make_two_channel_range_ds(self, n_pings=50, n_samples=200):
        """channel 0 keeps a strong signal much further out than channel
        1 -- distinguishable usable_range_m per channel, so pooling vs.
        restricting is directly observable in the result."""
        sample_idx = np.arange(n_samples)
        decay_far = np.where(sample_idx < 150, 50.0 * np.exp(-sample_idx / 60.0), 0.0)
        decay_near = np.where(sample_idx < 40, 50.0 * np.exp(-sample_idx / 12.0), 0.0)
        rng = np.random.default_rng(1)
        bs = np.zeros((n_pings, 2, n_samples))
        bs[:, 0, :] = decay_far
        bs[:, 1, :] = decay_near
        bs += rng.normal(0, 0.01, size=bs.shape)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        return compute_geometric_ranges(ds)

    def test_channel_none_pools_both_channels(self):
        ds = self._make_two_channel_range_ds()

        pooled = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)

        assert pooled.metrics["channel"] is None

    def test_channel_restricts_usable_range_to_one_side(self):
        ds = self._make_two_channel_range_ds()

        ch0 = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=0)
        ch1 = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=1)

        assert ch0.metrics["channel"] == 0
        assert ch1.metrics["channel"] == 1
        # channel 0's signal reaches much further than channel 1's --
        # proof each was fit against only its own side, not pooled.
        assert ch0.metrics["usable_range_m"] > ch1.metrics["usable_range_m"]

    def test_channel_string_matches_int_position(self):
        ds = self._make_two_channel_range_ds()

        by_string = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel="port")
        by_int = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=0)

        assert by_string.metrics["usable_range_m"] == pytest.approx(by_int.metrics["usable_range_m"])
        assert by_string.metrics["channel"] == 0

    def test_invalid_channel_string_raises(self):
        ds = self._make_two_channel_range_ds()

        with pytest.raises(ValueError, match="not found in channel_id"):
            stage3_usable_range(ds, channel="midship")

    def test_water_column_n_excluded_matches_geometry(self):
        """Same physics as TestStage2Nadir.test_geometric_boundary_matches_geometry:
        altitude=10.0, sound_vel=1500.0, sample_rate=1000.0 -> water
        column ends at sample 14 -> 14 excluded samples per (ping, channel)."""
        n_pings, n_samples = 50, 200
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        result = stage3_usable_range(ds, bin_width_m=1.0)

        assert result.metrics["water_column_n_excluded"] == 14 * n_pings * 2

    def test_exclude_before_sample_masks_nadir_gap(self):
        """exclude_before_sample should additionally exclude samples
        the altitude-crossing mask alone doesn't catch."""
        n_pings, n_samples = 20, 200
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        boundary = np.full((n_pings, 2), 50.0)  # well past the altitude sample (~14)
        result = stage3_usable_range(ds, bin_width_m=1.0, exclude_before_sample=boundary)

        assert result.metrics["nadir_gap_n_excluded"] == (50 - 14) * n_pings * 2

    def test_reject_outliers_lowers_far_range_noise_std(self):
        """Inject bright spikes into the far-range 'noise floor' region
        (simulating a target sitting past the real signal cutoff).
        `noise_floor` (a median of bin medians) is already fairly
        robust to a sparse handful of spikes -- `noise_std` (a mean of
        bin std-devs) is not, and is where the effect should show up
        clearly."""
        n_pings, n_samples = 50, 200
        rng = np.random.default_rng(3)
        sample_idx = np.arange(n_samples)
        decay = np.where(sample_idx < 100, 50.0 * np.exp(-sample_idx / 30.0), 0.0)
        bs = np.broadcast_to(decay, (n_pings, 2, n_samples)).copy()
        bs += rng.normal(0, 0.01, size=bs.shape)

        spike_idx = rng.choice(n_pings * 2 * (n_samples - 120), size=15, replace=False)
        far_region = bs[:, :, 120:]
        far_flat = far_region.reshape(-1)
        far_flat[spike_idx] += 20.0
        bs[:, :, 120:] = far_flat.reshape(far_region.shape)

        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        with_spikes = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)
        cleaned = stage3_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, reject_outliers=True)

        assert cleaned.metrics["outlier_n_rejected"] > 0
        assert cleaned.metrics["noise_std"] < with_spikes.metrics["noise_std"]

    def test_per_ping_boundary_avoids_thin_contaminated_near_bin_vs_uniform(self):
        """Regression test for a real LF-channel bug: a
        single survey-wide (`uniform_exclusion_boundary`) exclusion
        boundary under-masks whichever pings' true per-ping boundary
        exceeds it. Because ground range at a fixed sample index
        depends on a ping's own altitude, those under-masked pings'
        still-invalid (water-column) samples can land in a different,
        nearer range bin than where the rest of the survey's valid
        signal starts -- producing a thin, near-zero "first point"
        artifact on the usable-range plot (confirmed on real HF/LF
        data: a uniform boundary left an anomalous count=30,
        median=0.39 bin at 8-10m; the exact per-ping boundary removes
        it entirely). `run_investigation` now passes the exact
        per-ping array to Stage 3 for this reason -- this test locks
        that choice in."""
        n_pings, n_samples = 60, 300
        rng = np.random.default_rng(7)

        # 90% shallow pings (boundary=100, range grid 0-30m); 10% deep
        # pings (boundary=250, range grid 0-10m -- larger altitude
        # compresses ground range at a given sample, same direction as
        # real geometry). `uniform_exclusion_boundary` collapses to
        # ~the shallow group's boundary since it dominates by count,
        # under-masking the deep group by ~150 samples.
        is_deep = np.zeros(n_pings, dtype=bool)
        is_deep[:6] = True
        rng.shuffle(is_deep)

        boundary = np.broadcast_to(np.where(is_deep, 250.0, 100.0)[:, None], (n_pings, 2)).astype(np.float64)

        sample_idx = np.arange(n_samples)
        range_grid = np.where(is_deep[:, None], sample_idx[None, :] / 30.0, sample_idx[None, :] / 10.0)
        ranges = np.broadcast_to(range_grid[:, None, :], (n_pings, 2, n_samples)).astype(np.float64).copy()

        valid = 20.0 * np.exp(-(sample_idx[None, None, :] - boundary[:, :, None]) / 80.0)
        arr = np.where(sample_idx[None, None, :] >= boundary[:, :, None], valid, 0.01)
        arr = arr + rng.normal(0, 0.02, arr.shape)

        ds = xr.Dataset(
            {"_v": (("ping", "channel", "sample"), arr), "_r": (("ping", "channel", "sample"), ranges)}
        )

        uniform = uniform_exclusion_boundary(boundary, percentile=90.0)
        s3_per_ping = stage3_usable_range(ds, "_v", range_var="_r", bin_width_m=1.0, exclude_before_sample=boundary)
        s3_uniform = stage3_usable_range(ds, "_v", range_var="_r", bin_width_m=1.0, exclude_before_sample=uniform)

        per_ping_binned = s3_per_ping.data["binned"]
        uniform_binned = s3_uniform.data["binned"]

        # Exact per-ping masking: no valid signal exists below ~8m for
        # any ping, so no near-zero contamination bin appears there.
        assert not ((per_ping_binned["range_min_m"] < 8.0) & (per_ping_binned["median"] < 1.0)).any()

        # Uniform boundary: the deep group's still-invalid samples leak
        # through in [100, 250) at their own (compressed) range grid,
        # landing below 8m where no valid data exists -- a thin,
        # near-zero contaminated bin the per-ping boundary doesn't have.
        contaminated = uniform_binned[(uniform_binned["range_min_m"] < 8.0) & (uniform_binned["median"] < 1.0)]
        assert not contaminated.empty


class TestStage4Decay:
    def test_full_physics_curve_classified_as_none(self):
        """Raw data that follows the uncompensated physics model
        end-to-end should classify as TVG status 'none'."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)

        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)

        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        result = stage4_decay(ds, bin_width_m=2.0)

        assert result.classification == "none"
        assert result.metrics["best_candidate"] == "full_physics"

    def _make_full_physics_ds(self, n_pings=60, n_samples=300):
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)
        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)
        return ds

    def test_short_usable_range_auto_shrinks_bin_width_to_meet_min_bins(self):
        """Real-data motivation (2026-08-25): a survey flown very close
        to the bottom, with usable_range_m as short as ~15m, left the
        default bin_width_m=5.0 with only 2-3 bins in the fit window --
        not enough to reliably distinguish 4 competing single-parameter
        physics models. min_bins (default 6) now shrinks bin_width_m
        automatically for a short usable_range_m."""
        ds = self._make_full_physics_ds()

        # range_var pinned to ground_range_m: this test's numbers (and the
        # real report it reproduces) predate stage4_decay's 2026-09-02
        # range_var default change to slant_range_m -- pinning keeps this
        # regression test decoupled from that unrelated default.
        adapted = stage4_decay(ds, range_var="ground_range_m", bin_width_m=5.0, usable_range_m=15.0)
        unadapted = stage4_decay(ds, range_var="ground_range_m", bin_width_m=5.0, usable_range_m=15.0, min_bins=0)

        assert adapted.metrics["bin_width_m_requested"] == 5.0
        assert adapted.metrics["bin_width_m"] == pytest.approx(15.0 / 6)
        assert adapted.metrics["bin_width_m"] < 5.0
        assert any("bin_width_m auto-reduced" in n for n in adapted.notes)
        # The exact fit-window bin count depends on where r_ref_m lands
        # (see min_bins's own docstring -- the guarantee targets the full
        # [0, usable_range_m] span, not the fit window specifically), but
        # it must be a real, substantial improvement over the unadapted
        # 5.0m-bin case this real report was about.
        assert adapted.metrics["n_bins_used"] > unadapted.metrics["n_bins_used"]
        assert unadapted.metrics["n_bins_used"] <= 3  # reproduces the reported "only 3 bins"
        assert adapted.metrics["n_bins_used"] >= 4

    def test_wide_usable_range_leaves_bin_width_unchanged(self):
        """The default bin_width_m=5.0 already gives well over min_bins
        bins across a normal 40-70m-range survey -- confirms the
        adaptation is a floor, not a universal override, and existing
        callers with plenty of natural bins see no behavior change."""
        ds = self._make_full_physics_ds()

        result = stage4_decay(ds, bin_width_m=5.0, usable_range_m=60.0)

        assert result.metrics["bin_width_m"] == 5.0
        assert result.metrics["bin_width_m_requested"] == 5.0
        assert not any("bin_width_m auto-reduced" in n for n in result.notes)

    def test_min_bins_zero_disables_adaptation(self):
        ds = self._make_full_physics_ds()

        result = stage4_decay(ds, bin_width_m=5.0, usable_range_m=15.0, min_bins=0)

        assert result.metrics["bin_width_m"] == 5.0
        assert result.metrics["bin_width_m_requested"] == 5.0

    def test_no_usable_range_m_leaves_bin_width_unchanged(self):
        """usable_range_m=None (Stage 3 couldn't determine one) has
        nothing to adapt against -- must not crash, must leave the
        requested bin_width_m as-is."""
        ds = self._make_full_physics_ds()

        result = stage4_decay(ds, bin_width_m=5.0, usable_range_m=None)

        assert result.metrics["bin_width_m"] == 5.0
        assert result.metrics["bin_width_m_requested"] == 5.0

    def test_classification_ranked_by_fixed_scale_rmse_not_free_fit_rmse(self, monkeypatch):
        """Isolates stage4_decay's own ranking logic from
        fit_hardware_tvg_curve's real fitting behavior entirely --
        injects a hand-built TVGCandidateFitResult where free-fit
        rmse_db and fixed_scale_rmse_db disagree about the winner by
        construction, and confirms stage4_decay's classification
        follows fixed_scale_rmse_db (2026-08-24 change)."""
        import sidescanpy.analysis.investigation as inv_mod
        from sidescanpy.analysis.tvg_diagnostics import TVGCandidateFit, TVGCandidateFitResult

        ds = _make_raw_dataset(n_pings=10, n_samples=100)
        ds = compute_geometric_ranges(ds)

        # "absorption_only" wins on free-fit rmse_db (0.1, lowest) but
        # needs an implausible scale=5.0 to get there; "full_physics"
        # wins on fixed_scale_rmse_db (0.2, lowest) with a free-fit
        # scale of 1.02 -- close to the literal, uncalibrated model.
        candidates = [
            TVGCandidateFit(name="flat", params=TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0),
                             scale=0.0, rmse_db=10.0, fixed_scale_rmse_db=10.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="spreading_only", params=TVGParams(absorption_db_per_m=0.0),
                             scale=1.5, rmse_db=3.0, fixed_scale_rmse_db=5.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="absorption_only", params=TVGParams(spreading_loss_db=0.0),
                             scale=5.0, rmse_db=0.1, fixed_scale_rmse_db=8.0,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
            TVGCandidateFit(name="full_physics", params=TVGParams(),
                             scale=1.02, rmse_db=1.0, fixed_scale_rmse_db=0.2,
                             r_ref_m=5.0, r_fit_max_m=40.0, n_bins_used=20),
        ]
        fake_result = TVGCandidateFitResult(
            candidates=candidates,
            best=min(candidates, key=lambda c: c.fixed_scale_rmse_db),
            observed_range_m=np.array([5.0, 40.0]),
            observed_loss_db=np.array([0.0, 10.0]),
        )
        monkeypatch.setattr(inv_mod, "fit_hardware_tvg_curve", lambda *a, **kw: fake_result)

        result = inv_mod.stage4_decay(ds, bin_width_m=2.0)

        assert result.metrics["best_candidate"] == "full_physics"
        assert result.metrics["best_rmse_db"] == pytest.approx(0.2)
        assert result.metrics["best_free_fit_scale"] == pytest.approx(1.02)
        assert result.metrics["best_free_fit_rmse_db"] == pytest.approx(1.0)
        assert result.metrics["candidates"]["absorption_only"]["fixed_scale_rmse_db"] == pytest.approx(8.0)
        # runner-up by fixed_scale_rmse_db is spreading_only (5.0), not
        # absorption_only (8.0) -- confirms the whole ranking, not just best
        assert result.classification == "none"  # _TVG_CANDIDATE_TO_STATUS["full_physics"]
        assert "spreading_only" in result.notes[0]

    def test_classification_invariant_to_db_reference(self):
        """The self-referenced dB conversion's reference percentile is
        a constant offset in dB space, which cancels out of every
        loss/residual computation `fit_hardware_tvg_curve` does — so
        changing it must not change any classification output, only
        `metrics["db_reference"]["value"]` itself."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        low_ref = stage4_decay(ds, bin_width_m=2.0, db_reference_percentile=50.0)
        high_ref = stage4_decay(ds, bin_width_m=2.0, db_reference_percentile=99.9)

        assert low_ref.metrics["db_reference"]["value"] != high_ref.metrics["db_reference"]["value"]
        assert low_ref.classification == high_ref.classification
        assert low_ref.metrics["best_candidate"] == high_ref.metrics["best_candidate"]
        # float32 backscatter + the +eps offset means the two reference
        # choices aren't bit-identical, just equal well past float32
        # precision -- rtol chosen loose enough to absorb that, tight
        # enough that a real reference-dependence bug would still fail it.
        assert np.isclose(low_ref.metrics["best_scale"], high_ref.metrics["best_scale"], rtol=1e-4)
        assert np.isclose(low_ref.metrics["best_rmse_db"], high_ref.metrics["best_rmse_db"], rtol=1e-4, atol=1e-4)

    def test_flat_curve_yields_negligible_residual(self):
        """Raw data with no range-dependent decay at all (hardware
        already fully compensated): every candidate's fitted scale
        collapses toward 0 with tiny RMSE -- with only noise left to
        fit, `spreading_only`/`absorption_only`/`full_physics` can
        occasionally out-fit `flat` by chance (their free scale
        overfits the noise itself), which is exactly the
        near-flat-data fit-degeneracy `fit_hardware_tvg_curve` already
        documents (see that function's `r_ref_m` docstring). So this
        checks "no real decay detected" (uniformly tiny RMSE, status
        reads as no-correction-needed-or-ambiguous), not a specific
        winning candidate name."""
        n_pings, n_samples = 30, 300
        bs = np.full((n_pings, 2, n_samples), 5.0, dtype=np.float32)
        rng = np.random.default_rng(1)
        bs += rng.normal(0, 0.05, size=bs.shape).astype(np.float32)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = stage4_decay(ds, bin_width_m=2.0)

        assert result.classification in ("full", "ambiguous")
        assert result.metrics["best_rmse_db"] < 0.1
        assert all(c["rmse_db"] < 0.1 for c in result.metrics["candidates"].values())

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="slant_range_m"):
            stage4_decay(ds)

    def test_exclude_first_n_bins_zero_is_exact_prior_behavior(self):
        ds = self._make_full_physics_ds()

        default = stage4_decay(ds, bin_width_m=2.0)
        explicit_zero = stage4_decay(ds, bin_width_m=2.0, exclude_first_n_bins=0)

        assert default.metrics == explicit_zero.metrics
        assert default.classification == explicit_zero.classification
        assert default.metrics["exclude_first_n_bins"] == 0
        assert default.metrics["n_bins_excluded_after_bottom"] == 0

    def test_exclude_first_n_bins_moves_r_ref_away_from_a_bright_artifact_at_the_bottom_edge(self):
        """Real-world motivation: a bright leading-edge specular
        artifact right at the first-bottom-return boundary can itself
        win r_ref_m's near-field max-mean auto-selection, corrupting
        the reference (and every candidate's fit against it) --
        distinct from the ring-down artifact at the very START of a
        ping, which reads anomalously LOW and is already handled by
        picking the max instead of the first bin (see
        fit_hardware_tvg_curve's own r_ref_m docstring).
        exclude_first_n_bins drops the artifact bin(s) before r_ref_m
        ever sees them."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)
        # altitude=10m, sample_rate=1000Hz, sound_vel=1500m/s -> 0.75m/
        # sample -> water column ends ~sample 14 (slant_range > altitude,
        # ground_range ~3.2m there). Sample 14 alone lands cleanly in
        # the first 2m-wide bin -- inject a bright artifact there.
        bs = ds["backscatter_intensity"].values.copy()
        bs[:, :, 14:15] *= 80.0
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        without_exclusion = stage4_decay(ds, bin_width_m=2.0)
        with_exclusion = stage4_decay(ds, bin_width_m=2.0, exclude_first_n_bins=1)

        assert without_exclusion.metrics["n_bins_excluded_after_bottom"] == 0
        assert with_exclusion.metrics["n_bins_excluded_after_bottom"] == 1
        assert any("Dropped 1 bin(s)" in n for n in with_exclusion.notes)
        # the artifact should have corrupted r_ref_m's auto-selection
        # without exclusion, and dropping it should move the pick
        assert with_exclusion.metrics["r_ref_m"] != without_exclusion.metrics["r_ref_m"]
        # underlying curve is still genuinely full_physics -- dropping
        # the artifact should recover a real, better fit against it
        assert with_exclusion.metrics["best_candidate"] == "full_physics"
        assert with_exclusion.metrics["best_rmse_db"] < without_exclusion.metrics["best_rmse_db"]

    def test_r_fit_min_m_passthrough_fits_both_sides_of_reference(self):
        """Combined real use case (2026-09-02): r_ref_m pinned mid-swath
        + r_fit_min_m=0.0 to fit both sides + exclude_first_n_bins to
        drop only the near-boundary artifact bins, not the whole near
        half of the swath."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        one_sided = stage4_decay(ds, bin_width_m=2.0, r_ref_m=20.0)
        both_sides = stage4_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m=0.0)

        assert both_sides.metrics["r_ref_m"] == 20.0  # unchanged
        assert both_sides.metrics["r_fit_min_m"] == 0.0
        assert one_sided.metrics["r_fit_min_m"] == 20.0  # defaults to r_ref_m
        assert both_sides.metrics["n_bins_used"] > one_sided.metrics["n_bins_used"]

    def test_r_fit_min_m_auto_resolves_to_numeric_value_and_records_which_mode_ran(self):
        """r_fit_min_m="auto": a fixed near-edge number picked by
        eye on one mission didn't generalize across a dataset whose
        near-field ramp varies mission to mission. metrics["r_fit_min_m"]
        must carry the *resolved* number (what the fit actually used);
        metrics["r_fit_min_m_override"] must keep the literal "auto" so
        the JSON export still shows which mode produced the run."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m="auto")

        assert isinstance(result.metrics["r_fit_min_m"], float)
        assert result.metrics["r_fit_min_m_override"] == "auto"
        assert result.metrics["r_fit_min_auto_margin_m"] == pytest.approx(5.0)
        assert any("r_fit_min_m='auto' resolved to" in n for n in result.notes)

    def test_r_fit_min_m_auto_without_r_ref_m_or_usable_range_m_falls_back_gracefully(self):
        """Graceful fallback instead of raising --
        "auto" is a default (not just an opt-in
        override), so a single difficult mission/section with no
        determinable usable_range_m must not take down the whole
        analyze run. Falls back to the prior one-sided fit window
        (r_fit_min_m=None, starts at r_ref_m) instead."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, r_fit_min_m="auto")

        assert result.metrics["r_fit_min_m"] == result.metrics["r_ref_m"]  # one-sided fallback
        assert result.metrics["r_fit_min_m_override"] == "auto"
        assert any("falling back to the prior one-sided fit window" in n for n in result.notes)

    def test_r_fit_min_m_auto_uses_r_ref_search_fraction_of_usable_range_when_r_ref_m_none(self):
        """r_ref_m=None + usable_range_m given: auto mode's search bound
        falls back to r_ref_search_fraction * usable_range_m rather than
        requiring r_ref_m to be pinned."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_fit_min_m="auto")

        assert isinstance(result.metrics["r_fit_min_m"], float)

    def test_r_fit_min_m_auto_clamps_to_r_ref_m_when_naive_value_would_exceed_it(self):
        """Regression: the search
        restricts by *bin center* <= r_ref_m, but the returned value is
        that bin's far edge PLUS the margin -- on data whose true
        near-field peak sits close to r_ref_m itself (seen on real
        data, whose near-field ramp varies enough that it isn't reliably
        confined to the near half of the swath), the naive auto value
        can still land past r_ref_m even though the search never looked
        beyond it. Must clamp down to r_ref_m, not silently return a
        window that doesn't contain its own reference."""
        # Hand-built ramp (distinct from _make_full_physics_ds's
        # monotonic decay, whose peak always sits at range=0): peak at
        # range~19m, deliberately close to r_ref_m=20.0m below, so
        # peak_far_edge(~19.5m) + the 5.0m default margin (~24.5m)
        # would exceed r_ref_m without the clamp.
        range_m = np.arange(60.0) + 0.5
        amplitude = np.where(range_m < 19.0, range_m, np.maximum(38.0 - range_m, 0.1)).astype(np.float32)
        ds = xr.Dataset({
            "backscatter_intensity": ("sample", amplitude),
            "slant_range_m": ("sample", range_m),
        })

        result = stage4_decay(
            ds, bin_width_m=1.0, freq_hz=1_495_000.0, usable_range_m=40.0,
            r_ref_m=20.0, r_fit_min_m="auto", min_total_count=1,
        )

        assert result.metrics["r_fit_min_m"] == pytest.approx(20.0)
        assert any("clamped down to r_ref_m" in n for n in result.notes)

    def test_r_ref_m_auto_avoids_the_clamp_the_prior_test_needed(self):
        """2026-09-02, direct response to a real report: pinning
        r_ref_m independently (numeric, or the legacy "mid_swath") and
        clamping r_fit_min_m to fit around it lands the reference
        EXACTLY on the window's boundary when they collide (the prior
        test) -- often a noisy transition bin, since that's the region
        the window exists to exclude. r_ref_m="auto" fixes this at the
        root: resolve the window first, place the reference INSIDE it.
        Same pathological ramp as the clamp test above (peak close to
        where a pinned reference would land) -- this time the reference
        must NOT sit on either boundary."""
        range_m = np.arange(60.0) + 0.5
        amplitude = np.where(range_m < 19.0, range_m, np.maximum(38.0 - range_m, 0.1)).astype(np.float32)
        ds = xr.Dataset({
            "backscatter_intensity": ("sample", amplitude),
            "slant_range_m": ("sample", range_m),
        })

        result = stage4_decay(
            ds, bin_width_m=1.0, freq_hz=1_495_000.0, usable_range_m=40.0,
            r_ref_m="auto", r_fit_min_m="auto", min_total_count=1,
        )

        m = result.metrics
        assert m["r_fit_min_m"] < m["r_ref_m"] < m["r_fit_max_m"]  # strictly inside, not on a boundary
        assert not any("clamped down to r_ref_m" in n for n in result.notes)
        assert any("r_ref_m='auto' resolved to" in n for n in result.notes)

    def test_r_ref_m_auto_uses_window_fraction(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(
            ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="auto", r_fit_min_m=10.0,
            r_ref_window_fraction=0.25,
        )

        assert result.metrics["r_ref_m"] == pytest.approx(10.0 + 0.25 * (60.0 - 10.0))

    def test_r_ref_m_auto_without_usable_range_m_falls_back_gracefully(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, r_ref_m="auto")

        assert result.classification is not None  # completed, did not raise
        assert result.metrics["r_ref_m_override"] == "auto"
        assert any("falling back to the near-field auto-pick" in n for n in result.notes)

    def test_r_fit_min_m_invalid_string_raises(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        with pytest.raises(ValueError, match="must be 'auto'"):
            stage4_decay(ds, bin_width_m=2.0, r_ref_m=20.0, r_fit_min_m="near")

    def test_r_ref_m_mid_swath_resolves_to_half_usable_range(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="mid_swath")

        assert result.metrics["r_ref_m"] == pytest.approx(30.0)
        assert result.metrics["r_ref_m_override"] == "mid_swath"
        assert any("r_ref_m='mid_swath' resolved to" in n for n in result.notes)

    def test_r_ref_m_mid_swath_without_usable_range_m_falls_back_gracefully(self):
        """Graceful fallback instead of raising --
        "mid_swath" is a default (not just an opt-in
        override), so a single difficult mission/section with no
        determinable usable_range_m must not take down the whole
        analyze run. Falls back to the prior near-field auto-pick
        (r_ref_m=None) instead."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        result = stage4_decay(ds, bin_width_m=2.0, r_ref_m="mid_swath")

        assert result.classification is not None  # completed, did not raise
        assert result.metrics["r_ref_m_override"] == "mid_swath"
        assert any("falling back to the near-field auto-pick" in n for n in result.notes)

    def test_r_ref_m_invalid_string_raises(self):
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        with pytest.raises(ValueError, match="must be 'auto' or 'mid_swath'"):
            stage4_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="far_field")

    def test_mid_swath_paired_with_auto_keeps_reference_inside_fit_window(self):
        """The exact failure mode 'mid_swath' exists to prevent: left to
        auto-pick independently, r_ref_m (near-field peak bin itself)
        and r_fit_min_m="auto" (that same peak's far edge + margin)
        nearly always disagree about which side of the peak r_ref_m
        should be on -- r_ref_m ends up nearer than the fit window's own
        near edge, i.e. outside it. Pinning r_ref_m="mid_swath" must
        keep the resolved reference inside [r_fit_min_m, r_fit_max_m]."""
        ds = self._make_full_physics_ds(n_pings=60, n_samples=300)

        independent = stage4_decay(ds, bin_width_m=2.0, usable_range_m=60.0, r_fit_min_m="auto")
        paired = stage4_decay(
            ds, bin_width_m=2.0, usable_range_m=60.0, r_ref_m="mid_swath", r_fit_min_m="auto",
        )

        assert independent.metrics["r_ref_m"] < independent.metrics["r_fit_min_m"]  # the bug
        assert paired.metrics["r_fit_min_m"] <= paired.metrics["r_ref_m"] <= paired.metrics["r_fit_max_m"]

    def _make_two_channel_decay_ds(self, n_pings=60, n_samples=300):
        """channel 0 (port) follows the full uncompensated physics decay
        curve; channel 1 (starboard) is flat (already fully
        compensated, scale=1.0 everywhere) -- classifies as "none" vs.
        "full" respectively when fit independently. Pooled together,
        neither classification is expected to hold cleanly."""
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)
        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        ref = np.nanmin(loss_db[np.isfinite(loss_db)])
        port_amplitude = 10.0 ** (-(loss_db - ref) / 20.0)
        port_amplitude = np.nan_to_num(port_amplitude, nan=0.0).astype(np.float32)
        starboard_amplitude = np.where(np.isfinite(ground_range), 1.0, 0.0).astype(np.float32)

        bs = ds["backscatter_intensity"].values.copy()
        bs[:, 0, :] = port_amplitude[:, 0, :]
        bs[:, 1, :] = starboard_amplitude[:, 1, :]
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)
        return ds

    def test_channel_none_metrics_report_no_channel_restriction(self):
        ds = self._make_two_channel_decay_ds()

        pooled = stage4_decay(ds, bin_width_m=2.0)

        assert pooled.metrics["channel"] is None

    def test_channel_restricts_fit_to_one_side(self):
        ds = self._make_two_channel_decay_ds()

        port = stage4_decay(ds, bin_width_m=2.0, channel=0)
        starboard = stage4_decay(ds, bin_width_m=2.0, channel=1)

        assert port.metrics["channel"] == 0
        assert starboard.metrics["channel"] == 1
        assert port.classification == "none"  # nothing compensated yet
        assert starboard.classification == "full"  # already flat

    def test_channel_string_matches_int_position(self):
        ds = self._make_two_channel_decay_ds()

        by_string = stage4_decay(ds, bin_width_m=2.0, channel="starboard")
        by_int = stage4_decay(ds, bin_width_m=2.0, channel=1)

        assert by_string.classification == by_int.classification
        assert by_string.metrics["channel"] == 1

    def test_channel_invalid_string_raises(self):
        ds = self._make_two_channel_decay_ds()

        with pytest.raises(ValueError, match="not found in channel_id"):
            stage4_decay(ds, bin_width_m=2.0, channel="midship")

    def test_channel_pooled_differs_from_either_single_channel(self):
        ds = self._make_two_channel_decay_ds()

        pooled = stage4_decay(ds, bin_width_m=2.0)
        port = stage4_decay(ds, bin_width_m=2.0, channel=0)
        starboard = stage4_decay(ds, bin_width_m=2.0, channel=1)

        assert pooled.metrics["best_rmse_db"] != pytest.approx(port.metrics["best_rmse_db"])
        assert pooled.metrics["best_rmse_db"] != pytest.approx(starboard.metrics["best_rmse_db"])

    def test_exclude_first_n_bins_exceeding_available_bins_handled_gracefully(self):
        ds = self._make_full_physics_ds(n_pings=10, n_samples=60)

        result = stage4_decay(ds, bin_width_m=2.0, exclude_first_n_bins=10_000)

        assert result.classification == "ambiguous"
        assert "Not enough binned samples" in result.notes[0]
        assert result.metrics["exclude_first_n_bins"] == 10_000
        assert result.metrics["n_bins_excluded_after_bottom"] >= 0

    def test_reject_outliers_recovers_clean_fit_from_target_corrupted_curve(self):
        """Same full_physics curve as test_full_physics_curve_classified_as_none,
        but with a handful of huge point-target-like spikes injected --
        reject_outliers=False should be measurably degraded by them,
        reject_outliers=True should recover close to the clean fit."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)

        rng = np.random.default_rng(2)
        n_spikes = 40
        flat_idx = rng.choice(amplitude.size, size=n_spikes, replace=False)
        amplitude.flat[flat_idx] *= 50.0  # simulated hard-target/complex-terrain returns

        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        # range_var pinned to ground_range_m: the injected curve above is
        # built as an exact function of ground_range specifically, so the
        # fit must be evaluated against that same variable regardless of
        # stage4_decay's own (unrelated) 2026-09-02 default change to
        # slant_range_m.
        with_spikes = stage4_decay(ds, range_var="ground_range_m", bin_width_m=2.0)
        cleaned = stage4_decay(ds, range_var="ground_range_m", bin_width_m=2.0, reject_outliers=True)

        assert cleaned.metrics["outlier_n_rejected"] > 0
        assert cleaned.metrics["best_rmse_db"] < with_spikes.metrics["best_rmse_db"]
        assert cleaned.classification == "none"

    def test_fit_value_col_median_wired_through_and_recorded(self):
        """Contract test only -- real-data checking found `"median"`
        does NOT reliably improve RMSE over `"mean"` under
        reject_outliers=True (see stage4_decay's fit_value_col
        docstring), so this doesn't assert an RMSE direction, just that
        the parameter actually reaches fit_hardware_tvg_curve and gets
        recorded, and that classification stays correct either way."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        result = stage4_decay(ds, bin_width_m=2.0, reject_outliers=True, fit_value_col="median")

        assert result.metrics["fit_value_col"] == "median"
        assert result.classification == "none"
        assert result.metrics["best_candidate"] == "full_physics"

    def test_r_ref_m_override_is_used_instead_of_auto_selection(self):
        """Passing r_ref_m should reach fit_hardware_tvg_curve and pin
        the reported reference range exactly, not just influence it --
        on a clean full_physics curve the classification should stay
        correct at a manually-chosen mid-field reference too."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        auto = stage4_decay(ds, bin_width_m=2.0)
        explicit_r_ref_m = auto.metrics["r_ref_m"] + 5.0
        overridden = stage4_decay(ds, bin_width_m=2.0, r_ref_m=explicit_r_ref_m)

        assert overridden.metrics["r_ref_m"] == pytest.approx(explicit_r_ref_m)
        assert overridden.metrics["r_ref_m_override"] == pytest.approx(explicit_r_ref_m)
        assert overridden.metrics["r_ref_m"] != auto.metrics["r_ref_m"]
        assert overridden.metrics["best_candidate"] == "full_physics"

    def test_r_ref_m_override_defaults_to_none_when_auto_selecting(self):
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        result = stage4_decay(ds, bin_width_m=2.0)

        assert result.metrics["r_ref_m_override"] is None

    def test_r_ref_search_fraction_wired_through_to_fit(self):
        """Contract test: r_ref_search_fraction reaches
        fit_hardware_tvg_curve's own identically-named parameter (whose
        behavior is tested directly in test_tvg_diagnostics.py) --
        widening it away from the 0.5 default should be able to change
        the auto-selected r_ref_m on a curve with more than one local
        peak, without raising or otherwise misbehaving."""
        n_pings, n_samples = 30, 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        freq_hz = float(ds.attrs["freq_hz_start"])
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=alpha)
        ground_range = ds["ground_range_m"].values
        loss_db = tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
        amplitude = 10.0 ** (-(loss_db - loss_db[np.isfinite(loss_db)].min()) / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)

        narrow = stage4_decay(ds, bin_width_m=2.0, r_ref_search_fraction=0.1)
        wide = stage4_decay(ds, bin_width_m=2.0, r_ref_search_fraction=1.0)

        assert narrow.metrics["best_candidate"] == "full_physics"
        assert wide.metrics["best_candidate"] == "full_physics"


class TestDetectFirstBottomReturn:
    def test_detects_step_location(self):
        n_pings, n_channels, n_samples = 5, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., 300:] = 20.0  # sharp step to strong return at sample 300

        result = detect_first_bottom_return(arr, smooth_window=10, threshold_fraction=0.2, min_run=10)

        assert result.shape == (n_pings, n_channels)
        # Detector is smoothed (smooth_window=10), so it lands within a
        # window's width of the true step, not sample-exact.
        assert np.all(np.abs(result - 300) <= 15)

    def test_no_step_returns_nan_or_start(self):
        """Flat signal with no real step: the whole ping trivially
        clears the threshold from the start (nothing to distinguish),
        which is a legitimate, documented outcome -- not a crash."""
        arr = np.full((3, 2, 200), 1.0)
        result = detect_first_bottom_return(arr)
        assert result.shape == (3, 2)
        assert np.all(result < 20)  # detected essentially at the start, not NaN, not far out

    def test_short_ping_returns_all_nan(self):
        arr = np.full((2, 2, 5), 1.0)  # shorter than smooth_window + min_run
        result = detect_first_bottom_return(arr, smooth_window=10, min_run=10)
        assert result.shape == (2, 2)
        assert np.all(np.isnan(result))

    def test_threshold_fraction_is_not_knife_edge(self):
        """Real-data check found the detected index barely moves across
        threshold_fraction=0.1-0.3 -- reproduce that robustness on a
        synthetic step, matching the docstring's claim."""
        arr = np.full((1, 1, 500), 0.1)
        arr[..., 300:] = 20.0
        results = [
            detect_first_bottom_return(arr, threshold_fraction=tf)[0, 0]
            for tf in (0.1, 0.15, 0.2, 0.25, 0.3)
        ]
        assert max(results) - min(results) <= 15

    def test_min_sample_floor_skips_pre_altitude_spike(self):
        """Regression test for a real bug: a bright,
        sustained-enough spike before the vehicle's own altitude
        (residual nadir noise, or a thin non-bottom noise line) was
        winning as "first bottom return" -- physically impossible,
        confirmed on real data (detected sample 36, less than the
        nadir-noise burst's own detected extent of 43, and far short
        of the real altitude sample ~451). min_sample enforces the one
        absolute physical law: bottom can never be closer than
        altitude."""
        n_pings, n_channels, n_samples = 3, 2, 1000
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., 100:130] = 20.0  # bright pre-altitude spike (residual noise)
        arr[..., 600:] = 15.0     # real sustained bottom return, well past altitude

        no_floor = detect_first_bottom_return(arr, smooth_window=5, min_run=10)
        assert np.all(no_floor < 200)  # confirms the setup: without a floor, it's fooled

        altitude = np.full((n_pings, n_channels), 400.0)
        with_floor = detect_first_bottom_return(arr, smooth_window=5, min_run=10, min_sample=altitude)
        assert np.all(with_floor >= 400)
        assert np.all(np.abs(with_floor - 600) <= 15)

    def test_min_sample_scalar_applies_to_every_row(self):
        arr = np.full((2, 2, 500), 0.1)
        arr[..., 50:80] = 20.0   # before the scalar floor
        arr[..., 300:] = 15.0    # real return, past the floor

        result = detect_first_bottom_return(arr, smooth_window=5, min_run=10, min_sample=200.0)

        assert np.all(result >= 200)
        assert np.all(np.abs(result - 300) <= 15)

    def test_min_sample_nan_row_gets_no_floor(self):
        """A row with an unknown altitude (NaN) shouldn't be treated as
        'nothing is ever valid' -- falls back to no floor for that row."""
        arr = np.full((1, 1, 500), 0.1)
        arr[..., 50:80] = 20.0

        result = detect_first_bottom_return(arr, smooth_window=5, min_run=10, min_sample=np.array([[np.nan]]))

        assert np.abs(result[0, 0] - 50) <= 15

    def test_min_sample_buffer_skips_residual_noise_right_past_the_floor(self):
        """Regression test for a real finding: even
        with min_sample=altitude in place, a handful of bright points
        blank_nadir_noise didn't fully zero right at its own boundary's
        edge survived just past the floor -- bright/sustained enough to
        win as "first bottom return" instead of the real one further
        out, producing a per-ping trace that alternates between the
        floor and the real value rather than tracking smoothly.
        min_sample_buffer pushes the search past this residual band."""
        n_pings, n_channels, n_samples = 3, 2, 1000
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., 100:140] = 20.0  # residual noise straddling the floor
        arr[..., 400:] = 15.0     # real sustained bottom return, well past the residual band

        floor = np.full((n_pings, n_channels), 100.0)
        no_buffer = detect_first_bottom_return(arr, smooth_window=5, min_run=10, min_sample=floor)
        assert np.all(no_buffer < 200)  # confirms the setup: residual noise wins without a buffer

        with_buffer = detect_first_bottom_return(
            arr, smooth_window=5, min_run=10, min_sample=floor, min_sample_buffer=200.0,
        )
        assert np.all(with_buffer >= 300)
        assert np.all(np.abs(with_buffer - 400) <= 15)

    def test_peak_percentile_100_is_exact_prior_behavior(self):
        """peak_percentile's default (100.0) must be bit-for-bit
        equivalent to the pre-existing nanmax-based behavior -- every
        prior caller's results must not shift."""
        n_pings, n_channels, n_samples = 4, 2, 400
        rng = np.random.default_rng(7)
        arr = rng.uniform(0.05, 0.3, size=(n_pings, n_channels, n_samples))
        arr[..., 250:] = 10.0

        default = detect_first_bottom_return(arr, smooth_window=5, min_run=10)
        explicit_100 = detect_first_bottom_return(arr, smooth_window=5, min_run=10, peak_percentile=100.0)

        assert np.array_equal(default, explicit_100, equal_nan=True)

    def test_peak_percentile_below_100_is_robust_to_one_extreme_outlier_sample(self):
        """A single very bright, non-sustained outlier sample can
        dominate a plain nanmax peak reference enough that the real
        sustained region no longer clears threshold_fraction * peak --
        a lower peak_percentile excludes that outlier from the
        reference instead."""
        n_pings, n_channels, n_samples = 2, 2, 600
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., 300:500] = 15.0     # real sustained bottom return
        arr[..., 350] = 5000.0       # one extreme, isolated outlier sample within it

        with_outlier_default = detect_first_bottom_return(arr, smooth_window=5, min_run=10)
        assert np.all(np.isnan(with_outlier_default))  # confirms the setup: the outlier breaks detection entirely

        robust = detect_first_bottom_return(arr, smooth_window=5, min_run=10, peak_percentile=95.0)
        assert np.all(np.isfinite(robust))
        assert np.all(np.abs(robust - 300) <= 15)


class TestSmoothBoundaryAcrossPings:
    def test_rejects_isolated_outlier_ping(self):
        """Regression test for a real finding: raw per-ping
        detection has occasional large, physically implausible jumps
        (one ping reads very differently from its neighbors) that a
        median filter should reject without disturbing the surrounding
        constant baseline."""
        n_pings, n_channels = 41, 2
        boundary = np.full((n_pings, n_channels), 1000.0)
        boundary[20, :] = 4000.0  # single-ping outlier spike

        result = smooth_boundary_across_pings(boundary, window=15)

        assert result[20, 0] == pytest.approx(1000.0)
        assert np.all(result[:15, :] == pytest.approx(1000.0))
        assert np.all(result[25:, :] == pytest.approx(1000.0))

    def test_preserves_gradual_monotonic_trend(self):
        n_pings, n_channels = 100, 1
        ramp = np.linspace(2000.0, 500.0, n_pings)
        boundary = np.broadcast_to(ramp[:, None], (n_pings, n_channels)).copy()

        result = smooth_boundary_across_pings(boundary, window=15)

        # Away from the edges (where edge-padding distorts the ramp
        # slightly), smoothing a monotonic ramp should barely move it.
        mid = slice(30, 70)
        assert np.allclose(result[mid, 0], boundary[mid, 0], atol=50.0)
        assert np.all(np.diff(result[:, 0]) <= 1e-9)  # still monotonic non-increasing

    def test_even_window_rounds_up_to_odd(self):
        boundary = np.full((21, 1), 5.0)
        boundary[10, 0] = 999.0
        result_even = smooth_boundary_across_pings(boundary, window=14)
        result_odd = smooth_boundary_across_pings(boundary, window=15)
        assert np.array_equal(result_even, result_odd)

    def test_window_one_is_identity(self):
        rng = np.random.default_rng(3)
        boundary = rng.uniform(0, 1000, size=(30, 2))
        boundary[5, 0] = np.nan

        result = smooth_boundary_across_pings(boundary, window=1)

        assert np.array_equal(result, boundary, equal_nan=True)

    def test_all_nan_window_stays_nan(self):
        boundary = np.full((10, 1), np.nan)
        result = smooth_boundary_across_pings(boundary, window=5)
        assert np.all(np.isnan(result))

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            smooth_boundary_across_pings(np.zeros((5, 1)), window=0)


class TestDetectNadirNoiseExtent:
    def test_detects_burst_extent(self):
        n_pings, n_channels, n_samples = 5, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., :40] = 20.0  # noisy burst at the very start, real signal quiet after

        result = detect_nadir_noise_extent(arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10)

        assert result.shape == (n_pings, n_channels)
        assert np.all(np.abs(result - 40) <= 10)

    def test_already_zero_is_a_no_op(self):
        """EdgeTech/JSF case: burst already zeroed before sidescanpy
        sees the file -- detected extent should be 0, not a spurious
        nonzero value."""
        arr = np.full((3, 2, 200), 1.0)
        arr[..., :50] = 0.0

        result = detect_nadir_noise_extent(arr)

        assert np.all(result == 0)

    def test_bright_real_bottom_further_out_does_not_mask_a_short_burst(self):
        """search_window keeps the burst's own peak reference narrow --
        a real bottom return much further out and brighter than the
        burst shouldn't inflate the threshold and hide a short burst."""
        n_pings, n_channels, n_samples = 3, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., :20] = 5.0     # short noisy burst
        arr[..., 300:] = 50.0   # much brighter real bottom return, far away

        result = detect_nadir_noise_extent(arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10)

        assert np.all(result < 40)  # burst extent found near 20, not pulled out toward 300

    def test_short_ping_returns_all_nan(self):
        arr = np.full((2, 2, 5), 1.0)
        result = detect_nadir_noise_extent(arr, smooth_window=10, min_run=10)
        assert result.shape == (2, 2)
        assert np.all(np.isnan(result))

    def test_margin_samples_extends_the_detected_extent(self):
        n_pings, n_channels, n_samples = 4, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., :40] = 20.0

        base = detect_nadir_noise_extent(arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10)
        widened = detect_nadir_noise_extent(
            arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10, margin_samples=15,
        )

        assert np.array_equal(widened, base + 15)

    def test_margin_samples_does_not_touch_nan_rows(self):
        arr = np.full((2, 2, 5), 1.0)  # too short -- base result is all-NaN
        result = detect_nadir_noise_extent(arr, smooth_window=10, min_run=10, margin_samples=15)
        assert np.all(np.isnan(result))

    def test_min_extent_samples_raises_a_smaller_detection(self):
        n_pings, n_channels, n_samples = 4, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., :20] = 20.0  # detected extent will land near 20

        base = detect_nadir_noise_extent(arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10)
        assert np.all(base < 30)  # confirms the setup

        floored = detect_nadir_noise_extent(
            arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10, min_extent_samples=100.0,
        )
        assert np.all(floored == 100.0)

    def test_min_extent_samples_does_not_lower_a_larger_detection(self):
        n_pings, n_channels, n_samples = 4, 2, 500
        arr = np.full((n_pings, n_channels, n_samples), 0.1)
        arr[..., :40] = 20.0

        base = detect_nadir_noise_extent(arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10)
        floored = detect_nadir_noise_extent(
            arr, search_window=100, smooth_window=5, threshold_fraction=0.2, min_run=10, min_extent_samples=10.0,
        )
        assert np.array_equal(base, floored)  # floor is below the real detection, no effect

    def test_min_extent_samples_overrides_nan_rows_unlike_margin_samples(self):
        """The one deliberate behavioral difference from margin_samples:
        a user-configured minimum reflects known hardware behavior and
        should apply even when nothing was detected at all."""
        arr = np.full((2, 2, 5), 1.0)  # too short -- base result is all-NaN

        result = detect_nadir_noise_extent(arr, smooth_window=10, min_run=10, min_extent_samples=3.0)

        assert np.all(result == 3.0)


class TestBlankNadirNoise:
    def test_zeros_burst_leaves_rest_untouched(self):
        n_pings, n_samples = 5, 500
        arr = np.full((n_pings, 2, n_samples), 20.0, dtype=np.float32)
        arr[..., 40:] = 0.05  # quiet after the burst (a bright-nadir case, not a real strong return)
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        out = blank_nadir_noise(ds)

        raw = out["backscatter_intensity"].values
        intensity = out["backscatter_intensity"].values
        assert np.all(raw[:, :, :35] == 0.0)
        assert np.all(intensity[:, :, :35] == 0.0)
        assert np.all(raw[:, :, 45:] == pytest.approx(0.05))

    def test_no_op_when_already_zero(self):
        """EdgeTech/JSF case -- must not touch data that's already
        blanked by the hardware."""
        n_pings, n_samples = 3, 300
        arr = np.full((n_pings, 2, n_samples), 1.0, dtype=np.float32)
        arr[..., :50] = 0.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        out = blank_nadir_noise(ds)

        assert np.array_equal(out["backscatter_intensity"].values, ds["backscatter_intensity"].values)

    def test_original_dataset_not_mutated(self):
        n_pings, n_samples = 3, 300
        arr = np.full((n_pings, 2, n_samples), 20.0, dtype=np.float32)
        arr[..., 40:] = 0.05
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        blank_nadir_noise(ds)

        assert np.all(ds["backscatter_intensity"].values[:, :, :10] == 20.0)

    def test_margin_samples_zeros_a_wider_region(self):
        """Regression test for the real finding motivating
        margin_samples: a few bright points right at the detected
        boundary's edge weren't fully zeroed by the base detection --
        a margin sweeps them in too."""
        n_pings, n_samples = 3, 500
        arr = np.full((n_pings, 2, n_samples), 20.0, dtype=np.float32)
        arr[..., 40:] = 0.05
        arr[..., 42:45] = 8.0  # residual bright points just past the plain-detected boundary
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        out = blank_nadir_noise(ds, margin_samples=10)

        raw = out["backscatter_intensity"].values
        assert np.all(raw[:, :, :50] == 0.0)  # the residual band at 42:45 is now zeroed too
        assert np.all(raw[:, :, 55:] == pytest.approx(0.05))

    def test_min_extent_samples_zeros_at_least_that_many_samples(self):
        n_pings, n_samples = 3, 500
        arr = np.full((n_pings, 2, n_samples), 20.0, dtype=np.float32)
        arr[..., 20:] = 0.05  # real detection would land near sample 20

        out = blank_nadir_noise(
            _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy()),
            min_extent_samples=80.0,
        )

        raw = out["backscatter_intensity"].values
        assert np.all(raw[:, :, :80] == 0.0)
        assert np.all(raw[:, :, 85:] == pytest.approx(0.05))


class TestMaskNadirGap:
    def test_masks_before_boundary_leaves_rest_untouched(self):
        n_pings, n_samples = 5, 500
        arr = np.full((n_pings, 2, n_samples), 0.1, dtype=np.float32)
        arr[..., 300:] = 20.0  # sharp step to strong return at sample 300
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        out = mask_nadir_gap(ds)

        raw = out["backscatter_intensity"].values
        intensity = out["backscatter_intensity"].values
        # Well before the detected boundary: masked in both vars.
        assert np.all(np.isnan(raw[:, :, :50]))
        assert np.all(np.isnan(intensity[:, :, :50]))
        # Well past it: untouched, original strong-return value intact.
        assert np.all(raw[:, :, 450:] == 20.0)
        assert np.all(intensity[:, :, 450:] == 20.0)

    def test_only_masks_vars_present(self):
        n_pings, n_samples = 3, 500
        arr = np.full((n_pings, 2, n_samples), 0.1, dtype=np.float32)
        arr[..., 300:] = 20.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        out = mask_nadir_gap(ds, value_vars=("backscatter_intensity", "not_present"))

        assert "not_present" not in out.data_vars
        assert np.any(np.isnan(out["backscatter_intensity"].values[:, :, :50]))

    def test_original_dataset_not_mutated(self):
        n_pings, n_samples = 3, 500
        arr = np.full((n_pings, 2, n_samples), 0.1, dtype=np.float32)
        arr[..., 300:] = 20.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        mask_nadir_gap(ds)

        assert not np.any(np.isnan(ds["backscatter_intensity"].values))

    def test_pre_altitude_spike_does_not_win_the_boundary(self):
        """mask_nadir_gap computes its own boundary independently of
        stage2_nadir -- must apply the same altitude floor, not just
        stage2_nadir. altitude=10.0 (fixture default) -> altitude
        sample ~14."""
        n_pings, n_samples = 10, 200
        arr = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        arr[..., 5:12] = 20.0   # bright pre-altitude spike
        arr[..., 100:] = 15.0   # real bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())
        ds = compute_geometric_ranges(ds)

        out = mask_nadir_gap(ds, smooth_window=3)

        # The pre-altitude spike itself should be masked (NaN), not
        # left as the boundary the rest of the ping is measured from.
        assert np.all(np.isnan(out["backscatter_intensity"].values[:, :, 5:12]))
        assert np.all(out["backscatter_intensity"].values[:, :, 150:] == 15.0)

    def test_min_first_bottom_return_sample_widens_the_mask(self):
        """Regression coverage for downstream sweep/ping-profile charts need mask_nadir_gap's
        boundary to respect the same explicit floor stage2_nadir/
        detect_first_bottom_return already support, so the two stay
        consistent when called with the same override."""
        n_pings, n_samples = 10, 400
        arr = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        arr[..., 20:60] = 20.0   # residual noise, would otherwise win as the boundary
        arr[..., 200:] = 15.0    # real bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        without_floor = mask_nadir_gap(ds, smooth_window=5)
        assert not np.all(np.isnan(without_floor["backscatter_intensity"].values[:, :, 20:60]))  # confirms setup

        with_floor = mask_nadir_gap(ds, smooth_window=5, min_first_bottom_return_sample=150.0)
        assert np.all(np.isnan(with_floor["backscatter_intensity"].values[:, :, :150]))
        assert np.all(with_floor["backscatter_intensity"].values[:, :, 250:] == 15.0)

    def test_min_sample_buffer_and_peak_percentile_are_no_ops_by_default(self):
        """0.0 / 100.0 must reproduce this function's exact prior
        behavior -- confirmed by identity against a call that doesn't
        pass them at all."""
        n_pings, n_samples = 5, 300
        arr = np.full((n_pings, 2, n_samples), 0.1, dtype=np.float32)
        arr[..., 150:] = 10.0
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=arr.copy())

        default = mask_nadir_gap(ds)
        explicit = mask_nadir_gap(ds, min_sample_buffer=0.0, peak_percentile=100.0)

        assert np.array_equal(default["backscatter_intensity"].values, explicit["backscatter_intensity"].values, equal_nan=True)


class TestUniformExclusionBoundary:
    def test_collapses_to_single_percentile_value(self):
        per_ping = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        result = uniform_exclusion_boundary(per_ping, percentile=90.0)

        assert result.shape == per_ping.shape
        assert np.all(result == result[0, 0])  # single value everywhere
        assert result[0, 0] == pytest.approx(np.percentile(per_ping, 90.0))

    def test_all_nan_input_stays_all_nan(self):
        per_ping = np.full((3, 2), np.nan)
        result = uniform_exclusion_boundary(per_ping)
        assert result.shape == (3, 2)
        assert np.all(np.isnan(result))

    def test_avoids_near_field_selection_bias_vs_per_ping_boundary(self):
        """The actual bug this function exists to fix: real backscatter
        physics correlates a ping's boundary location with its own
        near-field amplitude (shallower altitude -> both an earlier
        strong-return onset AND a stronger near-field signal). Feeding
        a per-ping-varying boundary straight into aggregate range
        binning then selectively keeps only the strong,
        early-boundary pings' data in near-range bins, inflating their
        mean above the survey's true (decaying) level -- a real
        artifact found via real-data testing, reproduced here with
        correlated synthetic physics. A uniform boundary should not
        exhibit it."""
        n_pings, n_channels, n_samples = 40, 2, 400
        rng = np.random.default_rng(5)

        # Each ping's own "true" near-field amplitude is inversely tied
        # to its own boundary sample (shallower altitude -> earlier
        # boundary -> stronger near return), matching real physics.
        boundary = rng.integers(100, 300, size=(n_pings, n_channels)).astype(np.float64)
        near_field_level = 1000.0 / boundary  # smaller boundary -> larger amplitude

        sample_idx = np.arange(n_samples)
        decay = np.exp(-sample_idx / 150.0)  # generic range decay, same shape for every ping
        arr = near_field_level[..., None] * decay[None, None, :]
        arr += rng.normal(0, 0.02, arr.shape)
        arr[sample_idx[None, None, :] < boundary[..., None]] = 0.01  # weak zone, per-ping varying extent

        ranges = np.broadcast_to(np.linspace(0, 40, n_samples), arr.shape).copy()

        import xarray as xr

        from sidescanpy.analysis.investigation import _mask_before_sample
        from sidescanpy.analysis.tvg_diagnostics import mean_backscatter_vs_range

        def _bin(masked):
            ds = xr.Dataset(
                {"_v": (("ping", "channel", "sample"), masked), "_r": (("ping", "channel", "sample"), ranges)}
            )
            return mean_backscatter_vs_range(ds, value_var="_v", range_var="_r", bin_width_m=2.0)

        per_ping_binned = _bin(_mask_before_sample(arr, boundary))
        uniform_binned = _bin(_mask_before_sample(arr, uniform_exclusion_boundary(boundary, percentile=90.0)))

        # With the per-ping boundary, the first non-empty near-range bin
        # is populated only by the strongest (earliest-boundary) pings,
        # so its mean is far above the next bin out -- not monotonic.
        per_ping_nonempty = per_ping_binned[per_ping_binned["count"] > 0].reset_index(drop=True)
        assert per_ping_nonempty.loc[0, "mean"] > per_ping_nonempty.loc[1, "mean"]  # non-monotonic: the bug

        # With a uniform boundary, no ping contributes to a bin its own
        # boundary hasn't cleared -- the surviving near bins reflect a
        # smooth decay instead of a selection-biased spike.
        uniform_nonempty = uniform_binned[uniform_binned["count"] > 0].reset_index(drop=True)
        assert uniform_nonempty["mean"].is_monotonic_decreasing


class TestRejectRangeOutliers:
    def test_flags_injected_spikes_leaves_normal_values(self):
        rng = np.random.default_rng(0)
        n = 2000
        ranges = rng.uniform(0, 100, size=n)
        values = 10.0 + rng.normal(0, 0.5, size=n)  # flat "sandy bottom" signal

        spike_idx = rng.choice(n, size=20, replace=False)
        values[spike_idx] += 500.0  # far outside the local population

        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5)

        assert np.all(np.isnan(cleaned[spike_idx]))
        non_spike = np.setdiff1d(np.arange(n), spike_idx)
        # Overwhelming majority of the clean population should survive --
        # a small amount of edge-case rejection near bin boundaries is
        # expected and fine, a mass rejection would not be.
        assert np.isfinite(cleaned[non_spike]).mean() > 0.95

    def test_small_bins_left_unfiltered(self):
        ranges = np.array([1.0, 1.0, 1.0])  # below min_bin_count
        values = np.array([1.0, 2.0, 1000.0])
        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5, min_bin_count=5)
        np.testing.assert_array_equal(cleaned, values)

    def test_degenerate_zero_mad_bin_left_unfiltered(self):
        ranges = np.full(10, 5.0)
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0])
        cleaned = reject_range_outliers(values, ranges, bin_width_m=10.0, mad_k=2.5, min_bin_count=5)
        # Median of the mostly-identical bin has MAD=0 -- no scale to
        # reject against, so nothing gets flagged even though 1000.0
        # is visibly anomalous. Documents the known limitation rather
        # than silently disagreeing with it.
        np.testing.assert_array_equal(cleaned, values)

    def test_shape_preserved(self):
        rng = np.random.default_rng(1)
        values = rng.uniform(0, 10, size=(4, 2, 50))
        ranges = rng.uniform(0, 60, size=(4, 2, 50))
        cleaned = reject_range_outliers(values, ranges, bin_width_m=5.0)
        assert cleaned.shape == values.shape


class TestStage5Units:
    def test_db_like_values_classified_as_db(self):
        s1 = stage1_characterize(
            _make_raw_dataset(backscatter=np.full((3, 2, 10), -40.0, dtype=np.float32))
        )
        result = stage5_units(s1)
        assert result.classification == "dB"

    def test_raw_counts_classified(self):
        bs = np.zeros((3, 2, 10), dtype=np.float32)
        bs[..., 5:] = 81.0
        s1 = stage1_characterize(_make_raw_dataset(backscatter=bs))
        result = stage5_units(s1)
        assert result.classification == "raw_counts"

    def test_normalized_classified(self):
        bs = np.full((3, 2, 10), 0.5, dtype=np.float32)
        s1 = stage1_characterize(_make_raw_dataset(backscatter=bs))
        result = stage5_units(s1)
        assert result.classification == "normalized"


class TestInvertForInvestigation:
    def test_legacy_radiometric_file_restores_backscatter_raw_and_drops_geometry(self):
        ds = _make_raw_dataset()
        ds["backscatter_raw"] = ds["backscatter_intensity"].copy()
        ds = compute_geometric_ranges(ds)
        ds.attrs["processing_level"] = "radiometric"  # legacy: TVG known only from the attribute
        ds["backscatter_intensity"] = ds["backscatter_intensity"] * 0.0 + 999.0

        restored = invert(ds)

        assert restored.sidescan.processing_level == "raw"
        assert "processing_level" not in restored.attrs
        np.testing.assert_array_equal(
            restored["backscatter_intensity"].values, restored["backscatter_raw"].values
        )
        assert "ground_range_m" not in restored


class TestAutoNearEdgeM:
    """Direct tests of `_auto_near_edge_m`, the helper `stage4_decay`'s
    `r_fit_min_m="auto"` delegates to -- uses `mean_backscatter_vs_range`'s
    own documented contract ("any dataset with value_var and range_var,
    same shape, arbitrary dims") to build a minimal synthetic ramp
    directly, rather than routing through the full raw-dataset/geometric
    pipeline the rest of this file uses."""

    def _make_ramp_ds(self) -> xr.Dataset:
        # 1m-wide "pings" from 0 to 39m, amplitude rises to a clear peak
        # at range 10-11m then falls -- a deliberately non-monotonic
        # near-field ramp, distinct from the pure-decay curves the rest
        # of this file's fixtures use.
        range_m = np.arange(40.0) + 0.5
        amplitude = np.where(range_m < 11.0, range_m, np.maximum(22.0 - range_m, 0.1))
        return xr.Dataset(
            {"value": ("sample", amplitude), "range_m": ("sample", range_m)}
        )

    def test_finds_peak_bin_far_edge_plus_margin(self):
        ds = self._make_ramp_ds()

        result = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=40.0, margin_m=5.0,
        )

        # Peak bin covers [10, 11) -> far edge 11.0, + 5.0m margin.
        assert result == pytest.approx(16.0)

    def test_search_max_m_restricts_the_search_window(self):
        """A search bound tighter than where the true peak sits should
        still resolve, just against whatever's the best bin inside the
        window -- not silently ignore the bound."""
        ds = self._make_ramp_ds()

        restricted = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=5.0, margin_m=5.0,
        )

        # Within [0, 5], amplitude is still monotonically rising (peak
        # is at 10-11m, outside this window) -- best bin is the last one.
        assert restricted == pytest.approx(5.0 + 5.0)

    def test_no_bins_in_search_window_returns_none(self):
        ds = self._make_ramp_ds()

        result = _auto_near_edge_m(
            ds, value_var="value", range_var="range_m", bin_width_m=1.0,
            search_max_m=-1.0, margin_m=5.0,
        )

        assert result is None


class TestWaterColumnExclusionIndependentOfRangeVar:
    """2026-09-02: stage3_usable_range/stage4_decay's range_var default
    changed from ground_range_m to slant_range_m (the physics-correct
    choice for a TVG loss-vs-range fit -- matches what apply_tvg() itself
    uses). But water-column exclusion must keep working regardless --
    slant_range_m is pure time-of-flight with no NaNs of its own, so it
    can't detect the water column the way ground_range_m's own NaN
    pattern can. These confirm the decoupling fix actually holds (water
    column exclusion stays keyed to ground_range_m internally), not just
    that the range_var default flipped."""

    def test_stage3_water_column_count_matches_regardless_of_range_var(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        via_slant = stage3_usable_range(ds, range_var="slant_range_m")
        via_ground = stage3_usable_range(ds, range_var="ground_range_m")

        assert via_slant.metrics["water_column_n_excluded"] > 0
        assert via_slant.metrics["water_column_n_excluded"] == via_ground.metrics["water_column_n_excluded"]
        assert via_slant.metrics["range_var"] == "slant_range_m"

    def test_stage4_water_column_count_matches_regardless_of_range_var(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        via_slant = stage4_decay(ds, range_var="slant_range_m", bin_width_m=2.0)
        via_ground = stage4_decay(ds, range_var="ground_range_m", bin_width_m=2.0)

        assert via_slant.metrics["water_column_n_excluded"] > 0
        assert via_slant.metrics["water_column_n_excluded"] == via_ground.metrics["water_column_n_excluded"]
        assert via_slant.metrics["range_var"] == "slant_range_m"

    def test_stage3_garbage_in_water_column_does_not_skew_noise_floor_with_slant_range_default(self):
        """Direct behavioral proof, not just a count match: a huge
        garbage value placed only in the water column must not appear in
        the binned curve when range_var defaults to slant_range_m --
        would have leaked straight in before this fix, biasing the
        noise-floor estimate."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)
        water_column_mask = ~np.isfinite(ds["ground_range_m"].values)
        assert water_column_mask.any(), "fixture must have a real water column for this test to mean anything"

        bs = ds["backscatter_intensity"].values.copy()
        bs[water_column_mask] = 1e6
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        result = stage3_usable_range(ds, bin_width_m=2.0)  # default range_var=slant_range_m
        binned = result.data["binned"]
        assert not binned.empty
        assert binned["mean"].max() < 200.0, (
            f"garbage water-column value leaked into the binned curve: max bin mean = {binned['mean'].max()}"
        )

    def test_stage4_garbage_in_water_column_does_not_appear_in_fit_curve_with_slant_range_default(self):
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)
        water_column_mask = ~np.isfinite(ds["ground_range_m"].values)
        assert water_column_mask.any()

        bs = ds["backscatter_intensity"].values.copy()
        bs[water_column_mask] = 1e6
        ds["backscatter_intensity"] = (["ping", "channel", "sample"], bs)

        result = stage4_decay(ds, bin_width_m=2.0)  # default range_var=slant_range_m
        binned = result.data["binned"]
        assert not binned.empty
        assert binned["mean"].max() < 200.0, (
            f"garbage water-column value leaked into the fit curve: max bin mean = {binned['mean'].max()}"
        )
