"""
Unit tests for sidescanpy.analysis.nadir.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest

from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestAnalyzeNadir:
    def test_detects_gap(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        bs[..., :50] = 0.0  # leading zero run on every ping/channel
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)

        result = analyze_nadir(ds, nadir_n_samples=50)

        assert result.nadir_type == "gap"
        assert result.nadir_pct_zero == 100.0

    def test_detects_bright_spike(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        bs[..., :50] = 10.0  # bright near-range return, no zeros
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)

        result = analyze_nadir(ds, nadir_n_samples=50)

        assert result.nadir_type == "bright_spike"

    def test_detects_clean(self):
        bs = np.full((5, 2, 200), 1.0, dtype=np.float32)
        ds = _make_raw_dataset(n_pings=5, n_samples=200, backscatter=bs)
        result = analyze_nadir(ds, nadir_n_samples=50)
        assert result.nadir_type == "clean"

    def test_water_column_boundary_skipped_without_range_var(self):
        """Must still run and classify the nadir gap even when geometric
        ranges haven't been computed -- the altitude sample is a bonus,
        not a requirement, for backward compatibility."""
        ds = _make_raw_dataset(n_pings=5, n_samples=200)
        result = analyze_nadir(ds, nadir_n_samples=50)

        assert result.nadir_type is not None
        assert result.altitude_sample_median is None
        assert any("ground_range_m" in n for n in result.notes)

    def test_altitude_sample_matches_geometry(self):
        """altitude_m=10.0, sound_vel_ms=1500.0, sample_rate_hz=1000.0
        (fixture defaults) -> slant_range = sample_idx * 0.75m ->
        slant_range crosses altitude once sample index > 13.33 -> first
        finite ground_range sample = 14 (not a real bottom-contact
        boundary -- just altitude re-expressed in sample units)."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        result = analyze_nadir(ds, nadir_n_samples=5)

        assert result.altitude_sample_median == pytest.approx(14.0, abs=1.0)
        altitude_arr = result.altitude_sample
        assert altitude_arr.shape == (20, 2)
        assert np.all(np.isfinite(altitude_arr))

    def test_strong_return_min_sample_buffer_skips_residual_noise_past_altitude(self):
        """Integration-level regression test for the real
        finding: residual noise right past the altitude
        floor won as first_bottom_return_sample instead of the real
        bottom return further out. strong_return_min_sample_buffer
        (wired through to detect_first_bottom_return's min_sample_buffer)
        fixes it at the analyze_nadir level, not just the unit level."""
        n_pings, n_samples = 20, 600
        bs = np.full((n_pings, 2, n_samples), 0.05, dtype=np.float32)
        bs[..., 20:50] = 20.0   # residual noise just past altitude (~14)
        bs[..., 300:] = 15.0    # real sustained bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        without_buffer = analyze_nadir(ds, nadir_n_samples=5)
        assert without_buffer.first_bottom_return_sample_median < 100  # confirms the setup

        with_buffer = analyze_nadir(ds, nadir_n_samples=5, strong_return_min_sample_buffer=150.0)
        assert with_buffer.first_bottom_return_sample_median == pytest.approx(300, abs=15)

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

        without_floor = analyze_nadir(ds, nadir_n_samples=5)
        assert without_floor.first_bottom_return_sample_median < 100  # confirms the setup

        with_floor = analyze_nadir(ds, nadir_n_samples=5, min_first_bottom_return_sample=200.0)
        assert with_floor.first_bottom_return_sample_median == pytest.approx(300, abs=15)

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

        without_floor = analyze_nadir(ds, nadir_n_samples=5)
        assert without_floor.first_bottom_return_sample_median < 100  # confirms the setup

        with_floor = analyze_nadir(ds, nadir_n_samples=5, min_first_bottom_return_sample=150.0)
        assert with_floor.first_bottom_return_sample_median == pytest.approx(200, abs=15)

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

        result = analyze_nadir(ds, nadir_n_samples=2)

        assert any("never reaches the vehicle's own altitude" in n for n in result.notes)

    def test_strong_return_boundary_too_close_to_start_skips_step_check(self):
        """Constant backscatter everywhere: detect_first_bottom_return
        correctly reports the strong return starting essentially at
        sample 0 (there's no water column to distinguish it from) --
        too close to the ping start for a pre/post step check, which
        should be flagged as such rather than silently computing NaN."""
        ds = _make_raw_dataset(n_pings=20, n_samples=200)
        ds = compute_geometric_ranges(ds)

        result = analyze_nadir(ds, nadir_n_samples=5)

        assert result.first_bottom_return_sample_median < 20
        assert result.signal_step_ratio is None
        assert any("too close to the start" in n for n in result.notes)

    def test_signal_step_no_note_with_real_step(self):
        n_pings, n_samples = 20, 300
        bs = np.full((n_pings, 2, n_samples), 1.0, dtype=np.float32)
        bs[..., :100] = 0.01  # weak water column, long enough for a pre/post window
        bs[..., 100:] = 5.0   # strong bottom return
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples, backscatter=bs)
        ds = compute_geometric_ranges(ds)

        result = analyze_nadir(ds, nadir_n_samples=5)

        assert result.signal_step_ratio > 1.2
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

        result = analyze_nadir(ds, nadir_n_samples=5)

        assert result.altitude_sample_median < 20
        assert result.first_bottom_return_sample_median > 100
        assert result.altitude_to_bottom_gap_sample_median > 100
        assert any("disagree by a median of" in n for n in result.notes)

    def test_pre_altitude_spike_does_not_win_as_first_bottom_return(self):
        """Integration-level regression test for a real
        bug: analyze_nadir must pass its own altitude_sample through to
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

        result = analyze_nadir(ds, nadir_n_samples=5, strong_return_smooth_window=3)

        assert result.first_bottom_return_sample_median >= 14
        assert result.first_bottom_return_sample_median > 50

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

        smoothed = analyze_nadir(ds, nadir_n_samples=5)
        raw = analyze_nadir(ds, nadir_n_samples=5, boundary_smoothing_window=1)

        smoothed_boundary = smoothed.first_bottom_return_sample
        raw_boundary = raw.first_bottom_return_sample

        assert abs(raw_boundary[20, 0] - 180) <= 15
        assert abs(smoothed_boundary[20, 0] - 100) <= 15
