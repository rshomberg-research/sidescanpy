"""
Unit tests for sidescanpy.processing.bottom.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.history import read_history
from sidescanpy.processing.bottom import (
    altitude_from_bottom_track,
    blank_nadir_noise,
    detect_first_bottom_return,
    detect_nadir_noise_extent,
    mask_before_sample,
    mask_nadir_gap,
    smooth_boundary_across_pings,
    uniform_exclusion_boundary,
)
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.profiles import backscatter_vs_range
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


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
        analyze_nadir -- must apply the same altitude floor, not just
        analyze_nadir. altitude=10.0 (fixture default) -> altitude
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
        boundary to respect the same explicit floor analyze_nadir/
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

        def _bin(masked):
            ds = xr.Dataset(
                {"_v": (("ping", "channel", "sample"), masked), "_r": (("ping", "channel", "sample"), ranges)}
            )
            return backscatter_vs_range(ds, value_var="_v", range_var="_r", bin_width_m=2.0)

        per_ping_binned = _bin(mask_before_sample(arr, boundary))
        uniform_binned = _bin(mask_before_sample(arr, uniform_exclusion_boundary(boundary, percentile=90.0)))

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


# 1000 Hz at 1500 m/s: 0.75 m of slant range per sample.
_DR = 0.75


def _tracked_dataset(onsets, altitude=None, stbd_offset=0, burst=3):
    """Dataset whose bottom return starts at ``onsets[i]`` (port) per ping,
    with a transmit burst in the first ``burst`` samples of every ping."""
    n_pings, n_samples = len(onsets), 200
    bs = np.full((n_pings, 2, n_samples), 0.01, dtype=np.float32)
    for i, k in enumerate(onsets):
        bs[i, 0, k:] = 1.0
        bs[i, 1, k + stbd_offset:] = 1.0
    bs[:, :, :burst] = 50.0  # transmit burst, nearer than min_altitude_m
    ds = _make_raw_dataset(backscatter=bs)
    if altitude is not None:
        ds["altitude_m"] = ("ping", np.asarray(altitude, dtype=np.float32))
    return ds


class TestAltitudeFromBottomTrack:
    def test_fills_nan_and_zero_altitude(self):
        onsets = [40, 42, 44, 46]
        ds = _tracked_dataset(onsets, altitude=[np.nan, 0.0, -1.0, np.nan])
        out = altitude_from_bottom_track(ds, min_altitude_m=3.0)
        np.testing.assert_allclose(out["altitude_m"].values, np.array(onsets) * _DR, atol=_DR)

    def test_keeps_valid_recorded_altitude_by_default(self):
        ds = _tracked_dataset([40, 40, 40], altitude=[12.0, np.nan, 50.0])
        out = altitude_from_bottom_track(ds, min_altitude_m=3.0)
        alt = out["altitude_m"].values
        assert alt[0] == pytest.approx(12.0)
        assert alt[1] == pytest.approx(40 * _DR, abs=_DR)
        assert alt[2] == pytest.approx(50.0)

    def test_max_diff_replaces_implausible_recorded_altitude(self):
        ds = _tracked_dataset([40, 40, 40], altitude=[30.5, 30.0, 50.0])
        out = altitude_from_bottom_track(ds, min_altitude_m=3.0, max_diff_m=5.0)
        alt = out["altitude_m"].values
        assert alt[0] == pytest.approx(30.5) and alt[1] == pytest.approx(30.0)
        assert alt[2] == pytest.approx(40 * _DR, abs=_DR)

    def test_replace_all(self):
        ds = _tracked_dataset([40, 60], altitude=[12.0, 12.0])
        out = altitude_from_bottom_track(ds, replace="all", min_altitude_m=3.0)
        np.testing.assert_allclose(out["altitude_m"].values, [40 * _DR, 60 * _DR], atol=_DR)

    def test_uses_nearer_channel(self):
        ds = _tracked_dataset([40, 40], altitude=[np.nan, np.nan], stbd_offset=20)
        out = altitude_from_bottom_track(ds, min_altitude_m=3.0)
        np.testing.assert_allclose(out["altitude_m"].values, 40 * _DR, atol=_DR)

    def test_transmit_burst_is_skipped(self):
        # A burst long enough to look like a sustained return (12 samples, 9 m).
        ds = _tracked_dataset([40], altitude=[np.nan], burst=12)
        skipped = altitude_from_bottom_track(ds, min_altitude_m=10.0)["altitude_m"].values[0]
        assert skipped == pytest.approx(40 * _DR, abs=_DR)
        burst_hit = altitude_from_bottom_track(ds, min_altitude_m=0.0)["altitude_m"].values[0]
        assert not burst_hit == pytest.approx(40 * _DR, abs=_DR)

    def test_no_return_keeps_value_and_warns(self, caplog):
        ds = _tracked_dataset([40, 40], altitude=[np.nan, np.nan])
        ds["backscatter_intensity"][1] = 0.01
        with caplog.at_level("WARNING", logger="sidescanpy.processing.bottom"):
            out = altitude_from_bottom_track(ds, min_altitude_m=3.0)
        assert np.isfinite(out["altitude_m"].values[0])
        assert np.isnan(out["altitude_m"].values[1])
        assert any("no bottom return" in r.getMessage() for r in caplog.records)

    def test_history_record_and_backscatter_unchanged(self):
        ds = _tracked_dataset([40], altitude=[np.nan])
        out = altitude_from_bottom_track(ds, min_altitude_m=3.0)
        record = read_history(out)[-1]
        assert record["step"] == "bottom_track"
        assert record["params"]["replace"] == "missing"
        np.testing.assert_array_equal(out["backscatter_intensity"].values, ds["backscatter_intensity"].values)

    def test_geometric_after_tracking_has_ground_range(self):
        ds = _tracked_dataset([40, 40], altitude=[np.nan, np.nan])
        assert np.isnan(compute_geometric_ranges(ds)["ground_range_m"].values).all()
        out = compute_geometric_ranges(altitude_from_bottom_track(ds, min_altitude_m=3.0))
        assert np.isfinite(out["ground_range_m"].values).any()

    def test_bad_arguments_raise(self):
        ds = _tracked_dataset([40])
        with pytest.raises(ValueError, match="replace"):
            altitude_from_bottom_track(ds, replace="sometimes")
        with pytest.raises(ValueError, match="max_diff_m"):
            altitude_from_bottom_track(ds, replace="all", max_diff_m=1.0)


class TestAltitudeFromBottomTrackFineSampling:
    """0.075 m per sample, so ``min_return_m`` spans many samples."""

    DR = 0.075

    def _ds(self, profile):
        bs = np.repeat(np.asarray(profile, dtype=np.float32)[None, None, :], 2, axis=1).repeat(3, axis=0)
        ds = _make_raw_dataset(backscatter=bs, sample_rate_hz=10_000.0)
        ds["altitude_m"] = ("ping", np.full(3, np.nan, dtype=np.float32))
        return ds

    def test_bright_burst_after_quiet_lead_in_is_not_the_bottom(self):
        # Quiet to 1 m, a 2 m burst as bright as the seabed, quiet, seabed from 30 m.
        profile = np.full(1000, 0.05)
        profile[13:40] = 1.0
        profile[400:] = 1.0
        out = altitude_from_bottom_track(self._ds(profile))
        np.testing.assert_allclose(out["altitude_m"].values, 400 * self.DR, atol=2 * self.DR)

    def test_gaps_in_the_return_do_not_push_the_onset_out(self):
        profile = np.full(1000, 0.05)
        profile[400:] = 1.0
        for start in range(430, 1000, 60):  # 1.5 m shadows every 4.5 m
            profile[start:start + 20] = 0.05
        out = altitude_from_bottom_track(self._ds(profile))
        np.testing.assert_allclose(out["altitude_m"].values, 400 * self.DR, atol=2 * self.DR)


class TestMinRunFraction:
    def test_default_requires_every_sample(self):
        arr = np.full((1, 1, 300), 0.1)
        arr[..., 100:] = 10.0
        arr[..., 105] = 0.1  # one dip right after the onset
        strict = detect_first_bottom_return(arr, smooth_window=1, min_run=20)
        loose = detect_first_bottom_return(arr, smooth_window=1, min_run=20, min_run_fraction=0.9)
        assert strict[0, 0] == 106
        assert loose[0, 0] == 100


def test_geometric_warns_on_missing_or_zero_altitude(caplog):
    ds = _make_raw_dataset(n_pings=4)
    ds["altitude_m"] = ("ping", np.array([np.nan, 0.0, 10.0, 10.0], dtype=np.float32))
    with caplog.at_level("WARNING", logger="sidescanpy.processing.geometric"):
        compute_geometric_ranges(ds)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("1 of 4 pings have NaN altitude_m" in m and "1 have altitude_m <= 0" in m for m in msgs)
