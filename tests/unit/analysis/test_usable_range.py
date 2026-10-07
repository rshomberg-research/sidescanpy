"""
Unit tests for sidescanpy.analysis.usable_range.

Small synthetic fixtures with known, hand-constructed signal shapes.
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.analysis.usable_range import estimate_usable_range
from sidescanpy.processing.bottom import uniform_exclusion_boundary
from sidescanpy.processing.geometric import compute_geometric_ranges
from tests.unit.synthetic import make_raw_dataset as _make_raw_dataset


class TestEstimateUsableRange:
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

        result = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)

        assert result.usable_range_m is not None
        # Signal decays to near-nothing well before the far range edge.
        assert result.usable_range_m < ds["ground_range_m"].max().item()

    def test_missing_range_var_raises(self):
        ds = _make_raw_dataset()
        with pytest.raises(ValueError, match="slant_range_m"):
            estimate_usable_range(ds)

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

        pooled = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)

        assert pooled.channel is None

    def test_channel_restricts_usable_range_to_one_side(self):
        ds = self._make_two_channel_range_ds()

        ch0 = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=0)
        ch1 = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=1)

        assert ch0.channel == 0
        assert ch1.channel == 1
        # channel 0's signal reaches much further than channel 1's --
        # proof each was fit against only its own side, not pooled.
        assert ch0.usable_range_m > ch1.usable_range_m

    def test_channel_string_matches_int_position(self):
        ds = self._make_two_channel_range_ds()

        by_string = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel="port")
        by_int = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, channel=0)

        assert by_string.usable_range_m == pytest.approx(by_int.usable_range_m)
        assert by_string.channel == 0

    def test_invalid_channel_string_raises(self):
        ds = self._make_two_channel_range_ds()

        with pytest.raises(ValueError, match="not found in channel_id"):
            estimate_usable_range(ds, channel="midship")

    def test_water_column_n_excluded_matches_geometry(self):
        """Same physics as TestAnalyzeNadir.test_geometric_boundary_matches_geometry:
        altitude=10.0, sound_vel=1500.0, sample_rate=1000.0 -> water
        column ends at sample 14 -> 14 excluded samples per (ping, channel)."""
        n_pings, n_samples = 50, 200
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        result = estimate_usable_range(ds, bin_width_m=1.0)

        assert result.water_column_n_excluded == 14 * n_pings * 2

    def test_exclude_before_sample_masks_nadir_gap(self):
        """exclude_before_sample should additionally exclude samples
        the altitude-crossing mask alone doesn't catch."""
        n_pings, n_samples = 20, 200
        ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
        ds = compute_geometric_ranges(ds)

        boundary = np.full((n_pings, 2), 50.0)  # well past the altitude sample (~14)
        result = estimate_usable_range(ds, bin_width_m=1.0, exclude_before_sample=boundary)

        assert result.nadir_gap_n_excluded == (50 - 14) * n_pings * 2

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

        with_spikes = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0)
        cleaned = estimate_usable_range(ds, bin_width_m=1.0, snr_threshold=3.0, reject_outliers=True)

        assert cleaned.outlier_n_rejected > 0
        assert cleaned.noise_std < with_spikes.noise_std

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
        s3_per_ping = estimate_usable_range(ds, "_v", range_var="_r", bin_width_m=1.0, exclude_before_sample=boundary)
        s3_uniform = estimate_usable_range(ds, "_v", range_var="_r", bin_width_m=1.0, exclude_before_sample=uniform)

        per_ping_binned = s3_per_ping.binned
        uniform_binned = s3_uniform.binned

        # Exact per-ping masking: no valid signal exists below ~8m for
        # any ping, so no near-zero contamination bin appears there.
        assert not ((per_ping_binned["range_min_m"] < 8.0) & (per_ping_binned["median"] < 1.0)).any()

        # Uniform boundary: the deep group's still-invalid samples leak
        # through in [100, 250) at their own (compressed) range grid,
        # landing below 8m where no valid data exists -- a thin,
        # near-zero contaminated bin the per-ping boundary doesn't have.
        contaminated = uniform_binned[(uniform_binned["range_min_m"] < 8.0) & (uniform_binned["median"] < 1.0)]
        assert not contaminated.empty
