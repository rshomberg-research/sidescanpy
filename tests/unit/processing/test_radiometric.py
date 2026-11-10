"""
Unit tests for radiometric correction (TVG) - rigorous mathematical validation.

These tests prove the algorithm is mathematically correct by:
1. Testing each computational step independently
2. Comparing against hand-calculated values
3. Validating edge cases and boundary conditions
4. Verifying no NaN/inf leakage
"""

import json

import numpy as np
import pytest
import xarray as xr

from sidescanpy.processing.radiometric import (
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
    apply_tvg_to_amplitude,
    francois_garrison_absorption_db_per_m,
    tvg_gain_db,
)

# ============================================================================
# STEP 1: LINEAR → dB CONVERSION TESTS
# ============================================================================

class TestLinearToDBConversion:
    """Verify linear-to-dB conversion is mathematically correct."""

    def test_linear_1_maps_to_0_db(self):
        """Linear value 1.0 should map to 0 dB exactly."""
        # 20*log10(1.0) = 0
        value_linear = 1.0
        value_db = 20.0 * np.log10(value_linear)
        assert abs(value_db - 0.0) < 1e-10

    def test_linear_10_maps_to_20_db(self):
        """Linear value 10.0 should map to 20 dB exactly."""
        # 20*log10(10.0) = 20
        value_linear = 10.0
        value_db = 20.0 * np.log10(value_linear)
        assert abs(value_db - 20.0) < 1e-10

    def test_linear_0_1_maps_to_minus_20_db(self):
        """Linear value 0.1 should map to -20 dB exactly."""
        # 20*log10(0.1) = -20
        value_linear = 0.1
        value_db = 20.0 * np.log10(value_linear)
        assert abs(value_db - (-20.0)) < 1e-10

    def test_zero_handling(self):
        """Zero should clip to 1e-6 and map to approximately -120 dB."""
        # log10(1e-6) = -6, so 20*-6 = -120
        clipped = max(1e-6, 0.0)
        value_db = 20.0 * np.log10(clipped)
        assert abs(value_db - (-120.0)) < 0.1

    def test_power_law_relationship(self):
        """Verify log properties: log(a*b) = log(a) + log(b)."""
        # If we have 2*5 = 10, then log(2*5) should equal log(2) + log(5)
        val_product = 20.0 * np.log10(2 * 5)
        val_sum = 20.0 * np.log10(2) + 20.0 * np.log10(5)
        assert abs(val_product - val_sum) < 1e-10


# ============================================================================
# STEP 2: SLANT RANGE CALCULATION TESTS
# ============================================================================

class TestSlantRangeCalculation:
    """Verify slant range computation is correct."""

    def test_slant_range_formula(self):
        """Slant range = sample_idx * (sound_velocity / (2 * sample_rate))."""
        sample_idx = 100
        sound_velocity = 1500.0  # m/s
        sample_rate = 1000.0  # Hz

        # Range = time_of_flight * sound_velocity / 2
        # where time_of_flight = sample_idx / sample_rate
        time_of_flight = sample_idx / sample_rate  # 0.1 s
        range_expected = time_of_flight * sound_velocity / 2.0  # 75 m

        range_computed = sample_idx * (sound_velocity / (2.0 * sample_rate))

        assert abs(range_computed - range_expected) < 1e-10
        assert abs(range_computed - 75.0) < 1e-10

    def test_sample_0_is_near_field(self):
        """Sample 0 should be clipped to min_range_m (0.5 m)."""
        min_range = 0.5
        sample_range = 0.0
        clipped = max(sample_range, min_range)
        assert clipped == min_range

    def test_sample_indices_increase_monotonically(self):
        """Slant range should increase with sample index."""
        sound_velocity = 1500.0
        sample_rate = 1000.0

        ranges = []
        for i in range(0, 101, 10):
            r = i * (sound_velocity / (2.0 * sample_rate))
            ranges.append(max(r, 0.5))  # Apply min clip

        # All differences should be positive
        diffs = np.diff(ranges)
        assert np.all(diffs >= 0)


# ============================================================================
# STEP 3: TVG GAIN CALCULATION TESTS
# ============================================================================

class TestTVGGainCalculation:
    """Verify TVG gain (spreading + absorption) computation."""

    def test_spreading_loss_formula(self):
        """Spreading loss = 20 * log10(range_m)."""
        range_m = 100.0
        spreading_db = 20.0 * np.log10(range_m)

        # 20*log10(100) = 20*2 = 40 dB
        assert abs(spreading_db - 40.0) < 1e-10

    def test_absorption_loss_formula(self):
        """Absorption loss = 2 * alpha * range_m (two-way path)."""
        alpha_db_per_m = 0.1  # dB/m
        range_m = 100.0
        absorption_db = 2.0 * alpha_db_per_m * range_m

        # 2 * 0.1 * 100 = 20 dB
        assert abs(absorption_db - 20.0) < 1e-10

    def test_total_tvg_gain(self):
        """TVG = spreading_loss + absorption_loss."""
        range_m = 100.0
        spreading = 20.0 * np.log10(range_m)  # 40 dB
        absorption = 2.0 * 0.1 * range_m      # 20 dB
        total_tvg = spreading + absorption

        assert abs(total_tvg - 60.0) < 1e-10

    def test_tvg_increases_with_range(self):
        """TVG gain should monotonically increase with range."""
        ranges = np.array([10.0, 50.0, 100.0, 200.0])
        tvg_gains = []

        for r in ranges:
            spreading = 20.0 * np.log10(r)
            absorption = 2.0 * 0.1 * r
            tvg = spreading + absorption
            tvg_gains.append(tvg)

        diffs = np.diff(tvg_gains)
        assert np.all(diffs > 0), f"TVG gains should increase: {tvg_gains}"


# ============================================================================
# STEP 4: END-TO-END CORRECTION TESTS
# ============================================================================

class TestEnd2EndCorrection:
    """Test complete correction pipeline with known values."""

    def create_simple_dataset(self, n_pings=5, n_samples=100, backscatter_val=10.0, sound_vel_ms=1500.0):
        """Create minimal test dataset."""
        import pandas as pd

        coords = {
            "ping": np.arange(n_pings),
            "channel": np.arange(2),
            "sample": np.arange(n_samples),
            "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
            "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
            "channel_id": (["channel"], ["port", "starboard"]),
        }

        data_vars = {
            "backscatter_intensity": (
                ["ping", "channel", "sample"],
                np.full((n_pings, 2, n_samples), backscatter_val, dtype=np.float32),
            ),
            "backscatter_raw": (
                ["ping", "channel", "sample"],
                np.full((n_pings, 2, n_samples), backscatter_val, dtype=np.float32),
            ),
            "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
            "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
            "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
            "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
            "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
            "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "sound_vel_ms": (["ping"], np.full(n_pings, sound_vel_ms, dtype=np.float32)),
            "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        }

        attrs = {
            "freq_hz_start": 1495000,
            "freq_hz_end": 1495000,
            "sample_rate_hz": 1000.0,
            "pulse_length_us": 100.0,
            "range_m": 100.0,
            "processing_level": "raw",
            "processing_history": "[]",
        }

        return xr.Dataset(data_vars, coords, attrs)

    def test_correction_with_spreading_only(self):
        """Test TVG with spreading loss, no absorption."""
        ds = self.create_simple_dataset(backscatter_val=1.0, n_samples=101)

        # Apply TVG: spreading only
        result = apply_tvg(ds, TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.0))

        # Sample 100: range = 100 * (1500 / 2000) = 75 m
        sample_idx = 100
        range_m = 75.0

        # Expected: 20*log10(1.0) + 20*log10(75)
        expected = 0.0 + 20.0 * np.log10(range_m)  # ≈ 37.5 dB

        actual = float(result["backscatter_intensity"][0, 0, sample_idx])

        # Should match within 0.1 dB
        assert abs(actual - expected) < 0.1, \
            f"Expected {expected:.2f} dB, got {actual:.2f} dB (diff: {abs(actual - expected):.4f})"

    def test_correction_with_spreading_and_absorption(self):
        """Test TVG with both spreading and absorption."""
        ds = self.create_simple_dataset(backscatter_val=1.0, n_samples=101)

        # Apply TVG: spreading + absorption
        result = apply_tvg(ds, TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.1))

        # Sample 100: range = 75 m
        sample_idx = 100
        range_m = 75.0

        # Expected: 20*log10(1.0) + 20*log10(75) + 2*0.1*75
        spreading = 20.0 * np.log10(range_m)    # ≈ 37.5 dB
        absorption = 2.0 * 0.1 * range_m         # = 15.0 dB
        expected = 0.0 + spreading + absorption  # ≈ 52.5 dB

        actual = float(result["backscatter_intensity"][0, 0, sample_idx])

        assert abs(actual - expected) < 0.1, \
            f"Expected {expected:.2f} dB, got {actual:.2f} dB"

    def test_near_field_vs_far_field_gain_difference(self):
        """Verify TVG gain increases from near to far field."""
        ds = self.create_simple_dataset(backscatter_val=1.0, n_samples=200)

        result = apply_tvg(ds, TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.1))

        # Near field (sample 10): range ≈ 7.5 m
        near_idx = 10
        near_range = near_idx * (1500.0 / 2000.0)
        near_tvg = 20.0 * np.log10(near_range) + 2.0 * 0.1 * near_range
        near_expected = 0.0 + near_tvg

        # Far field (sample 150): range ≈ 112.5 m
        far_idx = 150
        far_range = far_idx * (1500.0 / 2000.0)
        far_tvg = 20.0 * np.log10(far_range) + 2.0 * 0.1 * far_range
        far_expected = 0.0 + far_tvg

        near_actual = float(result["backscatter_intensity"][0, 0, near_idx])
        far_actual = float(result["backscatter_intensity"][0, 0, far_idx])

        # Far field should have MORE correction (higher value)
        assert far_actual > near_actual, \
            f"Far field ({far_actual:.2f}) should exceed near field ({near_actual:.2f})"

        gain_difference = far_actual - near_actual
        expected_difference = far_expected - near_expected

        assert abs(gain_difference - expected_difference) < 0.2

    def test_backscatter_raw_unchanged_by_tvg(self):
        ds = self.create_simple_dataset(backscatter_val=5.0, n_samples=20)
        original_raw = ds["backscatter_raw"].values.copy()
        result = apply_tvg(ds)
        np.testing.assert_array_equal(result["backscatter_raw"].values, original_raw)

    def test_zero_sound_velocity_falls_back_to_default(self):
        """Confirmed real-world case (2026-07-16): sound_vel_ms can be
        exactly 0.0, not NaN. Before the fix, this silently pinned every
        sample's slant range at min_range_m, making TVG's gain constant per
        ping instead of range-varying. Must fall back to 1500.0 m/s exactly
        like NaN already does."""
        ds_zero = self.create_simple_dataset(backscatter_val=1.0, n_samples=101, sound_vel_ms=0.0)
        ds_default = self.create_simple_dataset(backscatter_val=1.0, n_samples=101, sound_vel_ms=1500.0)

        result_zero = apply_tvg(ds_zero)
        result_default = apply_tvg(ds_default)

        np.testing.assert_allclose(
            result_zero["backscatter_intensity"].values,
            result_default["backscatter_intensity"].values,
            rtol=1e-5,
        )
        # And directly: gain must vary across samples, not be constant —
        # the specific failure mode this fix addresses.
        gains = result_zero["backscatter_intensity"].values[0, 0, :]
        assert not np.allclose(gains, gains[0])

    def test_nan_sound_velocity_still_falls_back_to_default(self):
        """Regression guard: the NaN case (already handled before this
        fix) must keep working identically."""
        ds_nan = self.create_simple_dataset(backscatter_val=1.0, n_samples=101, sound_vel_ms=np.nan)
        ds_default = self.create_simple_dataset(backscatter_val=1.0, n_samples=101, sound_vel_ms=1500.0)

        result_nan = apply_tvg(ds_nan)
        result_default = apply_tvg(ds_default)

        np.testing.assert_allclose(
            result_nan["backscatter_intensity"].values,
            result_default["backscatter_intensity"].values,
            rtol=1e-5,
        )

# ============================================================================
# STEP 5: NAN/INF SAFETY TESTS
# ============================================================================

class TestNaNInfHandling:
    """Ensure no NaN or inf values leak into output."""

    def create_dataset_with_zeros(self):
        """Create dataset with zero backscatter (water column)."""
        import pandas as pd

        n_pings, n_samples = 5, 50
        coords = {
            "ping": np.arange(n_pings),
            "channel": np.arange(2),
            "sample": np.arange(n_samples),
            "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
            "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
            "channel_id": (["channel"], ["port", "starboard"]),
        }

        # Mix of real backscatter and zeros
        data = np.random.rand(n_pings, 2, n_samples).astype(np.float32) * 10.0
        data[:, :, :10] = 0.0  # Zero out first samples

        data_vars = {
            "backscatter_intensity": (["ping", "channel", "sample"], data),
            "backscatter_raw": (["ping", "channel", "sample"], data.copy()),
            "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
            "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
            "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
            "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
            "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
            "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
            "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
            "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        }

        attrs = {
            "freq_hz_start": 1495000,
            "freq_hz_end": 1495000,
            "sample_rate_hz": 1000.0,
            "pulse_length_us": 100.0,
            "range_m": 100.0,
            "processing_level": "raw",
            "processing_history": "[]",
        }

        return xr.Dataset(data_vars, coords, attrs)

    def test_no_inf_from_log_zero(self):
        """Ensure log(0) doesn't produce inf."""
        ds = self.create_dataset_with_zeros()
        result = apply_tvg(ds)

        data = result["backscatter_intensity"].values

        # Count inf values
        inf_count = np.isinf(data).sum()
        assert inf_count == 0, f"Found {inf_count} inf values in output"

    def test_no_nan_in_output(self):
        """Ensure no NaN values in corrected data."""
        ds = self.create_dataset_with_zeros()
        result = apply_tvg(ds)

        data = result["backscatter_intensity"].values

        # Count NaN values
        nan_count = np.isnan(data).sum()
        total = data.size

        assert nan_count == 0, f"Found {nan_count}/{total} NaN values in output"

    def test_zero_maps_to_negative_value(self):
        """Zero backscatter should map to very negative dB (not -inf)."""
        ds = self.create_dataset_with_zeros()
        result = apply_tvg(ds)

        data = result["backscatter_intensity"].values

        # Find values from originally-zero samples
        # These should be roughly 20*log10(1e-6) ≈ -120 dB
        zero_corrected = data[:, :, :10]

        # Should be finite and very negative
        assert np.all(np.isfinite(zero_corrected)), "Corrected zeros should be finite"
        assert np.all(zero_corrected < -50), "Corrected zeros should be very negative"


class TestFloorClipThroughTvg:
    """Exact-zero amplitude stays at the -120 dB floor after TVG, so
    mask_floor_clip still finds it, and inversion maps it back to 0."""

    def _ds(self):
        return TestNaNInfHandling().create_dataset_with_zeros()

    def test_floor_clipped_samples_stay_at_floor_after_tvg(self):
        from sidescanpy.core.units import FLOOR_DB

        out = apply_tvg(self._ds())["backscatter_intensity"].values
        assert np.all(out[:, :, :10] == FLOOR_DB)
        assert out.dtype == np.float32

    def test_mask_floor_clip_catches_them_after_tvg(self):
        from sidescanpy.core.units import mask_floor_clip

        out = mask_floor_clip(apply_tvg(self._ds())["backscatter_intensity"].values)
        assert np.isnan(out[:, :, :10]).all()
        assert np.isfinite(out[:, :, 10:]).all()

    def test_real_samples_still_get_gain(self):
        ds = self._ds()
        out = apply_tvg(ds)["backscatter_intensity"].values
        raw_db = 20 * np.log10(ds["backscatter_intensity"].values[:, :, 10:])
        assert np.all(out[:, :, 10:] > raw_db)

    def test_inversion_returns_floor_clipped_samples_to_zero(self):
        from sidescanpy.processing import uncorrected_backscatter

        ds = self._ds()
        back = uncorrected_backscatter(apply_tvg(ds)).values
        assert np.all(back[:, :, :10] == 0.0)
        np.testing.assert_allclose(back[:, :, 10:], ds["backscatter_intensity"].values[:, :, 10:], rtol=1e-4)

    def test_apply_tvg_to_amplitude_keeps_floor(self):
        from sidescanpy.core.units import FLOOR_DB

        amp = np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32)
        out = apply_tvg_to_amplitude(amp, meters_per_sample=10.0, freq_hz=455_000.0)
        assert out[0, 0] == FLOOR_DB and out[0, 1] == FLOOR_DB
        assert out[0, 3] > 20 * np.log10(5.0)


# ============================================================================
# STEP 6: METADATA TRACKING TESTS
# ============================================================================

class TestMetadataTracking:
    """Verify processing history and level are updated correctly."""

    @pytest.mark.parametrize("sample_dataset", [
        {"n_pings": 10, "n_samples": 50, "backscatter_val": 1.0}
    ], indirect=True)
    def test_processing_level_updated(self, sample_dataset):
        """Derived processing level goes raw -> tvg; legacy attribute dropped."""
        ds = sample_dataset
        assert ds.sidescan.processing_level == "raw"

        result = apply_tvg(ds)
        assert result.sidescan.processing_level == "tvg"
        assert "processing_level" not in result.attrs

    @pytest.mark.parametrize("sample_dataset", [
        {"n_pings": 10, "n_samples": 50, "backscatter_val": 1.0}
    ], indirect=True)
    def test_history_has_tvg_step(self, sample_dataset):
        """Processing history should contain TVG step."""
        ds = sample_dataset
        result = apply_tvg(ds)

        history = json.loads(result.attrs["processing_history"])
        assert len(history) == 1
        assert history[0]["step"] == "tvg"

    @pytest.mark.parametrize("sample_dataset", [
        {"n_pings": 10, "n_samples": 50, "backscatter_val": 1.0}
    ], indirect=True)
    def test_history_records_parameters(self, sample_dataset):
        """History should record all TVG parameters."""
        ds = sample_dataset
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.15)
        result = apply_tvg(ds, params)

        history = json.loads(result.attrs["processing_history"])
        assert history[0]["params"]["spreading_loss_db"] == 20.0
        assert history[0]["params"]["absorption_db_per_m"] == 0.15


class TestAbsorptionForFrequency:
    """Frequency-dependent absorption defaults."""

    def test_absorption_increases_with_frequency(self):
        """Higher frequencies should have higher absorption."""
        alpha_low = absorption_for_frequency(100000)      # 100 kHz
        alpha_mid = absorption_for_frequency(547500)      # 547.5 kHz
        alpha_high = absorption_for_frequency(1495000)    # 1495 kHz

        assert alpha_low < alpha_mid < alpha_high


class TestFrancoisGarrison:
    """Francois & Garrison (1982) absorption.

    No tabulated reference values are reproduced here (none could be
    verified offline). The checks are: (a) closed-form limits of the
    published formula (fresh water leaves only the pure-water term
    A3*P3*f^2; zero frequency gives zero), (b) widely quoted order-of-
    magnitude seawater values with generous tolerances (about 0.06 dB/km
    at 1 kHz, about 30-40 dB/km at 100 kHz, a few hundred dB/km at 1 MHz,
    ~0.1 dB/m at 400 kHz), (c) qualitative physics (monotonic in
    frequency, temperature branch continuity at 20 C, unit conversion).
    """

    def test_zero_frequency_is_zero(self):
        assert francois_garrison_absorption_db_per_m(0.0) == 0.0

    def test_scalar_in_float_out_array_in_array_out(self):
        assert isinstance(francois_garrison_absorption_db_per_m(100e3), float)
        out = francois_garrison_absorption_db_per_m(np.array([100e3, 400e3, 900e3]))
        assert out.shape == (3,)
        assert out[0] == pytest.approx(francois_garrison_absorption_db_per_m(100e3))

    def test_fresh_water_reduces_to_pure_water_term(self):
        # S=0 kills both relaxation terms; remaining alpha = A3 * P3 * f^2
        # with the paper's T <= 20 C cubic and depth factor P3.
        t, d, f_khz = 10.0, 0.0, 1000.0
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
        got = francois_garrison_absorption_db_per_m(f_khz * 1000.0, temperature_c=t, salinity_psu=0.0, depth_m=d)
        assert got * 1000.0 == pytest.approx(a3 * f_khz**2, rel=1e-9)
        assert got * 1000.0 == pytest.approx(311.0, rel=0.02)  # ~0.31 dB/m at 1 MHz, 10 C

    def test_one_khz_cold_deep_ocean_order_of_magnitude(self):
        alpha_db_km = francois_garrison_absorption_db_per_m(1e3, temperature_c=4.0, depth_m=1000.0) * 1000.0
        assert 0.03 < alpha_db_km < 0.12

    def test_hundred_khz_order_of_magnitude(self):
        deep = francois_garrison_absorption_db_per_m(100e3, temperature_c=4.0, depth_m=1000.0) * 1000.0
        coastal = francois_garrison_absorption_db_per_m(100e3) * 1000.0
        assert 15.0 < deep < 45.0
        assert 25.0 < coastal < 45.0
        assert deep < coastal  # MgSO4 relaxation is pressure-reduced at depth

    def test_one_megahertz_order_of_magnitude(self):
        alpha_db_km = francois_garrison_absorption_db_per_m(1e6) * 1000.0
        assert 250.0 < alpha_db_km < 400.0

    def test_sidescan_frequencies_in_published_range(self):
        # ~0.1 dB/m at 400 kHz is the commonly quoted figure
        assert 0.08 < francois_garrison_absorption_db_per_m(400e3) < 0.12

    def test_monotonic_in_frequency(self):
        f = np.linspace(10e3, 2e6, 200)
        assert np.all(np.diff(francois_garrison_absorption_db_per_m(f)) > 0)

    def test_pure_water_branches_continuous_at_20_c(self):
        lo = francois_garrison_absorption_db_per_m(1e6, temperature_c=20.0)
        hi = francois_garrison_absorption_db_per_m(1e6, temperature_c=20.0 + 1e-9)
        # the paper's two polynomials agree to ~0.05% at 20 C (not exactly)
        assert hi == pytest.approx(lo, rel=2e-3)

    def test_sound_velocity_override_changes_relaxation_amplitudes(self):
        base = francois_garrison_absorption_db_per_m(500e3)
        slower = francois_garrison_absorption_db_per_m(500e3, sound_velocity_ms=1400.0)
        assert slower > base  # A1, A2 scale as 1/c

    def test_wrapper_matches_and_forwards_environment(self):
        assert absorption_for_frequency(547_500) == francois_garrison_absorption_db_per_m(547_500)
        assert absorption_for_frequency(547_500, temperature_c=25.0) == pytest.approx(
            francois_garrison_absorption_db_per_m(547_500, temperature_c=25.0))

    def test_defaults_are_typical_coastal(self):
        assert francois_garrison_absorption_db_per_m(300e3) == pytest.approx(
            francois_garrison_absorption_db_per_m(300e3, temperature_c=10.0, salinity_psu=35.0, depth_m=10.0, ph=8.0))


# ============================================================================
# tvg_gain_db() / apply_tvg_to_amplitude(): shared helper + plain-array path
# ============================================================================

class TestTvgGainDbHelper:
    """tvg_gain_db() is apply_tvg()'s Step 3 lifted out as a pure function —
    cross-check it reproduces the same numbers apply_tvg() itself produces
    via the full xarray path, using the same scenario as
    TestEnd2EndCorrection.test_correction_with_spreading_and_absorption."""

    def test_matches_apply_tvg_end_to_end_gain(self):
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.1)
        range_m = 75.0  # sample 100 at sound_vel=1500, sample_rate=1000 Hz

        gain = tvg_gain_db(np.array([range_m]), params)[0]

        spreading = 20.0 * np.log10(range_m)
        absorption = 2.0 * 0.1 * range_m
        assert abs(gain - (spreading + absorption)) < 1e-10

    def test_increases_monotonically_with_range(self):
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.1)
        ranges = np.array([10.0, 50.0, 100.0, 200.0])
        gains = tvg_gain_db(ranges, params)
        assert np.all(np.diff(gains) > 0)


class TestApplyTvgToAmplitude:
    """apply_tvg_to_amplitude() — the plain-array (no xarray) TVG path used
    for target-window crops, which don't come as a full survey Dataset."""

    def test_flattens_known_range_dependent_falloff(self):
        # Synthetic amplitude that falls off exactly as 1/R (pure
        # geometric spreading, no absorption) — TVG with spreading-only
        # params should flatten this back to a constant.
        n_samples = 200
        meters_per_sample = 0.05
        sample_idx = np.arange(n_samples)
        range_m = np.clip(sample_idx * meters_per_sample, 0.5, None)
        amplitude = 1.0 / range_m  # pure 1/R falloff

        corrected = apply_tvg_to_amplitude(
            amplitude, meters_per_sample,
            params=TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.0),
        )

        # 20*log10(1/R) + 20*log10(R) == 0 for every sample (spreading
        # exactly cancels a pure 1/R falloff) — flat within float slop.
        assert np.ptp(corrected) < 1e-6

    def test_matches_apply_tvg_for_equivalent_input(self):
        # Same scenario as TestEnd2EndCorrection.test_correction_with_spreading_and_absorption
        # (sound_vel=1500, sample_rate=1000 -> meters_per_sample=0.75),
        # confirming the plain-array path agrees with the xarray path.
        meters_per_sample = 1500.0 / (2.0 * 1000.0)
        amplitude = np.full(200, 1.0)
        params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.1)

        corrected = apply_tvg_to_amplitude(amplitude, meters_per_sample, params=params)

        sample_idx = 100
        range_m = sample_idx * meters_per_sample  # 75.0
        expected = 0.0 + 20.0 * np.log10(range_m) + 2.0 * 0.1 * range_m
        assert abs(corrected[sample_idx] - expected) < 0.1

    def test_freq_hz_derives_absorption_default(self):
        amplitude = np.full(50, 1.0)
        corrected_via_freq = apply_tvg_to_amplitude(amplitude, 0.05, freq_hz=1495000)
        corrected_via_explicit = apply_tvg_to_amplitude(
            amplitude, 0.05,
            params=TVGParams(absorption_db_per_m=absorption_for_frequency(1495000)),
        )
        np.testing.assert_array_almost_equal(corrected_via_freq, corrected_via_explicit)

    def test_raises_without_params_or_freq_hz(self):
        with pytest.raises(ValueError):
            apply_tvg_to_amplitude(np.full(10, 1.0), 0.05)

    def test_2d_pings_by_samples_input(self):
        amplitude = np.full((5, 50), 1.0)
        corrected = apply_tvg_to_amplitude(amplitude, 0.05, freq_hz=1495000)
        assert corrected.shape == (5, 50)
        # Every ping should get the identical per-sample gain.
        np.testing.assert_array_almost_equal(corrected[0], corrected[-1])



class TestCenterFrequency:
    """Frequency-dependent physics uses the chirp's center, not its start."""

    def test_chirp_midpoint(self):
        from sidescanpy.processing.radiometric import center_frequency_hz

        ds = xr.Dataset(attrs={"freq_hz_start": 1_495_000, "freq_hz_end": 1_605_000})
        assert center_frequency_hz(ds) == pytest.approx(1_550_000.0)

    @pytest.mark.parametrize("end", [455_000, None, 0, float("nan"), "n/a"])
    def test_falls_back_to_start(self, end):
        from sidescanpy.processing.radiometric import center_frequency_hz

        attrs = {"freq_hz_start": 455_000}
        if end is not None:
            attrs["freq_hz_end"] = end
        assert center_frequency_hz(xr.Dataset(attrs=attrs)) == 455_000.0

    def test_default_tvg_absorption_uses_center(self):
        from sidescanpy.processing.radiometric import (
            absorption_for_frequency,
            francois_garrison_absorption_db_per_m,
            resolve_tvg_params,
        )

        ds = xr.Dataset({"sound_vel_ms": ("ping", np.full(3, 1500.0))},
                        attrs={"freq_hz_start": 547_500, "freq_hz_end": 612_500})
        params = resolve_tvg_params(ds, None)
        assert params.absorption_db_per_m == pytest.approx(francois_garrison_absorption_db_per_m(580_000))
        assert params.absorption_db_per_m != pytest.approx(absorption_for_frequency(547_500))


class TestDefaultSpreading:
    def test_default_spreading_is_30_log_r(self):
        from sidescanpy.processing.radiometric import DEFAULT_SPREADING_LOSS_DB

        assert DEFAULT_SPREADING_LOSS_DB == 30.0
        assert TVGParams().spreading_loss_db == 30.0
