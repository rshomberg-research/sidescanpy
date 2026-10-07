"""
Unit tests for sidescanpy.processing.geometric.compute_geometric_ranges.

Validates that per-sample geometric variables (slant_range_m,
ground_range_m, across_track_m, along_track_m, vertical_offset_m,
depth_per_sample_m) are computed correctly from per-ping altitude,
pitch, depth, and sound velocity.

The function adds new data variables but does not modify backscatter
or processing_level.

2026 rshomberg@uri.edu
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.processing.geometric import compute_geometric_ranges

# ============================================================
# HELPERS
# ============================================================

def _make_test_dataset(
    *,
    n_pings: int = 5,
    n_samples: int = 100,
    sample_rate_hz: float = 1000.0,
    altitude_m: float | np.ndarray = 10.0,
    sound_vel_ms: float | np.ndarray = 1500.0,
    pitch_deg: float | np.ndarray = 0.0,
    roll_deg: float | np.ndarray = 0.0,
    depth_m: float | np.ndarray = 50.0,
    backscatter_val: float = 1.0,
) -> xr.Dataset:
    """Build a minimal valid sidescan dataset for testing."""
    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    def _per_ping(value, dtype=np.float32):
        if np.isscalar(value):
            return np.full(n_pings, value, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    alt   = _per_ping(altitude_m)
    sv    = _per_ping(sound_vel_ms)
    pitch = _per_ping(pitch_deg)
    roll  = _per_ping(roll_deg)
    depth = _per_ping(depth_m)

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
        "pitch_deg": (["ping"], pitch),
        "roll_deg":  (["ping"], roll),
        "depth_m": (["ping"], depth),
        "altitude_m": (["ping"], alt),
        "fathometer_reading_m": (["ping"], alt),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], sv),
    }

    attrs = {
        "freq_hz_start": 1495000,
        "freq_hz_end": 1495000,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }

    return xr.Dataset(data_vars, coords, attrs)


# ============================================================
# SLANT RANGE MATH (preserved from earlier version)
# ============================================================

@pytest.mark.unit
class TestSlantRangeMath:
    def test_slant_range_formula(self):
        ds = _make_test_dataset(
            n_samples=100, sample_rate_hz=1000.0, sound_vel_ms=1500.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values
        for s_idx in (0, 50, 99):
            expected = s_idx * 1500.0 / (2.0 * 1000.0)
            actual = float(slant[0, 0, s_idx])
            assert abs(actual - expected) < 1e-5

    def test_slant_range_sample_zero_is_zero(self):
        ds = _make_test_dataset(n_samples=10)
        result = compute_geometric_ranges(ds)
        assert float(result["slant_range_m"].values[0, 0, 0]) == 0.0

    def test_slant_range_monotonic(self):
        ds = _make_test_dataset(n_samples=50)
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values[0, 0, :]
        assert np.all(np.diff(slant) > 0)


# ============================================================
# GROUND RANGE MATH (unchanged: flat bottom, no pitch correction)
# ============================================================

@pytest.mark.unit
class TestGroundRangeMath:
    def test_ground_range_pythagorean(self):
        ds = _make_test_dataset(
            n_samples=200, sample_rate_hz=1000.0, sound_vel_ms=1500.0,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values[0, 0, :]
        ground = result["ground_range_m"].values[0, 0, :]
        beyond_nadir = slant > 10.0
        expected = np.sqrt(slant[beyond_nadir] ** 2 - 10.0 ** 2)
        np.testing.assert_allclose(ground[beyond_nadir], expected, atol=1e-5)

    def test_zero_altitude_ground_equals_slant(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=0.0)
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values
        ground = result["ground_range_m"].values
        np.testing.assert_allclose(ground[0, 0, 1:], slant[0, 0, 1:], atol=1e-5)

    def test_ground_range_monotonic_beyond_nadir(self):
        ds = _make_test_dataset(n_samples=100, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values[0, 0, :]
        valid = ground[~np.isnan(ground)]
        assert np.all(np.diff(valid) > 0)

    def test_ground_range_unaffected_by_pitch(self):
        """ground_range_m uses the simple pythagorean — pitch correction
        lives in across_track_m/along_track_m instead."""
        ds_no_pitch = _make_test_dataset(n_samples=100, altitude_m=10.0, pitch_deg=0.0)
        ds_pitch = _make_test_dataset(n_samples=100, altitude_m=10.0, pitch_deg=10.0)

        no_pitch = compute_geometric_ranges(ds_no_pitch)["ground_range_m"].values
        with_pitch = compute_geometric_ranges(ds_pitch)["ground_range_m"].values

        np.testing.assert_array_equal(no_pitch, with_pitch)


# ============================================================
# NADIR GAP HANDLING
# ============================================================

@pytest.mark.unit
class TestNadirGap:
    def test_water_column_samples_are_nan(self):
        ds = _make_test_dataset(
            n_samples=50, sample_rate_hz=1000.0, sound_vel_ms=1500.0,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values[0, 0, :]
        ground = result["ground_range_m"].values[0, 0, :]
        in_water_column = slant < 10.0
        assert np.all(np.isnan(ground[in_water_column]))
        beyond_nadir = slant > 10.0
        assert np.all(np.isfinite(ground[beyond_nadir]))


# ============================================================
# ACROSS-TRACK MATH (new)
# ============================================================

@pytest.mark.unit
class TestAcrossTrackMath:
    def test_across_track_zero_pitch(self):
        """At zero pitch, |across_track| == ground_range."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=0.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        across = result["across_track_m"].values
        valid = ~np.isnan(ground)
        np.testing.assert_allclose(np.abs(across[valid]), ground[valid], atol=1e-5)

    def test_across_track_pitch_correction(self):
        """across_track_m = ±ground_range_m × cos(pitch_rad)."""
        pitch_deg_val = 15.0
        ds = _make_test_dataset(
            n_samples=50, altitude_m=10.0, pitch_deg=pitch_deg_val,
        )
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        across = result["across_track_m"].values
        valid = ~np.isnan(ground)
        cos_factor = np.cos(np.deg2rad(pitch_deg_val))
        expected_magnitude = ground[valid] * cos_factor
        np.testing.assert_allclose(np.abs(across[valid]), expected_magnitude, atol=1e-5)

    def test_across_track_port_negative(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        port_across = result["across_track_m"].values[:, 0, :]  # channel 0 = port
        valid = ~np.isnan(port_across)
        assert np.all(port_across[valid] <= 0)

    def test_across_track_starboard_positive(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        stbd_across = result["across_track_m"].values[:, 1, :]  # channel 1 = stbd
        valid = ~np.isnan(stbd_across)
        assert np.all(stbd_across[valid] >= 0)

    def test_across_track_nan_propagation(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        across = result["across_track_m"].values
        # Wherever ground is NaN, across should be NaN
        assert np.array_equal(np.isnan(ground), np.isnan(across))


# ============================================================
# ALONG-TRACK MATH (new)
# ============================================================

@pytest.mark.unit
class TestAlongTrackMath:
    def test_along_track_zero_pitch(self):
        """At zero pitch, along_track == 0 for all valid samples."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=0.0)
        result = compute_geometric_ranges(ds)
        along = result["along_track_m"].values
        valid = ~np.isnan(along)
        np.testing.assert_allclose(along[valid], 0.0, atol=1e-5)

    def test_along_track_pitch_positive_forward(self):
        """Bow up (positive pitch) → samples reflect forward of nadir
        → positive along_track_m."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=10.0)
        result = compute_geometric_ranges(ds)
        along = result["along_track_m"].values
        valid = ~np.isnan(along)
        assert np.all(along[valid] >= 0)
        assert np.any(along[valid] > 0)

    def test_along_track_pitch_negative_aft(self):
        """Bow down (negative pitch) → samples reflect aft of nadir
        → negative along_track_m."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=-10.0)
        result = compute_geometric_ranges(ds)
        along = result["along_track_m"].values
        valid = ~np.isnan(along)
        assert np.all(along[valid] <= 0)
        assert np.any(along[valid] < 0)

    def test_along_track_pitch_correction_formula(self):
        """along_track_m = +ground_range_m × sin(pitch_rad)."""
        pitch_deg_val = 5.0
        ds = _make_test_dataset(
            n_samples=50, altitude_m=10.0, pitch_deg=pitch_deg_val,
        )
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        along = result["along_track_m"].values
        valid = ~np.isnan(ground)
        sin_factor = np.sin(np.deg2rad(pitch_deg_val))
        expected = ground[valid] * sin_factor
        np.testing.assert_allclose(along[valid], expected, atol=1e-5)

    def test_along_track_nan_propagation(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        along = result["along_track_m"].values
        assert np.array_equal(np.isnan(ground), np.isnan(along))


# ============================================================
# VERTICAL OFFSET (new)
# ============================================================

@pytest.mark.unit
class TestVerticalOffset:
    def test_vertical_offset_equals_altitude_for_flat_bottom(self):
        """For a flat bottom, vertical_offset_m == altitude_m broadcast
        to (ping, channel, sample) shape, masked where ground is NaN."""
        ds = _make_test_dataset(n_samples=100, altitude_m=15.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        vertical = result["vertical_offset_m"].values
        valid = ~np.isnan(ground)
        np.testing.assert_allclose(vertical[valid], 15.0, atol=1e-5)

    def test_vertical_offset_nan_in_water_column(self):
        """By default, vertical_offset_m is NaN where ground_range_m is NaN."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        vertical = result["vertical_offset_m"].values
        assert np.array_equal(np.isnan(ground), np.isnan(vertical))

    def test_vertical_offset_per_ping_altitude_variation(self):
        altitudes = np.array([5.0, 10.0, 15.0], dtype=np.float32)
        ds = _make_test_dataset(n_pings=3, n_samples=200, altitude_m=altitudes)
        result = compute_geometric_ranges(ds)
        for p in range(3):
            ground = result["ground_range_m"].values[p, 0, :]
            vertical = result["vertical_offset_m"].values[p, 0, :]
            valid = ~np.isnan(ground)
            np.testing.assert_allclose(vertical[valid], altitudes[p], atol=1e-5)

    def test_vertical_offset_unmasked_option(self):
        """expose_unmasked=True → vertical_offset_m broadcast everywhere,
        no NaN masking."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0)
        result = compute_geometric_ranges(ds, expose_unmasked_z=True)
        vertical = result["vertical_offset_m"].values
        # Should be 10.0 everywhere, no NaN
        assert not np.any(np.isnan(vertical))
        np.testing.assert_allclose(vertical, 10.0, atol=1e-5)


# ============================================================
# DEPTH PER SAMPLE (new)
# ============================================================

@pytest.mark.unit
class TestDepthPerSample:
    def test_depth_per_sample_formula(self):
        """depth_per_sample_m = depth_m + altitude_m, broadcast."""
        ds = _make_test_dataset(
            n_samples=100, altitude_m=10.0, depth_m=50.0,
        )
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        depth_per_sample = result["depth_per_sample_m"].values
        valid = ~np.isnan(ground)
        np.testing.assert_allclose(depth_per_sample[valid], 60.0, atol=1e-5)

    def test_depth_per_sample_nan_in_water_column(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, depth_m=50.0)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        depth_per_sample = result["depth_per_sample_m"].values
        assert np.array_equal(np.isnan(ground), np.isnan(depth_per_sample))

    def test_depth_per_sample_with_varying_depths(self):
        depths = np.array([40.0, 50.0, 60.0], dtype=np.float32)
        altitudes = np.array([10.0, 12.0, 8.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3, n_samples=200,
            altitude_m=altitudes, depth_m=depths,
        )
        result = compute_geometric_ranges(ds)
        for p in range(3):
            ground = result["ground_range_m"].values[p, 0, :]
            depth_per_sample = result["depth_per_sample_m"].values[p, 0, :]
            valid = ~np.isnan(ground)
            expected = depths[p] + altitudes[p]
            np.testing.assert_allclose(depth_per_sample[valid], expected, atol=1e-5)

    def test_depth_per_sample_unmasked_option(self):
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, depth_m=50.0)
        result = compute_geometric_ranges(ds, expose_unmasked_z=True)
        depth_per_sample = result["depth_per_sample_m"].values
        assert not np.any(np.isnan(depth_per_sample))
        np.testing.assert_allclose(depth_per_sample, 60.0, atol=1e-5)


# ============================================================
# PORT/STARBOARD SYMMETRY
# ============================================================

@pytest.mark.unit
class TestPitchSymmetry:
    def test_port_starboard_symmetric_at_zero_pitch(self):
        """At zero pitch, port and starboard across_track values are
        magnitude-symmetric (port negative, stbd positive, equal magnitude)."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=0.0)
        result = compute_geometric_ranges(ds)
        across = result["across_track_m"].values
        port_mag = np.abs(across[:, 0, :])
        stbd_mag = np.abs(across[:, 1, :])
        np.testing.assert_array_equal(port_mag, stbd_mag)

    def test_port_starboard_along_track_identical(self):
        """Both channels share the same pitch-induced along_track offset."""
        ds = _make_test_dataset(n_samples=50, altitude_m=10.0, pitch_deg=10.0)
        result = compute_geometric_ranges(ds)
        along = result["along_track_m"].values
        np.testing.assert_array_equal(along[:, 0, :], along[:, 1, :])


# ============================================================
# PER-PING ATTITUDE VARIATION
# ============================================================

@pytest.mark.unit
class TestPerPingAttitude:
    def test_per_ping_pitch(self):
        pitches = np.array([0.0, 5.0, 10.0, -5.0, -10.0], dtype=np.float32)
        ds = _make_test_dataset(n_pings=5, n_samples=100, altitude_m=10.0,
                                pitch_deg=pitches)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        across = result["across_track_m"].values
        along = result["along_track_m"].values

        for p in range(5):
            valid = ~np.isnan(ground[p, 0, :])
            cos_p = np.cos(np.deg2rad(pitches[p]))
            sin_p = np.sin(np.deg2rad(pitches[p]))

            # Port across_track: -ground × cos(pitch)
            np.testing.assert_allclose(
                across[p, 0, valid],
                -ground[p, 0, valid] * cos_p,
                atol=1e-4,
            )
            # Along track: +ground × sin(pitch)
            np.testing.assert_allclose(
                along[p, 0, valid],
                ground[p, 0, valid] * sin_p,
                atol=1e-4,
            )


# ============================================================
# OUTPUT VARIABLE SHAPES & METADATA
# ============================================================

@pytest.mark.unit
class TestOutputVariables:
    @pytest.mark.parametrize("var_name", [
        "slant_range_m",
        "ground_range_m",
        "across_track_m",
        "along_track_m",
        "vertical_offset_m",
        "depth_per_sample_m",
    ])
    def test_variable_present_with_correct_shape(self, var_name):
        ds = _make_test_dataset(n_pings=4, n_samples=30)
        result = compute_geometric_ranges(ds)
        assert var_name in result.data_vars
        assert result[var_name].shape == (4, 2, 30)
        assert result[var_name].dims == ("ping", "channel", "sample")

    @pytest.mark.parametrize("var_name", [
        "slant_range_m", "ground_range_m", "across_track_m",
        "along_track_m", "vertical_offset_m", "depth_per_sample_m",
    ])
    def test_dtype_is_float(self, var_name):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        assert np.issubdtype(result[var_name].dtype, np.floating)

    def test_backscatter_unchanged(self):
        ds = _make_test_dataset(backscatter_val=5.0, n_samples=20)
        original_intensity = ds["backscatter_intensity"].values.copy()
        original_raw = ds["backscatter_raw"].values.copy()
        result = compute_geometric_ranges(ds)
        np.testing.assert_array_equal(
            result["backscatter_intensity"].values, original_intensity
        )
        np.testing.assert_array_equal(
            result["backscatter_raw"].values, original_raw
        )

    def test_dataset_still_validates(self):
        """Output must still pass schema validation."""
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        result.sidescan.validate()


# ============================================================
# PROCESSING METADATA
# ============================================================

@pytest.mark.unit
class TestProcessingMetadata:
    def test_processing_level_derived(self):
        ds = _make_test_dataset()
        assert ds.sidescan.processing_level == "raw"
        result = compute_geometric_ranges(ds)
        assert result.sidescan.processing_level == "geometric"
        assert "processing_level" not in result.attrs

    def test_history_entry_added(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        history = json.loads(result.attrs["processing_history"])
        assert len(history) == 1
        assert history[0]["step"] == "geometric"

    def test_history_records_function_path(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        history = json.loads(result.attrs["processing_history"])
        assert "geometric" in history[0]["function"]

    def test_history_records_expose_unmasked_param(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds, expose_unmasked_z=True)
        history = json.loads(result.attrs["processing_history"])
        assert history[0]["params"]["expose_unmasked_z"] is True

    def test_works_on_radiometric_input(self):
        ds = _make_test_dataset()
        ds.attrs["processing_level"] = "radiometric"  # legacy attribute
        result = compute_geometric_ranges(ds)
        assert result.sidescan.processing_level == "geometric"
        assert "slant_range_m" in result.data_vars


# ============================================================
# SOUND VELOCITY HANDLING (preserved)
# ============================================================

@pytest.mark.unit
class TestSoundVelocity:
    def test_per_ping_sound_velocity(self):
        sv = np.array([1450.0, 1500.0, 1550.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3, n_samples=100, sample_rate_hz=1000.0,
            sound_vel_ms=sv, altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values
        for p in range(3):
            expected = 50 * sv[p] / 2000.0
            actual = float(slant[p, 0, 50])
            assert abs(actual - expected) < 1e-4

    def test_nan_sound_velocity_falls_back(self):
        sv = np.array([1500.0, np.nan, 1500.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3, n_samples=20, sound_vel_ms=sv, altitude_m=5.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values
        assert np.all(np.isfinite(slant[0, :, :]))
        assert np.all(np.isfinite(slant[2, :, :]))
        ping1_slant = slant[1, 0, 10]
        expected_with_default = 10 * 1500.0 / (2.0 * 1000.0)
        assert (
            abs(ping1_slant - expected_with_default) < 1e-4
            or np.isnan(ping1_slant)
        ), f"Ping with NaN sound_vel should use default or be NaN, got {ping1_slant}"

    def test_zero_sound_velocity_falls_back(self):
        sv = np.array([1500.0, 0.0, 1500.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3, n_samples=20, sound_vel_ms=sv, altitude_m=5.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values
        # Ping 1 has zero sv → fallback to 1500 → slant non-zero for non-zero samples
        assert float(slant[1, 0, 10]) > 0


# ============================================================
# EDGE CASES
# ============================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_single_ping(self):
        ds = _make_test_dataset(n_pings=1, n_samples=20)
        result = compute_geometric_ranges(ds)
        for var in (
            "slant_range_m", "ground_range_m", "across_track_m",
            "along_track_m", "vertical_offset_m", "depth_per_sample_m",
        ):
            assert var in result.data_vars

    def test_nan_altitude_yields_nan_outputs(self):
        """NaN altitude on a ping → all per-sample outputs for that ping
        are NaN (across_track_m, along_track_m, vertical_offset_m,
        depth_per_sample_m, ground_range_m). slant_range_m is unaffected."""
        alt = np.array([10.0, np.nan, 10.0], dtype=np.float32)
        ds = _make_test_dataset(n_pings=3, n_samples=50, altitude_m=alt)
        result = compute_geometric_ranges(ds)

        # slant_range_m doesn't depend on altitude — unaffected
        assert np.all(np.isfinite(result["slant_range_m"].values[1, :, :]))

        # All other outputs NaN for ping 1
        for var in (
            "ground_range_m", "across_track_m", "along_track_m",
            "vertical_offset_m", "depth_per_sample_m",
        ):
            assert np.all(np.isnan(result[var].values[1, :, :]))

    def test_input_dataset_unchanged(self):
        ds = _make_test_dataset()
        original_vars = set(ds.data_vars)
        original_history = ds.attrs["processing_history"]
        _ = compute_geometric_ranges(ds)
        assert set(ds.data_vars) == original_vars
        assert ds.attrs["processing_history"] == original_history
