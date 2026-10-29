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
# PITCHED FAN BEAM (flat bottom)
# ============================================================

@pytest.mark.unit
class TestPitchedFanBeam:
    """A fan beam perpendicular to the body x-axis, pitched by θ at
    altitude h, meets a flat seafloor along a line h·tan θ ahead of nadir."""

    ALT = 10.0

    def _run(self, pitch_deg, n_samples=200):
        ds = _make_test_dataset(
            n_samples=n_samples, altitude_m=self.ALT, pitch_deg=pitch_deg,
        )
        return compute_geometric_ranges(ds)

    @pytest.mark.parametrize("pitch", [-8.0, -2.5, 3.0, 5.0, 12.0])
    def test_along_track_constant_within_ping_when_pitched(self, pitch):
        result = self._run(pitch)
        along = result["along_track_m"].values.astype(np.float64)
        expected = self.ALT * np.tan(np.deg2rad(pitch))
        valid = np.isfinite(along)
        assert valid.sum() > 100
        np.testing.assert_allclose(along[valid], expected, rtol=1e-6, atol=1e-6)
        # Same value at every range of the ping
        for p in range(along.shape[0]):
            row = along[p][np.isfinite(along[p])]
            assert np.ptp(row) < 1e-6

    @pytest.mark.parametrize("pitch", [-8.0, 5.0, 12.0])
    def test_horizontal_distance_is_ground_range_when_pitched(self, pitch):
        result = self._run(pitch)
        slant = result["slant_range_m"].values.astype(np.float64)
        across = result["across_track_m"].values.astype(np.float64)
        along = result["along_track_m"].values.astype(np.float64)
        ground = result["ground_range_m"].values.astype(np.float64)
        valid = np.isfinite(across)
        np.testing.assert_allclose(
            across[valid] ** 2 + along[valid] ** 2,
            slant[valid] ** 2 - self.ALT ** 2,
            rtol=1e-5, atol=1e-3,
        )
        np.testing.assert_allclose(
            np.hypot(across[valid], along[valid]), ground[valid],
            rtol=1e-5, atol=1e-3,
        )

    @pytest.mark.parametrize("pitch", [-8.0, 5.0, 12.0])
    def test_valid_mask_uses_fan_limit_when_pitched(self, pitch):
        result = self._run(pitch)
        slant = result["slant_range_m"].values.astype(np.float64)
        across = result["across_track_m"].values
        along = result["along_track_m"].values
        limit = self.ALT / np.cos(np.deg2rad(pitch))
        expected_valid = slant > limit
        np.testing.assert_array_equal(np.isfinite(across), expected_valid)
        np.testing.assert_array_equal(np.isfinite(along), expected_valid)
        # Samples between h and h/cos θ have a ground range but no position
        between = (slant > self.ALT) & (slant <= limit)
        assert np.all(np.isfinite(result["ground_range_m"].values[between]))
        assert np.all(np.isnan(across[between]))

    @pytest.mark.parametrize("pitch", [3.0, 7.5])
    def test_positive_and_negative_pitch_symmetric(self, pitch):
        up = self._run(pitch)
        down = self._run(-pitch)
        np.testing.assert_array_equal(
            up["across_track_m"].values, down["across_track_m"].values,
        )
        np.testing.assert_allclose(
            up["along_track_m"].values, -down["along_track_m"].values,
        )

    def test_zero_pitch_matches_pre_fix_output_exactly(self):
        """At zero pitch the old model (across = ±ground, along = 0) holds."""
        result = self._run(0.0)
        ground = result["ground_range_m"].values
        sign = np.array([-1.0, 1.0])[None, :, None]
        old_across = (sign * ground.astype(np.float64)).astype(np.float32)
        old_along = np.where(np.isnan(ground), np.nan, 0.0).astype(np.float32)
        np.testing.assert_array_equal(result["across_track_m"].values, old_across)
        np.testing.assert_array_equal(result["along_track_m"].values, old_along)

    def test_point_offset_same_at_every_range(self):
        """Old model grew the along offset with range (8.7 m at 100 m and 5°)."""
        ds = _make_test_dataset(
            n_samples=200, sample_rate_hz=750.0, altitude_m=self.ALT, pitch_deg=5.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values[0, 1]
        along = result["along_track_m"].values[0, 1]
        near = int(np.argmax(np.isfinite(along)))
        far = int(np.argmin(np.abs(slant - 100.0)))
        assert along[near] == pytest.approx(along[far], abs=1e-6)
        assert along[far] == pytest.approx(self.ALT * np.tan(np.deg2rad(5.0)), rel=1e-6)

    def test_nan_pitch_treated_as_zero_with_warning(self, caplog):
        pitches = np.array([np.nan, 5.0, 0.0, np.nan, -5.0], dtype=np.float32)
        ds = _make_test_dataset(n_samples=50, altitude_m=self.ALT, pitch_deg=pitches)
        zero = compute_geometric_ranges(
            _make_test_dataset(n_samples=50, altitude_m=self.ALT, pitch_deg=0.0)
        )
        with caplog.at_level("WARNING", logger="sidescanpy.processing.geometric"):
            result = compute_geometric_ranges(ds)
        assert "2 of 5 pings have NaN pitch_deg" in caplog.text
        for p in (0, 3):
            np.testing.assert_array_equal(
                result["across_track_m"].values[p], zero["across_track_m"].values[p],
            )
            np.testing.assert_array_equal(
                result["along_track_m"].values[p], zero["along_track_m"].values[p],
            )
        rec = json.loads(result.attrs["processing_history"])[-1]
        assert rec["params"]["pitch_model"] == "fan_beam_flat_bottom"
        assert rec["params"]["n_nan_pitch_as_zero"] == 2


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
        across = result["across_track_m"].values
        along = result["along_track_m"].values

        slant = result["slant_range_m"].values.astype(np.float64)
        for p in range(5):
            cos_p = np.cos(np.deg2rad(pitches[p]))
            valid_fan = slant[p, 0, :] > 10.0 / cos_p

            # Pitched fan beam: across = ±sqrt(R² − (h/cos θ)²), along = h·tan θ
            np.testing.assert_allclose(
                across[p, 0, valid_fan],
                -np.sqrt(slant[p, 0, valid_fan] ** 2 - (10.0 / cos_p) ** 2),
                atol=1e-4,
            )
            np.testing.assert_allclose(
                along[p, 0, valid_fan], 10.0 * np.tan(np.deg2rad(pitches[p])),
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
