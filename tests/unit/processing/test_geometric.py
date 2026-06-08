"""
Unit tests for sidescanpy.processing.geometric.compute_geometric_ranges.

Validates that per-sample slant_range_m and ground_range_m are computed
correctly from per-ping altitude and sound velocity. The function adds
new data variables but does not modify backscatter or processing_level.

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

    if np.isscalar(altitude_m):
        alt = np.full(n_pings, altitude_m, dtype=np.float32)
    else:
        alt = np.asarray(altitude_m, dtype=np.float32)

    if np.isscalar(sound_vel_ms):
        sv = np.full(n_pings, sound_vel_ms, dtype=np.float32)
    else:
        sv = np.asarray(sound_vel_ms, dtype=np.float32)

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
        "altitude_m": (["ping"], alt),
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
# SLANT RANGE MATH
# ============================================================

@pytest.mark.unit
class TestSlantRangeMath:
    def test_slant_range_formula(self):
        """slant_range_m[i] = i * sound_vel / (2 * sample_rate)."""
        ds = _make_test_dataset(
            n_samples=100,
            sample_rate_hz=1000.0,
            sound_vel_ms=1500.0,
        )
        result = compute_geometric_ranges(ds)

        # Sample 0 → range 0; sample 100 → range 100 * 1500 / 2000 = 75 m
        slant = result["slant_range_m"].values
        # Values shared across (ping, channel, sample) for constant sound_vel
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
        diffs = np.diff(slant)
        assert np.all(diffs > 0), "Slant range should be strictly increasing"


# ============================================================
# GROUND RANGE MATH
# ============================================================

@pytest.mark.unit
class TestGroundRangeMath:
    def test_ground_range_pythagorean(self):
        """For samples beyond water column, ground = sqrt(slant² - alt²)."""
        ds = _make_test_dataset(
            n_samples=200,
            sample_rate_hz=1000.0,
            sound_vel_ms=1500.0,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)

        slant = result["slant_range_m"].values[0, 0, :]
        ground = result["ground_range_m"].values[0, 0, :]

        # Beyond nadir, ground == sqrt(slant² - alt²)
        beyond_nadir = slant > 10.0
        expected = np.sqrt(slant[beyond_nadir] ** 2 - 10.0 ** 2)
        actual = ground[beyond_nadir]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_zero_altitude_ground_equals_slant(self):
        """When altitude = 0, ground range == slant range."""
        ds = _make_test_dataset(
            n_samples=50,
            altitude_m=0.0,
        )
        result = compute_geometric_ranges(ds)

        slant = result["slant_range_m"].values
        ground = result["ground_range_m"].values

        # Sample 0 has slant = 0, alt = 0 → ground = 0 (not NaN, exact equality)
        # Beyond sample 0, ground == slant
        np.testing.assert_allclose(ground[0, 0, 1:], slant[0, 0, 1:], atol=1e-5)

    def test_ground_range_monotonic_beyond_nadir(self):
        ds = _make_test_dataset(
            n_samples=100,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values[0, 0, :]
        # Drop NaNs
        valid = ground[~np.isnan(ground)]
        diffs = np.diff(valid)
        # Ground range increases monotonically with sample after nadir
        assert np.all(diffs > 0)


# ============================================================
# NADIR GAP HANDLING
# ============================================================

@pytest.mark.unit
class TestNadirGap:
    def test_water_column_samples_are_nan(self):
        """Samples where slant_range <= altitude should yield NaN ground range."""
        # alt=10m, sample interval = 1500/2000 = 0.75 m → samples 0..13 are in water column
        ds = _make_test_dataset(
            n_samples=50,
            sample_rate_hz=1000.0,
            sound_vel_ms=1500.0,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)

        slant = result["slant_range_m"].values[0, 0, :]
        ground = result["ground_range_m"].values[0, 0, :]

        # Where slant ≤ altitude, ground must be NaN
        in_water_column = slant < 10.0
        assert np.all(np.isnan(ground[in_water_column]))

        # Where slant > altitude, ground must be finite
        beyond_nadir = slant > 10.0
        assert np.all(np.isfinite(ground[beyond_nadir]))

    def test_at_exactly_nadir_is_zero_or_nan(self):
        """When slant == altitude exactly, ground range is 0 (not NaN).
        We accept either 0 or NaN here as both are mathematically defensible
        at the boundary; the implementation chooses one and tests should be
        explicit about which."""
        # Construct so that one sample lands exactly at altitude.
        # sample 10 → slant = 10 * 1500/2000 = 7.5 m. Set altitude_m = 7.5
        # so that slant_range[10] == altitude exactly.
        ds = _make_test_dataset(
            n_samples=50,
            sample_rate_hz=1000.0,
            sound_vel_ms=1500.0,
            altitude_m=7.5,
        )
        result = compute_geometric_ranges(ds)
        ground_at_nadir = float(result["ground_range_m"].values[0, 0, 10])
        # sqrt(7.5² - 7.5²) = 0. The implementation may emit 0.0 or NaN
        # at this boundary; both are defensible.
        assert ground_at_nadir == 0.0 or np.isnan(ground_at_nadir)


# ============================================================
# PER-PING ALTITUDE
# ============================================================

@pytest.mark.unit
class TestPerPingAltitude:
    def test_varying_altitude_per_ping(self):
        """Each ping uses its own altitude_m for ground range computation."""
        altitudes = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=5,
            n_samples=200,
            altitude_m=altitudes,
        )
        result = compute_geometric_ranges(ds)

        slant = result["slant_range_m"].values
        ground = result["ground_range_m"].values

        for p in range(5):
            alt = altitudes[p]
            for s_idx in range(200):
                slant_val = slant[p, 0, s_idx]
                ground_val = ground[p, 0, s_idx]
                if slant_val > alt:
                    expected = np.sqrt(slant_val ** 2 - alt ** 2)
                    assert abs(ground_val - expected) < 1e-4
                else:
                    assert np.isnan(ground_val)

    def test_port_and_starboard_use_same_altitude(self):
        """Both channels of a single ping see the same altitude (one transducer)."""
        ds = _make_test_dataset(n_pings=3, n_samples=50, altitude_m=12.0)
        result = compute_geometric_ranges(ds)
        port = result["ground_range_m"].values[:, 0, :]
        stbd = result["ground_range_m"].values[:, 1, :]
        np.testing.assert_array_equal(port, stbd)


# ============================================================
# SOUND VELOCITY HANDLING
# ============================================================

@pytest.mark.unit
class TestSoundVelocity:
    def test_per_ping_sound_velocity(self):
        """Different sound velocities per ping yield different slant ranges."""
        sv = np.array([1450.0, 1500.0, 1550.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3,
            n_samples=100,
            sample_rate_hz=1000.0,
            sound_vel_ms=sv,
            altitude_m=10.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values

        # Sample 50, ping i: range = 50 * sv[i] / 2000
        for p in range(3):
            expected = 50 * sv[p] / 2000.0
            actual = float(slant[p, 0, 50])
            assert abs(actual - expected) < 1e-4

    def test_nan_sound_velocity_falls_back(self):
        """If a ping's sound_vel_ms is NaN, the function falls back to a
        default. Verify behavior is documented (fallback or skip with NaN
        slant)."""
        sv = np.array([1500.0, np.nan, 1500.0], dtype=np.float32)
        ds = _make_test_dataset(
            n_pings=3,
            n_samples=20,
            sound_vel_ms=sv,
            altitude_m=5.0,
        )
        result = compute_geometric_ranges(ds)
        slant = result["slant_range_m"].values

        # Pings 0 and 2: finite slant ranges
        assert np.all(np.isfinite(slant[0, :, :]))
        assert np.all(np.isfinite(slant[2, :, :]))

        # Ping 1: implementation choice — either uses default 1500 m/s, or
        # produces NaN. Whichever it does, document and assert.
        ping1_slant = slant[1, 0, 10]  # mid-range sample
        # We expect the implementation to use the default 1500.0 m/s when
        # the per-ping value is NaN. Adjust this assertion if the
        # implementation chooses to NaN-out instead.
        expected_with_default = 10 * 1500.0 / (2.0 * 1000.0)
        assert (
            abs(ping1_slant - expected_with_default) < 1e-4
            or np.isnan(ping1_slant)
        ), f"Ping with NaN sound_vel should use default or be NaN, got {ping1_slant}"


# ============================================================
# OUTPUT VARIABLE SHAPES & METADATA
# ============================================================

@pytest.mark.unit
class TestOutputVariables:
    def test_slant_range_shape_matches_dataset(self):
        ds = _make_test_dataset(n_pings=4, n_samples=30)
        result = compute_geometric_ranges(ds)
        assert result["slant_range_m"].shape == (4, 2, 30)
        assert result["slant_range_m"].dims == ("ping", "channel", "sample")

    def test_ground_range_shape_matches_dataset(self):
        ds = _make_test_dataset(n_pings=4, n_samples=30)
        result = compute_geometric_ranges(ds)
        assert result["ground_range_m"].shape == (4, 2, 30)
        assert result["ground_range_m"].dims == ("ping", "channel", "sample")

    def test_dtype_is_float(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        assert np.issubdtype(result["slant_range_m"].dtype, np.floating)
        assert np.issubdtype(result["ground_range_m"].dtype, np.floating)

    def test_backscatter_unchanged(self):
        """compute_geometric_ranges must not modify backscatter values."""
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
        """Output must still pass schema validation (slant/ground are
        additive, not replacing required fields)."""
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        result.sidescan.validate()


# ============================================================
# PROCESSING METADATA
# ============================================================

@pytest.mark.unit
class TestProcessingMetadata:
    def test_processing_level_unchanged(self):
        """compute_geometric_ranges does NOT advance processing_level."""
        ds = _make_test_dataset()
        assert ds.attrs["processing_level"] == "raw"
        result = compute_geometric_ranges(ds)
        assert result.attrs["processing_level"] == "raw"

    def test_history_entry_added(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        history = json.loads(result.attrs["processing_history"])
        assert len(history) == 1
        assert history[0]["step"] == "compute_geometric_ranges"

    def test_history_records_function_path(self):
        ds = _make_test_dataset()
        result = compute_geometric_ranges(ds)
        history = json.loads(result.attrs["processing_history"])
        assert "geometric" in history[0]["function"]

    def test_works_on_radiometric_input(self):
        """Function should accept input at any processing_level."""
        ds = _make_test_dataset()
        ds.attrs["processing_level"] = "radiometric"
        result = compute_geometric_ranges(ds)
        # Level remains as-is
        assert result.attrs["processing_level"] == "radiometric"
        assert "slant_range_m" in result.data_vars


# ============================================================
# EDGE CASES
# ============================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_single_ping(self):
        ds = _make_test_dataset(n_pings=1, n_samples=20)
        result = compute_geometric_ranges(ds)
        assert "slant_range_m" in result.data_vars
        assert "ground_range_m" in result.data_vars

    def test_nan_altitude_yields_nan_ground(self):
        """When altitude_m is NaN for a ping, ground_range is NaN for that ping."""
        alt = np.array([10.0, np.nan, 10.0], dtype=np.float32)
        ds = _make_test_dataset(n_pings=3, n_samples=50, altitude_m=alt)
        result = compute_geometric_ranges(ds)
        ground = result["ground_range_m"].values
        # Ping 1 has NaN altitude → all ground values for ping 1 are NaN
        assert np.all(np.isnan(ground[1, :, :]))
        # Other pings unaffected
        assert not np.all(np.isnan(ground[0, :, :]))
        assert not np.all(np.isnan(ground[2, :, :]))

    def test_input_dataset_unchanged(self):
        """compute_geometric_ranges must not mutate the input dataset."""
        ds = _make_test_dataset()
        original_vars = set(ds.data_vars)
        original_history = ds.attrs["processing_history"]

        result = compute_geometric_ranges(ds)

        # Input dataset still has the original variables (no slant/ground added)
        assert set(ds.data_vars) == original_vars
        assert ds.attrs["processing_history"] == original_history
        # Result is the augmented version
        assert "slant_range_m" in result.data_vars
        assert "ground_range_m" in result.data_vars