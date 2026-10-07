"""
Unit tests for sidescanpy.processing.slicing.

Tests slicing operations that take and return an xr.Dataset.

2026 rshomberg@uri.edu
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.processing.slicing import (
    slice_by_bbox,
    slice_by_ground_range,
    slice_by_pings,
    slice_by_range,
    slice_by_samples,
    slice_by_time,
    slice_survey,
)

# ============================================================
# HELPERS
# ============================================================

def _make_channel(
    *,
    n_pings: int = 10,
    n_samples: int = 50,
    start_time: str = "2026-01-01T00:00:00",
    step_sec: float = 1.0,
    latitude: float | np.ndarray = 41.5,
    longitude: float | np.ndarray = -71.5,
    altitude_m: float | np.ndarray = 5.0,
    sound_vel_ms: float = 1500.0,
    sample_rate_hz: float = 1000.0,
    pitch_deg: float = 0.0,
    add_geometric: bool = False,
    add_georef: bool = False,
) -> xr.Dataset:
    """Build a minimal valid channel with optional geometric/georef vars."""
    times = pd.date_range(start_time, periods=n_pings, freq=f"{step_sec}s").values \
        .astype("datetime64[ns]")

    def _per_ping(value, dtype=np.float32):
        if np.isscalar(value):
            return np.full(n_pings, value, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    lat = _per_ping(latitude, dtype=np.float64)
    lon = _per_ping(longitude, dtype=np.float64)
    alt = _per_ping(altitude_m)

    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], times),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    data_vars = {
        "backscatter_intensity": (
            ["ping", "channel", "sample"],
            np.full((n_pings, 2, n_samples), 1.0, dtype=np.float32),
        ),
        "backscatter_raw": (
            ["ping", "channel", "sample"],
            np.full((n_pings, 2, n_samples), 1.0, dtype=np.float32),
        ),
        "latitude": (["ping"], lat),
        "longitude": (["ping"], lon),
        "heading_deg": (["ping"], np.full(n_pings, 0.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.full(n_pings, pitch_deg, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
        "altitude_m": (["ping"], alt),
        "fathometer_reading_m": (["ping"], alt),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, sound_vel_ms, dtype=np.float32)),
    }

    attrs = {
        "freq_hz_start": 1495000,
        "freq_hz_end": 1495000,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }

    ds = xr.Dataset(data_vars, coords, attrs)

    if add_geometric:
        from sidescanpy.processing.geometric import compute_geometric_ranges
        ds = compute_geometric_ranges(ds)

    if add_georef:
        from sidescanpy.processing.geometric import compute_geometric_ranges
        from sidescanpy.processing.georeferencing import compute_georeferencing
        if "ground_range_m" not in ds.data_vars:
            ds = compute_geometric_ranges(ds)
        ds = compute_georeferencing(ds)

    return ds


# ============================================================
# SLICE BY PINGS
# ============================================================

@pytest.mark.unit
class TestSliceByPings:
    def test_slice_keeps_specified_range(self):
        ch = _make_channel(n_pings=20, n_samples=10)
        result = slice_by_pings(ch, start=5, stop=15)
        assert result.sizes["ping"] == 10
        assert result.sizes["sample"] == 10
        # Check that we got pings 5-14
        assert int(result["ping_number"].values[0]) == 5
        assert int(result["ping_number"].values[-1]) == 14

    def test_default_start_zero(self):
        ch = _make_channel(n_pings=20)
        result = slice_by_pings(ch, stop=5)
        assert result.sizes["ping"] == 5

    def test_default_stop_end(self):
        ch = _make_channel(n_pings=20)
        result = slice_by_pings(ch, start=15)
        assert result.sizes["ping"] == 5

    def test_full_slice_returns_equivalent(self):
        ch = _make_channel(n_pings=10)
        result = slice_by_pings(ch)
        assert result.sizes["ping"] == 10
        np.testing.assert_array_equal(
            result["latitude"].values,
            ch["latitude"].values,
        )

    def test_empty_slice_validates(self):
        ch = _make_channel(n_pings=10)
        result = slice_by_pings(ch, start=10, stop=10)
        assert result.sizes["ping"] == 0
        # Should still validate
        result.sidescan.validate()


    def test_attrs_preserved(self):
        ch = _make_channel()
        result = slice_by_pings(ch, start=2, stop=5)
        assert result.attrs["sample_rate_hz"] == ch.attrs["sample_rate_hz"]
        assert result.attrs["processing_level"] == ch.attrs["processing_level"]

    def test_stop_beyond_size_clamps(self):
        """Stop beyond n_pings should yield all remaining pings."""
        ch = _make_channel(n_pings=10)
        result = slice_by_pings(ch, start=8, stop=100)
        assert result.sizes["ping"] == 2

    def test_start_beyond_size_returns_empty(self):
        ch = _make_channel(n_pings=10)
        result = slice_by_pings(ch, start=100)
        assert result.sizes["ping"] == 0


# ============================================================
# SLICE BY SAMPLES
# ============================================================

@pytest.mark.unit
class TestSliceBySamples:
    def test_slice_keeps_specified_range(self):
        ch = _make_channel(n_pings=5, n_samples=100)
        result = slice_by_samples(ch, start=10, stop=50)
        assert result.sizes["sample"] == 40
        assert result.sizes["ping"] == 5

    def test_default_start_zero(self):
        ch = _make_channel(n_samples=100)
        result = slice_by_samples(ch, stop=50)
        assert result.sizes["sample"] == 50

    def test_default_stop_end(self):
        ch = _make_channel(n_samples=100)
        result = slice_by_samples(ch, start=50)
        assert result.sizes["sample"] == 50

    def test_full_slice_returns_equivalent(self):
        ch = _make_channel(n_samples=20)
        result = slice_by_samples(ch)
        assert result.sizes["sample"] == 20
        np.testing.assert_array_equal(
            result["backscatter_intensity"].values,
            ch["backscatter_intensity"].values,
        )


# ============================================================
# SLICE BY TIME
# ============================================================

@pytest.mark.unit
class TestSliceByTime:
    def test_slice_within_range(self):
        # 10 pings at 1s intervals starting at 2026-01-01T00:00:00
        ch = _make_channel(n_pings=10, step_sec=1.0,
                          start_time="2026-01-01T00:00:00")
        result = slice_by_time(
            ch,
            start="2026-01-01T00:00:03",
            stop="2026-01-01T00:00:07",
        )
        # Pings at 3, 4, 5, 6 seconds — 4 pings (stop is exclusive)
        assert result.sizes["ping"] == 4

    def test_no_start_includes_from_beginning(self):
        ch = _make_channel(n_pings=10, step_sec=1.0)
        result = slice_by_time(ch, stop="2026-01-01T00:00:05")
        # Pings at 0, 1, 2, 3, 4 seconds — 5 pings
        assert result.sizes["ping"] == 5

    def test_no_stop_includes_to_end(self):
        ch = _make_channel(n_pings=10, step_sec=1.0)
        result = slice_by_time(ch, start="2026-01-01T00:00:05")
        # Pings at 5, 6, 7, 8, 9 — 5 pings
        assert result.sizes["ping"] == 5

    def test_outside_range_returns_empty(self):
        ch = _make_channel(n_pings=10, step_sec=1.0,
                          start_time="2026-01-01T00:00:00")
        result = slice_by_time(
            ch,
            start="2030-01-01T00:00:00",
            stop="2030-01-01T00:01:00",
        )
        assert result.sizes["ping"] == 0

    def test_string_datetime_input(self):
        ch = _make_channel(n_pings=5, step_sec=1.0)
        result = slice_by_time(ch, start="2026-01-01T00:00:02")
        assert result.sizes["ping"] == 3  # pings 2, 3, 4

    def test_numpy_datetime64_input(self):
        ch = _make_channel(n_pings=5, step_sec=1.0)
        start = np.datetime64("2026-01-01T00:00:02", "ns")
        result = slice_by_time(ch, start=start)
        assert result.sizes["ping"] == 3



# ============================================================
# SLICE BY BBOX
# ============================================================

@pytest.mark.unit
class TestSliceByBbox:
    def test_per_ping_default(self):
        # Pings at varying lat
        lats = np.array([41.0, 41.5, 42.0, 42.5, 43.0], dtype=np.float64)
        ch = _make_channel(n_pings=5, latitude=lats)
        result = slice_by_bbox(
            ch,
            lat_min=41.5, lat_max=42.5,
            lon_min=-72.0, lon_max=-71.0,
        )
        assert result.sizes["ping"] == 3  # 41.5, 42.0, 42.5

    def test_bbox_excluding_all_pings_returns_empty(self):
        lats = np.array([41.0, 41.5, 42.0], dtype=np.float64)
        ch = _make_channel(n_pings=3, latitude=lats)
        result = slice_by_bbox(
            ch,
            lat_min=50.0, lat_max=51.0,  # well outside survey range
            lon_min=-72.0, lon_max=-71.0,
        )
        assert result.sizes["ping"] == 0
        # Should still validate
        result.sidescan.validate()

    def test_bbox_includes_all_pings(self):
        ch = _make_channel(n_pings=5, latitude=41.5, longitude=-71.5)
        result = slice_by_bbox(
            ch,
            lat_min=40.0, lat_max=43.0,
            lon_min=-73.0, lon_max=-70.0,
        )
        assert result.sizes["ping"] == 5

    def test_bbox_excludes_by_longitude(self):
        lons = np.array([-72.0, -71.5, -71.0, -70.5, -70.0], dtype=np.float64)
        ch = _make_channel(n_pings=5, longitude=lons)
        result = slice_by_bbox(
            ch,
            lat_min=40.0, lat_max=43.0,
            lon_min=-71.5, lon_max=-70.5,
        )
        assert result.sizes["ping"] == 3  # -71.5, -71.0, -70.5

    def test_use_per_sample_requires_georef(self):
        """use_per_sample=True without latitude_per_sample raises."""
        ch = _make_channel(n_pings=3)
        with pytest.raises(ValueError, match="latitude_per_sample"):
            slice_by_bbox(
                ch,
                lat_min=40.0, lat_max=43.0,
                lon_min=-72.0, lon_max=-71.0,
                use_per_sample=True,
            )


# ============================================================
# SLICE BY GROUND RANGE
# ============================================================

@pytest.mark.unit
class TestSliceByGroundRange:
    def test_truncates_to_max_in_range(self):
        """With altitude=5m and sample_rate=1000, sound_vel=1500:
        sample_interval = 0.75m. Sample 7 has slant=5.25m, ground~1.5m.
        Sample 100 has slant=75m, ground~74.8m. So range_max=10m
        should keep samples 0..~13 (where ground crosses 10m)."""
        ch = _make_channel(
            n_pings=3, n_samples=100,
            altitude_m=5.0,
            sound_vel_ms=1500.0,
            sample_rate_hz=1000.0,
            add_geometric=True,
        )
        result = slice_by_ground_range(ch, range_max_m=10.0)
        # Should truncate samples — fewer than 100
        assert result.sizes["sample"] < ch.sizes["sample"]
        assert result.sizes["ping"] == ch.sizes["ping"]

    def test_requires_ground_range_m(self):
        """Without ground_range_m, raises ValueError."""
        ch = _make_channel(n_pings=3)
        with pytest.raises(ValueError, match="ground_range_m"):
            slice_by_ground_range(ch, range_max_m=10.0)

    def test_zero_max_returns_minimal(self):
        """range_max=0 means no samples are within range → empty sample dim."""
        ch = _make_channel(
            n_pings=3, n_samples=100,
            altitude_m=5.0,
            add_geometric=True,
        )
        result = slice_by_ground_range(ch, range_max_m=0.0)
        # No samples should satisfy ground_range <= 0 (water column has NaN)
        assert result.sizes["sample"] == 0

    def test_large_max_returns_full(self):
        """Very large range_max keeps all samples."""
        ch = _make_channel(
            n_pings=3, n_samples=50,
            altitude_m=5.0,
            add_geometric=True,
        )
        result = slice_by_ground_range(ch, range_max_m=1000.0)
        assert result.sizes["sample"] == 50


# ============================================================
# CROSS-CUTTING: SCHEMA, METADATA, FREQ_KEY
# ============================================================

@pytest.mark.unit
class TestRoundTripSchema:
    @pytest.mark.parametrize("slicer_name,kwargs", [
        ("slice_by_pings", {"start": 2, "stop": 5}),
        ("slice_by_samples", {"start": 5, "stop": 25}),
        ("slice_by_time", {"start": "2026-01-01T00:00:02"}),
        ("slice_by_bbox", {
            "lat_min": 40.0, "lat_max": 43.0,
            "lon_min": -73.0, "lon_max": -70.0,
        }),
    ])
    def test_sliced_channel_validates(self, slicer_name, kwargs):
        from sidescanpy.processing import slicing
        slicer = getattr(slicing, slicer_name)
        ch = _make_channel(n_pings=10, n_samples=50)
        result = slicer(ch, **kwargs)
        result.sidescan.validate()

    def test_processing_history_unchanged_by_pings(self):
        """Slicing is structural — does not change processing_history."""
        ch = _make_channel()
        result = slice_by_pings(ch, start=2, stop=5)
        assert (
            result.attrs["processing_history"]
            == ch.attrs["processing_history"]
        )

    def test_processing_level_unchanged(self):
        ch = _make_channel()
        ch.attrs["processing_level"] = "radiometric"
        result = slice_by_pings(ch, start=2, stop=5)
        assert result.attrs["processing_level"] == "radiometric"

    def test_chained_slicing(self):
        """Result of one slice can be sliced again."""
        ch = _make_channel(n_pings=20, n_samples=100)
        result = slice_by_pings(ch, start=5, stop=15)
        result = slice_by_samples(result, start=20, stop=80)
        assert result.sizes["ping"] == 10
        assert result.sizes["sample"] == 60
        result.sidescan.validate()


# ============================================================
# ABSOLUTE SAMPLE COORDINATE / RANGE / SURVEY MAPPING
# ============================================================

@pytest.mark.unit
class TestAbsoluteSampleSlicing:
    def test_slice_by_samples_is_label_based(self):
        ds = _make_channel(n_pings=3, n_samples=50)
        once = slice_by_samples(ds, start=10, stop=40)
        twice = slice_by_samples(once, start=20, stop=30)
        assert list(twice["sample"].values) == list(range(20, 30))

    def test_slice_by_samples_without_coord_is_positional(self):
        ds = _make_channel(n_pings=3, n_samples=50).drop_vars("sample")
        out = slice_by_samples(ds, start=5, stop=15)
        assert out.sizes["sample"] == 10

    def test_slice_by_range_uses_slant_range(self):
        # 1500 m/s, 1000 Hz -> 0.75 m per sample
        ds = _make_channel(n_pings=3, n_samples=50)
        out = slice_by_range(ds, min_m=7.5, max_m=15.0)
        assert list(out["sample"].values) == list(range(10, 21))

    def test_slice_by_range_open_ended(self):
        ds = _make_channel(n_pings=3, n_samples=50)
        assert slice_by_range(ds).sizes["sample"] == 50
        assert slice_by_range(ds, max_m=0.75).sizes["sample"] == 2

    def test_slice_survey_maps_over_frequencies(self):
        from sidescanpy.core.survey import SideScanSurvey

        survey = SideScanSurvey(source_files=[], source_format="test", survey_name="s")
        survey.add_frequency("hz_1495000", _make_channel(n_pings=10))
        out = slice_survey(survey, slice_by_pings, start=2, stop=5)
        assert out.get_frequency("hz_1495000").n_pings == 3
