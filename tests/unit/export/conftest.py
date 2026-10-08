"""
Shared fixtures for export unit tests.

Builds a minimal, schema-valid channel dataset (mirroring
``tests/unit/processing/test_geometric.py``'s own helper) with varying
per-ping lat/lon so along-track distance and georeferencing are
non-degenerate, then runs it through ``compute_geometric_ranges``/
``compute_georeferencing`` so tests exercise the real geometry/georef
pipeline rather than hand-built fake coordinates.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.georeferencing import compute_georeferencing


def make_test_channel(
    *,
    n_pings: int = 20,
    n_samples: int = 200,
    sample_rate_hz: float = 1000.0,
    altitude_m: float = 5.0,
    georef: bool = False,
) -> xr.Dataset:
    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    rng = np.random.default_rng(0)
    backscatter = rng.random((n_pings, 2, n_samples)).astype(np.float32) * 50.0

    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], backscatter),
        "backscatter_raw": (["ping", "channel", "sample"], backscatter),
        "latitude": (["ping"], np.linspace(41.50, 41.501, n_pings, dtype=np.float64)),
        "longitude": (["ping"], np.linspace(-71.50, -71.499, n_pings, dtype=np.float64)),
        "heading_deg": (["ping"], np.full(n_pings, 45.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 20.0, dtype=np.float32)),
        "altitude_m": (["ping"], np.full(n_pings, altitude_m, dtype=np.float32)),
        "fathometer_reading_m": (["ping"], np.full(n_pings, altitude_m, dtype=np.float32)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
    }

    attrs = {
        "freq_hz_start": 1495000,
        "freq_hz_end": 1495000,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds = compute_geometric_ranges(ds)
    if georef:
        ds = compute_georeferencing(ds)
    return ds


@pytest.fixture
def geometric_channel() -> xr.Dataset:
    return make_test_channel(georef=False)


@pytest.fixture
def georef_channel() -> xr.Dataset:
    return make_test_channel(georef=True)


@pytest.fixture
def geometric_channel_factory():
    """Factory for tests needing non-default parameters (e.g. an
    altitude that puts every sample in the water column)."""
    return lambda **kwargs: make_test_channel(georef=False, **kwargs)


@pytest.fixture
def georef_channel_factory():
    return lambda **kwargs: make_test_channel(georef=True, **kwargs)


@pytest.fixture
def georef_ss_h5_dir(tmp_path):
    """A corrected two-frequency survey (see ``surveys.geo_friendly_survey``) on disk."""
    from sidescanpy.io.hdf5 import write_survey
    from sidescanpy.processing import correct_survey
    from tests.unit.export.surveys import geo_friendly_survey

    d = tmp_path / "ss_h5"
    d.mkdir()
    write_survey(correct_survey(geo_friendly_survey()), d / "survey_a.ss.h5")
    return d


@pytest.fixture
def swath_ss_h5_dir(tmp_path):
    """A corrected 300 x 30 survey (see ``surveys.swath_friendly_survey``) on disk."""
    from sidescanpy.io.hdf5 import write_survey
    from sidescanpy.processing import correct_survey
    from tests.unit.export.surveys import swath_friendly_survey

    d = tmp_path / "ss_h5"
    d.mkdir()
    write_survey(correct_survey(swath_friendly_survey()), d / "survey_a.ss.h5")
    return d
