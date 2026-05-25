"""
Shared test fixtures for sidescanpy tests

2026 rshomberg@uri.edu
"""

import pytest
import json
import numpy as np
import xarray as xr
import pickle

@pytest.fixture
def sample_dataset():
    """Valid frequency channel dataset matching .ss.h5 schema with dask arrays."""
    n_pings = 1000
    n_channels = 2
    n_samples = 2000

    backscatter_raw = xr.DataArray(
        np.random.rand(n_pings, n_channels, n_samples).astype(np.float32),
        dims=["ping", "channel", "sample"],
    ).chunk({"ping": 256})

    ds = xr.Dataset(
        {
            "backscatter_raw": backscatter_raw,
            "backscatter_intensity": backscatter_raw.copy(),
            "latitude": (["ping"], np.linspace(40.0, 40.5, n_pings)),
            "longitude": (["ping"], np.linspace(-70.0, -69.9, n_pings)),
            "heading_deg": (["ping"], np.random.rand(n_pings) * 360),
            "depth_m": (["ping"], np.random.rand(n_pings) * 200 + 100),
            "roll_deg": (["ping"], np.random.randn(n_pings) * 5),
            "pitch_deg": (["ping"], np.random.randn(n_pings) * 5),
            "heave_m": (["ping"], np.random.randn(n_pings) * 2),
            "altitude_m": (["ping"], np.random.rand(n_pings) * 100 + 50),
            "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0)),
        },
        coords={
            "ping_number": (["ping"], np.arange(1, n_pings + 1, dtype=np.int32)),
            "time": (["ping"], np.linspace(0, 1000, n_pings)),
            "channel_id": (["channel"], ["port", "starboard"]),
        },
        attrs={
            "freq_hz_start": 1495000,
            "freq_hz_end": 1495000,
            "pulse_length_us": 0.5,
            "sample_rate_hz": 400000,
            "range_m": 200.0,
            "processing_level": "raw",
            "processing_history": json.dumps([]),
        },
    )
    return ds


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def sample_jsf_messages():
    """Load pickled JSF messages for testing."""
    from tests.data.conftest import generate_sample_jsf_messages
    
    pkl_path = generate_sample_jsf_messages(n_messages=100)
    
    with open(pkl_path, 'rb') as f:
        messages = pickle.load(f)
    
    return messages


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line("markers", "unit: unit test")
    config.addinivalue_line("markers", "integration: integration test")
