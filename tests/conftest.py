"""
Shared test fixtures and configuration for sidescanpy tests.

This module provides reusable fixtures for creating test data,
managing temporary directories, and configuring the test environment.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
import json

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for file I/O tests."""
    return tmp_path

@pytest.fixture
def sample_dataset():
    """Valid frequency channel dataset matching .ss.h5 schema with dask arrays."""
    n_pings = 1000
    n_channels = 2
    n_samples = 2000

    # Backscatter data with dask chunks
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



#  TODO: remove unnecessary fixtures
@pytest.fixture
def sample_survey_object(sample_dataset):
    """Create a SideScanSurvey object for testing."""
    from sidescanpy.core.survey import SideScanFrequencyChannel
    return SideScanFrequencyChannel(sample_dataset, freq_key='freq_547500_hz')


@pytest.fixture
def sample_multi_frequency_survey(sample_dataset):
    """Pre-configured MultiFrequencySideScanSurvey with multiple frequencies.
    
    Creates a survey with 2 frequencies, each using sample_dataset.
    """
    from sidescanpy.core.survey import MultiFrequencySideScanSurvey
    
    survey = MultiFrequencySideScanSurvey(
        source_file='test_source.jsf',
        source_format='JSF',
        survey_name='test_survey'
    )
    
    # Add first frequency
    survey.add_frequency('freq_547500_hz', sample_dataset)
    
    # Add second frequency (copy with different freq)
    ds2 = sample_dataset.copy()
    ds2.attrs['transmit_freq_start'] = 1495000
    ds2.attrs['transmit_freq_end'] = 1495000
    survey.add_frequency('freq_1495000_hz', ds2)
    
    return survey


@pytest.fixture
def logging_mock(mocker):
    """Provide a mock for logging to test log output."""
    return mocker.patch('logging.Logger')


# Markers for organizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring test data files"
    )