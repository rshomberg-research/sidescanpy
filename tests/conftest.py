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
from typing import Generator


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a sample xarray Dataset matching converter.py output structure.
    
    This dataset has the structure:
    - dims: [ping, channel, sample]
    - metadata: [ping, channel]
    """
    n_ping = 50
    n_channel = 2
    n_sample = 5000
    
    data = {
        'samples': (['ping', 'channel', 'sample'], 
                   np.random.randint(0, 255, (n_ping, n_channel, n_sample), dtype=np.uint8)),
        'ping_number': (['ping', 'channel'], 
                       np.tile(np.arange(n_ping), (n_channel, 1)).T),
        'timestamp': (['ping', 'channel'], 
                     np.tile(np.arange(n_ping, dtype=float), (n_channel, 1)).T),
        'latitude': (['ping', 'channel'], 
                    np.random.uniform(-90, 90, (n_ping, n_channel))),
        'longitude': (['ping', 'channel'], 
                     np.random.uniform(-180, 180, (n_ping, n_channel))),
        'heading': (['ping', 'channel'], 
                   np.random.uniform(0, 360, (n_ping, n_channel))),
        'depth': (['ping', 'channel'], 
                 np.random.uniform(0, 1000, (n_ping, n_channel))),
    }
    
    coords = {
        'ping': np.arange(n_ping),
        'channel': np.array([0, 1]),
        'sample': np.arange(n_sample),
    }
    
    attrs = {
        'transmit_freq_start': 547500,
        'transmit_freq_end': 547500,
        'survey_name': 'test_survey',
        'platform': 'test_platform',
    }
    
    return xr.Dataset(data, coords=coords, attrs=attrs)


@pytest.fixture
def sample_survey_object(sample_dataset):
    """Create a SideScanSurvey object for testing."""
    from sidescanpy.core.survey import SideScanSurvey
    return SideScanSurvey(sample_dataset, freq_key='freq_547500_hz')


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