"""
Unit tests for sidescanpy.core.survey module.

Tests the SideScanSurvey and MultiFrequencySideScanSurvey classes.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path

from sidescanpy.core.survey import SideScanSurvey, MultiFrequencySideScanSurvey


@pytest.mark.unit
class TestSideScanSurvey:
    """Unit tests for SideScanSurvey class."""
    
    def test_initialization_valid(self, sample_dataset):
        """Test initialization with valid dataset."""
        survey = SideScanSurvey(sample_dataset, freq_key='freq_547500')
        assert survey.freq_key == 'freq_547500'
        assert survey.dataset is sample_dataset
    
    def test_validation_missing_dimensions(self, sample_dataset):
        """Test that initialization fails without required dimensions."""
        # Remove a required dimension
        invalid_dataset = sample_dataset.drop_dims('channel')
        
        with pytest.raises(ValueError, match="Missing dimensions"):
            SideScanSurvey(invalid_dataset, freq_key='freq_547500')
    
    def test_validation_missing_variables(self, sample_dataset):
        """Test that initialization fails without required variables."""
        # Remove a required variable
        invalid_dataset = sample_dataset.drop_vars('latitude')
        
        with pytest.raises(ValueError, match="Missing variables"):
            SideScanSurvey(invalid_dataset, freq_key='freq_547500')
    
    def test_frequency_property(self, sample_survey_object):
        """Test frequency property extraction."""
        freq = sample_survey_object.frequency
        assert freq == 547500
    
    def test_n_pings_property(self, sample_survey_object):
        """Test number of pings calculation."""
        n_pings = sample_survey_object.n_pings
        assert n_pings == 50
    
    def test_n_channels_property(self, sample_survey_object):
        """Test number of channels calculation."""
        n_channels = sample_survey_object.n_channels
        assert n_channels == 2
    
    def test_n_samples_property(self, sample_survey_object):
        """Test number of samples calculation."""
        n_samples = sample_survey_object.n_samples
        assert n_samples == 5000
    
    def test_repr(self, sample_survey_object):
        """Test string representation."""
        repr_str = repr(sample_survey_object)
        assert "SideScanSurvey" in repr_str
        assert "freq_547500" in repr_str
        assert "50" in repr_str
        assert "channels=2" in repr_str
    
    def test_dataset_property_immutability(self, sample_dataset):
        """Test that accessing dataset property returns original data."""
        survey = SideScanSurvey(sample_dataset, freq_key='test')
        assert survey.dataset is sample_dataset
        # Verify we can access data through the property
        assert 'ping' in survey.dataset.dims


@pytest.mark.unit
class TestMultiFrequencySideScanSurvey:
    """Unit tests for MultiFrequencySideScanSurvey class."""
    
    def test_initialization(self, tmp_dir):
        """Test initialization of MultiFrequencySideScanSurvey."""
        test_file = tmp_dir / "test.jsf"
        test_file.touch()
        
        survey = MultiFrequencySideScanSurvey(
            source_file=test_file,
            source_format="JSF",
            survey_name="test_survey"
        )
        
        assert survey.source_file == test_file
        assert survey.source_format == "JSF"
    
    def test_source_file_stored_correctly(self, sample_multi_frequency_survey):
        """Test that source file is stored correctly."""
        assert isinstance(sample_multi_frequency_survey.source_file, str)
    
    def test_source_format_stored_correctly(self, sample_multi_frequency_survey):
        """Test that source format is stored correctly."""
        assert sample_multi_frequency_survey.source_format == "JSF"


@pytest.mark.unit
class TestDatasetValidation:
    """Test dataset validation logic."""
    
    def test_missing_ping_dimension(self):
        """Test validation catches missing 'ping' dimension."""
        data = {
            'samples': (['channel', 'sample'], np.random.rand(2, 500)),
            'ping_number': (['channel'], [0, 1]),
            'timestamp': (['channel'], [0.0, 1.0]),
            'latitude': (['channel'], [0.0, 1.0]),
            'longitude': (['channel'], [0.0, 1.0]),
            'heading': (['channel'], [0.0, 1.0]),
            'depth': (['channel'], [0.0, 1.0]),
        }
        coords = {'channel': [0, 1], 'sample': np.arange(500)}
        invalid_ds = xr.Dataset(data, coords=coords)
        
        with pytest.raises(ValueError):
            SideScanSurvey(invalid_ds, freq_key='test')
    
    def test_missing_channel_dimension(self):
        """Test validation catches missing 'channel' dimension."""
        data = {
            'samples': (['ping', 'sample'], np.random.rand(10, 500)),
            'ping_number': (['ping'], np.arange(10)),
            'timestamp': (['ping'], np.arange(10, dtype=float)),
            'latitude': (['ping'], np.random.rand(10)),
            'longitude': (['ping'], np.random.rand(10)),
            'heading': (['ping'], np.random.rand(10)),
            'depth': (['ping'], np.random.rand(10)),
        }
        coords = {'ping': np.arange(10), 'sample': np.arange(500)}
        invalid_ds = xr.Dataset(data, coords=coords)
        
        with pytest.raises(ValueError):
            SideScanSurvey(invalid_ds, freq_key='test')
    
    @pytest.mark.parametrize("missing_var", [
        'ping_number', 'timestamp', 'latitude', 
        'longitude', 'heading', 'depth'
    ])
    def test_missing_required_variables(self, sample_dataset, missing_var):
        """Test validation catches each missing required variable."""
        invalid_ds = sample_dataset.drop_vars(missing_var)
        
        with pytest.raises(ValueError, match="Missing variables"):
            SideScanSurvey(invalid_ds, freq_key='test')
