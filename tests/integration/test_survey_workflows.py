"""
Integration tests for sidescanpy.

Tests interactions between multiple modules and end-to-end workflows.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile

from sidescanpy.core.survey import SideScanSurvey, MultiFrequencySideScanSurvey


@pytest.mark.integration
class TestSurveyCreationWorkflow:
    """Integration tests for creating and manipulating survey objects."""
    
    def test_create_survey_from_scratch(self):
        """Test creating a complete survey from scratch."""
        # Create test data
        n_ping = 20
        n_channel = 2
        n_sample = 1000
        
        data = {
            'samples': (['ping', 'channel', 'sample'], 
                       np.random.randint(0, 255, (n_ping, n_channel, n_sample), dtype=np.uint8)),
            'ping_number': (['ping'], np.arange(n_ping)),
            'timestamp': (['ping'], np.linspace(0, 100, n_ping)),
            'latitude': (['ping'], np.linspace(41.0, 41.5, n_ping)),
            'longitude': (['ping'], np.linspace(-71.5, -71.0, n_ping)),
            'heading': (['ping'], np.random.uniform(0, 360, n_ping)),
            'depth': (['ping'], np.linspace(10, 100, n_ping)),
        }
        
        coords = {
            'ping': np.arange(n_ping),
            'channel': np.array([0, 1]),
            'sample': np.arange(n_sample),
        }
        
        attrs = {
            'transmit_freq_start': 547500,
            'survey_name': 'integration_test',
        }
        
        dataset = xr.Dataset(data, coords=coords, attrs=attrs)
        
        # Create survey
        survey = SideScanSurvey(dataset, freq_key='freq_547500')
        
        # Verify all properties work
        assert survey.n_pings == n_ping
        assert survey.n_channels == n_channel
        assert survey.n_samples == n_sample
        assert survey.frequency == 547500
    
    def test_multiple_surveys_in_multi_frequency(self):
        """Test MultiFrequencySideScanSurvey can hold multiple surveys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "test.jsf"
            source_file.touch()
            
            survey = MultiFrequencySideScanSurvey(
                source_file=source_file,
                source_format="JSF",
                survey_name="multi_freq_test"
            )
            
            # Verify the survey object is properly initialized
            assert survey.source_file.exists()
            assert survey.source_format == "JSF"


@pytest.mark.integration
class TestDatasetConsistency:
    """Integration tests for dataset consistency across operations."""
    
    def test_survey_preserves_data(self, sample_dataset):
        """Test that creating a survey preserves dataset integrity."""
        survey = SideScanSurvey(sample_dataset, freq_key='test')
        
        # Verify original data is preserved
        original_samples = sample_dataset['samples'].values
        retrieved_samples = survey.dataset['samples'].values
        
        np.testing.assert_array_equal(original_samples, retrieved_samples)
    
    def test_survey_coordinate_access(self, sample_dataset):
        """Test that survey can access all coordinates."""
        survey = SideScanSurvey(sample_dataset, freq_key='test')
        
        # Verify all coordinates are accessible
        assert 'ping' in survey.dataset.coords
        assert 'channel' in survey.dataset.coords
        assert 'sample' in survey.dataset.coords
    
    def test_survey_attributes_preserved(self, sample_dataset):
        """Test that dataset attributes are preserved."""
        survey = SideScanSurvey(sample_dataset, freq_key='test')
        
        assert survey.dataset.attrs['transmit_freq_start'] == 547500
        assert survey.dataset.attrs['survey_name'] == 'test_survey'


@pytest.mark.integration
class TestWorkflowSimulation:
    """Simulate realistic workflows with the module."""
    
    def test_create_analyze_save_workflow(self, tmp_dir, sample_dataset):
        """Test a typical workflow: create, analyze, prepare for save."""
        # Step 1: Create survey
        survey = SideScanSurvey(sample_dataset, freq_key='freq_547500')
        
        # Step 2: Validate properties
        assert survey.n_pings > 0
        assert survey.n_channels > 0
        assert survey.n_samples > 0
        
        # Step 3: Extract subsections (simulating analysis)
        first_ping = survey.dataset.isel(ping=0)
        # After selecting a single ping, 'ping' dimension is removed
        assert 'channel' in first_ping.dims and 'sample' in first_ping.dims
        
        # Step 4: Verify data can be written (prepare path)
        output_file = tmp_dir / "survey_output.nc"
        assert output_file.parent.exists()
    
    def test_multi_survey_workflow(self, tmp_dir):
        """Test workflow with multiple surveys at different frequencies."""
        survey1 = SideScanSurvey(
            self._create_test_dataset(freq=547500),
            freq_key='freq_547500'
        )
        
        survey2 = SideScanSurvey(
            self._create_test_dataset(freq=1495000),
            freq_key='freq_1495000'
        )
        
        # Create multi-frequency container
        source_file = tmp_dir / "test.jsf"
        source_file.touch()
        
        multi_survey = MultiFrequencySideScanSurvey(
            source_file=source_file,
            source_format="JSF",
            survey_name="multi_freq"
        )
        
        # Verify both surveys have different frequencies
        assert survey1.frequency != survey2.frequency
        assert survey1.frequency == 547500
        assert survey2.frequency == 1495000
    
    @staticmethod
    def _create_test_dataset(freq=547500, n_ping=10, n_channel=2, n_sample=500):
        """Helper to create test dataset."""
        data = {
            'samples': (['ping', 'channel', 'sample'], 
                       np.random.randint(0, 255, (n_ping, n_channel, n_sample), dtype=np.uint8)),
            'ping_number': (['ping'], np.arange(n_ping)),
            'timestamp': (['ping'], np.arange(n_ping, dtype=float)),
            'latitude': (['ping'], np.random.uniform(-90, 90, n_ping)),
            'longitude': (['ping'], np.random.uniform(-180, 180, n_ping)),
            'heading': (['ping'], np.random.uniform(0, 360, n_ping)),
            'depth': (['ping'], np.random.uniform(0, 1000, n_ping)),
        }
        
        coords = {
            'ping': np.arange(n_ping),
            'channel': np.array([0, 1]),
            'sample': np.arange(n_sample),
        }
        
        attrs = {'transmit_freq_start': freq, 'survey_name': f'test_{freq}'}
        
        return xr.Dataset(data, coords=coords, attrs=attrs)
