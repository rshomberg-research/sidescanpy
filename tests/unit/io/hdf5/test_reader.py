"""Tests for HDF5 reader module."""

import pytest
import logging
import numpy as np
from pathlib import Path

from sidescanpy.io.hdf5 import reader, writer
from sidescanpy.core.survey import MultiFrequencySideScanSurvey

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.unit
class TestHDF5Reader:
    """Tests for HDF5 reader."""
    
    @pytest.fixture
    def hdf5_file_with_data(self, sample_multi_frequency_survey, tmp_dir):
        """Fixture: Create an HDF5 file with test data."""
        output_file = tmp_dir / "survey.h5"
        writer.to_hdf5(sample_multi_frequency_survey, output_file)
        
        print(f"\n✓ Created test HDF5 file: {output_file}")
        
        return output_file
    
    def test_from_hdf5_reads_file(self, hdf5_file_with_data):
        """Test that from_hdf5 successfully reads an HDF5 file."""
        survey = reader.from_hdf5(hdf5_file_with_data)
        
        print(f"\n✓ Read survey from HDF5")
        print(f"  Type: {type(survey)}")
        print(f"  n_frequencies: {survey.n_frequencies}")
        
        assert survey is not None
        assert isinstance(survey, MultiFrequencySideScanSurvey)
    
    def test_from_hdf5_preserves_frequencies(self, hdf5_file_with_data, sample_multi_frequency_survey):
        """Test that all frequencies are read from HDF5."""
        survey = reader.from_hdf5(hdf5_file_with_data)
        
        print(f"\n✓ Frequencies read: {list(survey.frequencies)}")
        print(f"  Original: {list(sample_multi_frequency_survey.frequencies)}")
        
        assert survey.n_frequencies == sample_multi_frequency_survey.n_frequencies
        assert set(survey.frequencies) == set(sample_multi_frequency_survey.frequencies)
    
    def test_from_hdf5_preserves_data(self, hdf5_file_with_data):
        """Test that data is correctly preserved when reading."""
        survey = reader.from_hdf5(hdf5_file_with_data)
        
        freq_key = 'freq_547500_hz'
        freq_survey = survey.get_frequency(freq_key)
        ds = freq_survey.dataset
        
        print(f"\n✓ Dataset from {freq_key}:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Variables: {list(ds.data_vars)}")
        
        # Check dimensions
        assert 'ping' in ds.sizes
        assert 'channel' in ds.sizes
        assert 'sample' in ds.sizes
        assert ds.sizes['channel'] == 2
        
        # Check data variables
        required_vars = {'samples', 'ping_number', 'timestamp', 
                        'latitude', 'longitude', 'heading', 'depth'}
        assert required_vars.issubset(set(ds.data_vars))
    
    def test_from_hdf5_file_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            reader.from_hdf5(Path('/nonexistent/path/survey.h5'))
    
    def test_from_hdf5_empty_file_raises_error(self, tmp_dir):
        """Test that error is raised for empty HDF5 file."""
        empty_file = tmp_dir / "empty.h5"
        
        # Create an empty HDF5 file with no freq_* groups
        import h5py
        with h5py.File(str(empty_file), 'w') as f:
            f.create_group('dummy')
        
        print(f"\n✓ Created empty HDF5 file: {empty_file}")
        
        with pytest.raises(FileNotFoundError, match="freq_"):
            reader.from_hdf5(empty_file)
    
    def test_from_hdf5_attributes_preserved(self, hdf5_file_with_data):
        """Test that dataset attributes are preserved."""
        survey = reader.from_hdf5(hdf5_file_with_data)
        
        freq_key = 'freq_547500_hz'
        freq_survey = survey.get_frequency(freq_key)
        ds = freq_survey.dataset
        
        print(f"\n✓ Attributes: {dict(ds.attrs)}")
        
        assert 'transmit_freq_start' in ds.attrs
        assert 'transmit_freq_end' in ds.attrs


@pytest.mark.integration
class TestHDF5ReadWriteWorkflow:
    """Integration tests for HDF5 read/write workflows."""
    
    def test_roundtrip_write_read(self, sample_multi_frequency_survey, tmp_dir):
        """Test write → read roundtrip preserves data integrity."""
        hdf5_file = tmp_dir / "roundtrip.h5"
        
        # Write to HDF5
        writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
        
        # Read back
        restored_survey = reader.from_hdf5(hdf5_file)
        
        print(f"\n✓ Roundtrip test:")
        print(f"  Original frequencies: {list(sample_multi_frequency_survey.frequencies)}")
        print(f"  Restored frequencies: {list(restored_survey.frequencies)}")
        
        # Verify structure matches
        assert sample_multi_frequency_survey.n_frequencies == restored_survey.n_frequencies
        
        for freq_key in sample_multi_frequency_survey.frequencies:
            original_ds = sample_multi_frequency_survey.get_frequency(freq_key).dataset
            restored_ds = restored_survey.get_frequency(freq_key).dataset
            
            # Check dimensions match
            assert original_ds.sizes == restored_ds.sizes
            
            # Check variables exist
            assert set(original_ds.data_vars) == set(restored_ds.data_vars)
            
            print(f"  ✓ {freq_key} matches")
    
    def test_read_write_preserves_survey_metadata(self, sample_multi_frequency_survey, tmp_dir):
        """Test that survey metadata is preserved through write/read cycle."""
        hdf5_file = tmp_dir / "metadata.h5"
        
        writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
        restored_survey = reader.from_hdf5(hdf5_file)
        
        assert restored_survey.source_format == 'hdf5'
        assert list(sample_multi_frequency_survey.frequencies) == list(restored_survey.frequencies)
        
        print(f"✓ Metadata preserved")