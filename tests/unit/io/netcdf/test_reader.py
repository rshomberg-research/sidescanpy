"""Tests for netCDF reader module."""

import pytest
import numpy as np
from pathlib import Path

from sidescanpy.io.netcdf import reader, writer
from sidescanpy.core.survey import MultiFrequencySideScanSurvey


@pytest.mark.unit
class TestNetCDFReader:
    """Tests for netCDF reader."""
    
    @pytest.fixture
    def netcdf_directory_with_data(self, sample_multi_frequency_survey, tmp_dir):
        """Fixture: Create netCDF files with test data."""
        output_dir = tmp_dir / "netcdf_data"
        writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        print(f"\n✓ Created netCDF directory: {output_dir}")
        print(f"  Files: {[f.name for f in sorted(output_dir.glob('freq_*.nc'))]}")
        
        return output_dir
    
    def test_from_netcdf_reads_directory(self, netcdf_directory_with_data):
        """Test that from_netcdf successfully reads a directory."""
        survey = reader.from_netcdf(netcdf_directory_with_data)
        
        print(f"\n✓ Read survey from netCDF directory")
        print(f"  Type: {type(survey)}")
        print(f"  n_frequencies: {survey.n_frequencies}")
        
        assert survey is not None
        assert isinstance(survey, MultiFrequencySideScanSurvey)
    
    def test_from_netcdf_preserves_frequencies(self, netcdf_directory_with_data, sample_multi_frequency_survey):
        """Test that all frequencies are read from netCDF files."""
        survey = reader.from_netcdf(netcdf_directory_with_data)
        
        print(f"\n✓ Frequencies read: {list(survey.frequencies)}")
        print(f"  Original: {list(sample_multi_frequency_survey.frequencies)}")
        
        assert survey.n_frequencies == sample_multi_frequency_survey.n_frequencies
        assert set(survey.frequencies) == set(sample_multi_frequency_survey.frequencies)
    
    def test_from_netcdf_preserves_data(self, netcdf_directory_with_data):
        """Test that data is correctly preserved when reading."""
        survey = reader.from_netcdf(netcdf_directory_with_data)
        
        freq_key = list(survey.frequencies)[0]
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
    
    def test_from_netcdf_directory_not_found_raises_error(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            reader.from_netcdf(Path('/nonexistent/path/to/netcdf'))
    
    def test_from_netcdf_empty_directory_raises_error(self, tmp_dir):
        """Test that error is raised for directory with no freq_*.nc files."""
        empty_dir = tmp_dir / "empty_netcdf"
        empty_dir.mkdir()
        
        # Create some files that don't match pattern
        (empty_dir / "data.txt").write_text("test")
        (empty_dir / "survey.nc").write_text("test")
        
        print(f"\n✓ Created empty netCDF directory: {empty_dir}")
        print(f"  Contents: {[f.name for f in empty_dir.iterdir()]}")
        
        with pytest.raises(FileNotFoundError, match="No freq_.*\\.nc files found"):
            reader.from_netcdf(empty_dir)
    
    def test_from_netcdf_attributes_preserved(self, netcdf_directory_with_data):
        """Test that dataset attributes are preserved."""
        survey = reader.from_netcdf(netcdf_directory_with_data)
        
        freq_key = list(survey.frequencies)[0]
        freq_survey = survey.get_frequency(freq_key)
        ds = freq_survey.dataset
        
        print(f"\n✓ Attributes: {dict(ds.attrs)}")
        
        assert 'transmit_freq_start' in ds.attrs
        assert 'transmit_freq_end' in ds.attrs
    
    def test_from_netcdf_frequency_key_from_filename(self, netcdf_directory_with_data):
        """Test that frequency keys match the filename stems."""
        nc_files = sorted(netcdf_directory_with_data.glob('freq_*.nc'))
        survey = reader.from_netcdf(netcdf_directory_with_data)
        
        expected_keys = {f.stem for f in nc_files}
        actual_keys = set(survey.frequencies)
        
        print(f"\n✓ Frequency keys:")
        print(f"  Expected (from filenames): {expected_keys}")
        print(f"  Actual (from survey): {actual_keys}")
        
        assert expected_keys == actual_keys
    
    def test_from_netcdf_reads_all_available_frequencies(self, tmp_dir, sample_dataset):
        """Test that all freq_*.nc files in directory are read."""
        import xarray as xr
        
        netcdf_dir = tmp_dir / "all_frequencies"
        netcdf_dir.mkdir()
        
        # Create multiple frequency files manually
        frequencies = [547500, 1495000, 2000000]
        for freq in frequencies:
            ds = sample_dataset.copy()
            ds.attrs['transmit_freq_start'] = freq
            ds.attrs['transmit_freq_end'] = freq
            
            freq_key = f"freq_{freq}_hz"
            output_file = netcdf_dir / f"{freq_key}.nc"
            ds.to_netcdf(output_file, engine='netcdf4')
        
        print(f"\n✓ Created {len(frequencies)} frequency files")
        
        survey = reader.from_netcdf(netcdf_dir)
        
        print(f"  Read {survey.n_frequencies} frequencies")
        
        assert survey.n_frequencies == len(frequencies)
        for freq in frequencies:
            freq_key = f"freq_{freq}_hz"
            assert freq_key in survey.frequencies
    
    def test_from_netcdf_returns_correct_type(self, netcdf_directory_with_data):
        """Test that from_netcdf returns MultiFrequencySideScanSurvey."""
        result = reader.from_netcdf(netcdf_directory_with_data)
        
        print(f"\n✓ Return type: {type(result).__name__}")
        
        assert isinstance(result, MultiFrequencySideScanSurvey)
        assert hasattr(result, 'frequencies')
        assert hasattr(result, 'get_frequency')
