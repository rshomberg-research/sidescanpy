"""Tests for netCDF writer module."""

import pytest
from pathlib import Path

from sidescanpy.io.netcdf import writer


@pytest.mark.unit
class TestNetCDFWriter:
    """Tests for netCDF writer."""
    
    def test_to_netcdf_creates_files(self, sample_multi_frequency_survey, tmp_dir):
        """Test that to_netcdf creates one file per frequency."""
        output_dir = tmp_dir / "netcdf_output"
        
        print(f"\n📝 Writing survey to netCDF directory: {output_dir}")
        print(f"  Frequencies: {list(sample_multi_frequency_survey.frequencies)}")
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        print(f"  Created files: {[f.name for f in output_files]}")
        
        assert output_dir.exists()
        assert len(output_files) == sample_multi_frequency_survey.n_frequencies
        assert all(f.exists() for f in output_files)
    
    def test_to_netcdf_file_naming_convention(self, sample_multi_frequency_survey, tmp_dir):
        """Test that files follow freq_*_hz.nc naming convention."""
        output_dir = tmp_dir / "netcdf_naming"
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        print(f"\n📝 File names:")
        for f in output_files:
            print(f"  {f.name}")
        
        # Check naming convention
        for output_file in output_files:
            assert output_file.suffix == '.nc'
            assert output_file.parent.name.startswith('freq_')
            assert output_file.parent.name.endswith('_hz')
            assert output_file.name.endswith('.nc')
    
    def test_to_netcdf_file_content(self, sample_multi_frequency_survey, tmp_dir):
        """Test that netCDF files contain correct data."""
        output_dir = tmp_dir / "netcdf_content"
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        # Read one file back and verify
        import xarray as xr
        first_file = output_files[0]
        ds = xr.open_dataset(first_file, engine='netcdf4')
        
        print(f"\n📝 File content for {first_file.name}:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Variables: {list(ds.data_vars)}")
        
        # Verify structure
        assert 'ping' in ds.sizes
        assert 'channel' in ds.sizes
        assert 'sample' in ds.sizes
        
        required_vars = {'samples', 'ping_number', 'timestamp', 
                        'latitude', 'longitude', 'heading', 'depth'}
        assert required_vars.issubset(set(ds.data_vars))
    
    def test_to_netcdf_file_size_reasonable(self, sample_multi_frequency_survey, tmp_dir):
        """Test that netCDF files have reasonable size."""
        output_dir = tmp_dir / "netcdf_size"
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        print(f"\n📝 File sizes:")
        for f in output_files:
            size = f.stat().st_size
            print(f"  {f.name}: {size} bytes")
            assert size > 0, f"File is empty: {f}"
    
    def test_to_netcdf_creates_nested_directories(self, sample_multi_frequency_survey, tmp_dir):
        """Test that nested output directories are created."""
        nested_dir = tmp_dir / "deep" / "nested" / "netcdf"
        
        print(f"\n📝 Creating nested directory: {nested_dir}")
        print(f"  Exists before: {nested_dir.exists()}")
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, nested_dir)
        
        print(f"  Exists after: {nested_dir.exists()}")
        print(f"  Files created: {len(output_files)}")
        
        assert nested_dir.exists()
        assert len(output_files) == sample_multi_frequency_survey.n_frequencies
        assert all(f.parent.parent == nested_dir for f in output_files)
    
    def test_to_netcdf_attributes_preserved(self, sample_multi_frequency_survey, tmp_dir):
        """Test that dataset attributes are preserved in netCDF files."""
        output_dir = tmp_dir / "netcdf_attrs"
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        import xarray as xr
        ds = xr.open_dataset(output_files[0], engine='netcdf4')
        
        print(f"\n📝 Attributes in {output_files[0].name}:")
        for key, val in ds.attrs.items():
            print(f"  {key}: {val}")
        
        assert 'transmit_freq_start' in ds.attrs
        assert 'transmit_freq_end' in ds.attrs
    
    def test_to_netcdf_return_value(self, sample_multi_frequency_survey, tmp_dir):
        """Test that to_netcdf returns correct list of paths."""
        output_dir = tmp_dir / "netcdf_return"
        
        output_files = writer.to_netcdf(sample_multi_frequency_survey, output_dir)
        
        print(f"\n📝 Return value check:")
        print(f"  Type: {type(output_files)}")
        print(f"  Length: {len(output_files)}")
        
        assert isinstance(output_files, list)
        assert len(output_files) == sample_multi_frequency_survey.n_frequencies
        assert all(isinstance(f, Path) for f in output_files)
        assert all(f.exists() for f in output_files)

