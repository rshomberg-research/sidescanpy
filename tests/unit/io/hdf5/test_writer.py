"""Tests for HDF5 writer module."""

import pytest

from sidescanpy.io.hdf5 import writer



@pytest.mark.unit
class TestHDF5Writer:
    """Tests for HDF5 writer."""
    
    def test_to_hdf5_creates_file(self, sample_multi_frequency_survey, tmp_dir):
        """Test that to_hdf5 creates an HDF5 file."""
        output_file = tmp_dir / "survey.h5"
        
        print(f"\n📝 Writing survey to: {output_file}")
        print(f"  Frequencies: {list(sample_multi_frequency_survey.frequencies)}")
        
        writer.to_hdf5(sample_multi_frequency_survey, output_file)
        
        print(f"  File exists: {output_file.exists()}")
        if output_file.exists():
            print(f"  File size: {output_file.stat().st_size} bytes")
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_to_hdf5_preserves_frequencies(self, sample_multi_frequency_survey, tmp_dir):
        """Test that all frequencies are written to HDF5."""
        output_file = tmp_dir / "survey.h5"
        n_freqs = sample_multi_frequency_survey.n_frequencies
        
        print(f"\n📝 Test: Preserve frequencies")
        print(f"  Expected frequencies: {n_freqs}")
        
        writer.to_hdf5(sample_multi_frequency_survey, output_file)
        
        assert output_file.exists()
        
        # Read it back
        import h5py
        with h5py.File(output_file, 'r') as f:
            freq_groups = [k for k in f.keys() if k.startswith('freq_')]
            print(f"  Found groups: {freq_groups}")
            assert len(freq_groups) == n_freqs
    
    def test_to_hdf5_invalid_path_creates_directory(self, sample_multi_frequency_survey, tmp_dir):
        """Test that nested directories are created if needed."""
        nested_dir = tmp_dir / "deep" / "nested" / "path"
        output_file = nested_dir / "survey.h5"
        
        print(f"\n📝 Test: Create nested directories")
        print(f"  Output path: {output_file}")
        print(f"  Nested dir exists before: {nested_dir.exists()}")
        
        writer.to_hdf5(sample_multi_frequency_survey, output_file)
        
        print(f"  Nested dir exists after: {nested_dir.exists()}")
        print(f"  File exists: {output_file.exists()}")
        
        assert nested_dir.exists()
        assert output_file.exists()
    
    def test_to_hdf5_source_format_metadata(self, sample_multi_frequency_survey, tmp_dir):
        """Test that survey metadata is preserved in HDF5."""
        output_file = tmp_dir / "survey.h5"
        
        writer.to_hdf5(sample_multi_frequency_survey, output_file)
        
        # Read back and check attributes
        import xarray as xr
        import h5py
        
        with h5py.File(output_file, 'r') as f:
            freq_key = 'freq_547500_hz'
            if freq_key in f:
                ds = xr.open_dataset(output_file, group=freq_key, engine='h5netcdf')
                assert 'transmit_freq_start' in ds.attrs
                assert ds.attrs['transmit_freq_start'] == 547500


@pytest.mark.integration
class TestHDF5WriterIntegration:
    """Integration tests for HDF5 writer."""
    
    def test_write_multiple_surveys(self, sample_multi_frequency_survey, tmp_dir):
        """Test writing multiple surveys to separate files."""
        files = []
        
        for i in range(3):
            output_file = tmp_dir / f"survey_{i}.h5"
            writer.to_hdf5(sample_multi_frequency_survey, output_file)
            assert output_file.exists()
            files.append(output_file)
        
        print(f"\n✓ Created {len(files)} HDF5 files")
        assert len(files) == 3