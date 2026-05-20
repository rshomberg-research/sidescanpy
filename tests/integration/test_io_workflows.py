"""Integration tests for I/O modules (HDF5, netCDF, JSF)."""

import pytest
import numpy as np
from pathlib import Path

from sidescanpy.io.hdf5 import reader as hdf5_reader, writer as hdf5_writer
from sidescanpy.io.netcdf import reader as netcdf_reader, writer as netcdf_writer


@pytest.mark.integration
class TestHDF5ReadWriteWorkflow:
    """Integration tests for HDF5 read/write workflows."""
    
    def test_roundtrip_write_read(self, sample_multi_frequency_survey, tmp_dir):
        """Test write → read roundtrip preserves data integrity."""
        hdf5_file = tmp_dir / "roundtrip.h5"
        
        # Write to HDF5
        hdf5_writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
        
        # Read back
        restored_survey = hdf5_reader.from_hdf5(hdf5_file)
        
        print(f"\n✓ HDF5 Roundtrip test:")
        print(f"  Original frequencies: {list(sample_multi_frequency_survey.frequencies)}")
        print(f"  Restored frequencies: {list(restored_survey.frequencies)}")
        
        assert sample_multi_frequency_survey.n_frequencies == restored_survey.n_frequencies
    
    def test_read_write_preserves_survey_metadata_hdf5(self, sample_multi_frequency_survey, tmp_dir):
        """Test that survey metadata is preserved through HDF5 write/read cycle."""
        hdf5_file = tmp_dir / "metadata.h5"
        
        hdf5_writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
        restored_survey = hdf5_reader.from_hdf5(hdf5_file)
        
        assert restored_survey.source_format == 'hdf5'
        assert list(sample_multi_frequency_survey.frequencies) == list(restored_survey.frequencies)
        
        print(f"✓ HDF5 Metadata preserved")
    
    def test_write_multiple_surveys_hdf5(self, sample_multi_frequency_survey, tmp_dir):
        """Test writing multiple surveys to separate HDF5 files and reading them back."""
        files = []
        
        for i in range(3):
            hdf5_file = tmp_dir / f"survey_{i}.h5"
            hdf5_writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
            files.append(hdf5_file)
        
        print(f"\n✓ Created {len(files)} HDF5 files")
        
        for i, hdf5_file in enumerate(files):
            restored = hdf5_reader.from_hdf5(hdf5_file)
            assert restored.n_frequencies == sample_multi_frequency_survey.n_frequencies
            print(f"  Survey {i} roundtrip successful")


@pytest.mark.integration
class TestNetCDFReadWriteWorkflow:
    """Integration tests for netCDF read/write workflows."""
    
    def test_roundtrip_write_read(self, sample_multi_frequency_survey, tmp_dir):
        """Test write → read roundtrip preserves data integrity."""
        netcdf_dir = tmp_dir / "roundtrip"
        
        # Write to netCDF
        netcdf_writer.to_netcdf(sample_multi_frequency_survey, netcdf_dir)
        
        # Read back
        restored_survey = netcdf_reader.from_netcdf(netcdf_dir)
        
        print(f"\n✓ netCDF Roundtrip test:")
        print(f"  Original frequencies: {list(sample_multi_frequency_survey.frequencies)}")
        print(f"  Restored frequencies: {list(restored_survey.frequencies)}")
        
        assert sample_multi_frequency_survey.n_frequencies == restored_survey.n_frequencies
    
    def test_read_write_preserves_survey_metadata_netcdf(self, sample_multi_frequency_survey, tmp_dir):
        """Test that survey metadata is preserved through netCDF write/read cycle."""
        netcdf_dir = tmp_dir / "metadata"
        
        netcdf_writer.to_netcdf(sample_multi_frequency_survey, netcdf_dir)
        restored_survey = netcdf_reader.from_netcdf(netcdf_dir)
        
        assert restored_survey.source_format == 'netcdf'
        assert list(sample_multi_frequency_survey.frequencies) == list(restored_survey.frequencies)
        
        print(f"✓ netCDF Metadata preserved")
    
    def test_read_multiple_surveys_netcdf(self, sample_multi_frequency_survey, tmp_dir):
        """Test reading multiple surveys from separate directories."""
        print(f"\n✓ Reading multiple surveys from netCDF")
        
        surveys = []
        for i in range(3):
            netcdf_dir = tmp_dir / f"survey_{i}"
            netcdf_writer.to_netcdf(sample_multi_frequency_survey, netcdf_dir)
            
            survey = netcdf_reader.from_netcdf(netcdf_dir)
            surveys.append(survey)
            
            print(f"  Survey {i}: {survey.n_frequencies} frequencies")
        
        assert len(surveys) == 3
        assert all(s.n_frequencies == sample_multi_frequency_survey.n_frequencies 
                  for s in surveys)


@pytest.mark.integration
class TestIOFormatConversion:
    """Integration tests for converting between I/O formats."""
    
    def test_hdf5_to_netcdf_conversion(self, sample_multi_frequency_survey, tmp_dir):
        """Test writing to HDF5 then reading and writing to netCDF."""
        # Write to HDF5
        hdf5_file = tmp_dir / "survey.h5"
        hdf5_writer.to_hdf5(sample_multi_frequency_survey, hdf5_file)
        
        # Read from HDF5
        from_hdf5 = hdf5_reader.from_hdf5(hdf5_file)
        
        # Write to netCDF
        netcdf_dir = tmp_dir / "converted"
        netcdf_writer.to_netcdf(from_hdf5, netcdf_dir)
        
        # Read from netCDF
        from_netcdf = netcdf_reader.from_netcdf(netcdf_dir)
        
        print(f"\n✓ HDF5 → netCDF conversion:")
        print(f"  Original: {sample_multi_frequency_survey.n_frequencies} frequencies")
        print(f"  Via HDF5: {from_hdf5.n_frequencies} frequencies")
        print(f"  Via netCDF: {from_netcdf.n_frequencies} frequencies")
        
        assert from_hdf5.n_frequencies == from_netcdf.n_frequencies
        assert from_hdf5.n_frequencies == sample_multi_frequency_survey.n_frequencies
    
    def test_netcdf_to_hdf5_conversion(self, sample_multi_frequency_survey, tmp_dir):
        """Test writing to netCDF then reading and writing to HDF5."""
        # Write to netCDF
        netcdf_dir = tmp_dir / "source"
        netcdf_writer.to_netcdf(sample_multi_frequency_survey, netcdf_dir)
        
        # Read from netCDF
        from_netcdf = netcdf_reader.from_netcdf(netcdf_dir)
        
        # Write to HDF5
        hdf5_file = tmp_dir / "converted.h5"
        hdf5_writer.to_hdf5(from_netcdf, hdf5_file)
        
        # Read from HDF5
        from_hdf5 = hdf5_reader.from_hdf5(hdf5_file)
        
        print(f"\n✓ netCDF → HDF5 conversion:")
        print(f"  Original: {sample_multi_frequency_survey.n_frequencies} frequencies")
        print(f"  Via netCDF: {from_netcdf.n_frequencies} frequencies")
        print(f"  Via HDF5: {from_hdf5.n_frequencies} frequencies")
        
        assert from_netcdf.n_frequencies == from_hdf5.n_frequencies
        assert from_netcdf.n_frequencies == sample_multi_frequency_survey.n_frequencies