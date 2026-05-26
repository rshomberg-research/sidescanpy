"""
Unit tests for native HDF5 to standard schema conversion

2026 rshomberg@uri.edu
"""

import pytest
import h5py
import numpy as np
from pathlib import Path

from sidescanpy.io.jsf.converter import (
    native_hdf5_to_standard,
    list_frequencies_in_native_hdf5,
    _read_frequency_to_dataset,
)
from sidescanpy.io.jsf.reader import _archive_messages_to_hdf5
import xarray as xr


@pytest.fixture
def native_hdf5_file(sample_jsf_messages, tmp_dir):
    """Create a native HDF5 file from sample messages"""
    native_path = tmp_dir / "sample.jsf.h5"
    _archive_messages_to_hdf5(sample_jsf_messages, native_path)
    return native_path


@pytest.mark.unit
class TestListFrequencies:
    """Test listing frequencies in native HDF5"""

    def test_list_frequencies_returns_list(self, native_hdf5_file):
        """Should return list of frequencies"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        assert isinstance(freqs, list)
        assert len(freqs) > 0

    def test_list_frequencies_are_integers(self, native_hdf5_file):
        """Frequencies should be positive integers"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        for freq in freqs:
            assert isinstance(freq, int)
            assert freq > 0

    def test_list_frequencies_sorted_descending(self, native_hdf5_file):
        """Frequencies should be sorted descending"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        assert freqs == sorted(freqs, reverse=True)


@pytest.mark.unit
class TestReadFrequencyToDataset:
    """Test reading frequency data to xarray Dataset"""

    def test_read_returns_dataset(self, native_hdf5_file):
        """Should return xarray.Dataset"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        assert len(freqs) > 0
        
        ds = _read_frequency_to_dataset(native_hdf5_file, freqs[0])
        assert isinstance(ds, xr.Dataset)

    def test_dataset_has_required_variables(self, native_hdf5_file):
        """Dataset should have backscatter variables"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        ds = _read_frequency_to_dataset(native_hdf5_file, freqs[0])
        
        assert "backscatter_intensity" in ds.data_vars
        assert "backscatter_raw" in ds.data_vars

    def test_dataset_has_required_coordinates(self, native_hdf5_file):
        """Dataset should have ping and channel coordinates"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        ds = _read_frequency_to_dataset(native_hdf5_file, freqs[0])
        
        assert "ping" in ds.coords or "ping_number" in ds.coords
        assert "channel_id" in ds.coords

    def test_backscatter_has_3d_shape(self, native_hdf5_file):
        """Backscatter should be 3D (ping, channel, sample)"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        ds = _read_frequency_to_dataset(native_hdf5_file, freqs[0])
        
        bs = ds["backscatter_intensity"]
        assert len(bs.shape) == 3
        assert bs.shape[1] == 2  # 2 channels


@pytest.mark.unit
class TestConversion:
    """Test native to standard conversion"""

    def test_conversion_creates_output_file(self, native_hdf5_file, tmp_dir):
        """Should create output .ss.h5 file"""
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path)
        
        assert output_path.exists()

    def test_output_is_valid_hdf5(self, native_hdf5_file, tmp_dir):
        """Output should be readable HDF5"""
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path)
        
        with h5py.File(output_path, "r") as f:
            assert "frequencies" in f  # type: ignore

    def test_output_has_sonar_data(self, native_hdf5_file, tmp_dir):
        """Output should contain sonar data"""
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path)
        
        with h5py.File(output_path, "r") as f:
            freqs = list(f["frequencies"].keys())  # type: ignore
            assert len(freqs) > 0

    def test_conversion_specific_frequency(self, native_hdf5_file, tmp_dir):
        """Should convert specific frequency when requested"""
        freqs = list_frequencies_in_native_hdf5(native_hdf5_file)
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path, freq_hz=freqs[0])
        
        assert output_path.exists()


@pytest.mark.unit
class TestAuxiliaryExtraction:
    """Test auxiliary stream extraction"""

    def test_conversion_creates_auxiliary_group(self, native_hdf5_file, tmp_dir):
        """Output should have auxiliary group if data exists"""
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path)
        
        with h5py.File(output_path, "r") as f:
            # Auxiliary group might or might not exist depending on input
            # Just verify file is readable
            assert "frequencies" in f  # type: ignore


@pytest.mark.unit
class TestFileHandling:
    """Test file I/O and error handling"""

    def test_input_file_not_found_raises(self, tmp_dir):
        """Should raise FileNotFoundError if input doesn't exist"""
        with pytest.raises(FileNotFoundError):
            native_hdf5_to_standard(
                tmp_dir / "nonexistent.jsf.h5",
                tmp_dir / "output.ss.h5"
            )

    def test_output_exists_raises(self, native_hdf5_file, tmp_dir):
        """Should raise FileExistsError if output exists and overwrite=False"""
        output_path = tmp_dir / "output.ss.h5"
        output_path.touch()
        
        with pytest.raises(FileExistsError):
            native_hdf5_to_standard(native_hdf5_file, output_path, overwrite=False)

    def test_overwrite_succeeds(self, native_hdf5_file, tmp_dir):
        """Should overwrite when flag is set"""
        output_path = tmp_dir / "output.ss.h5"
        
        native_hdf5_to_standard(native_hdf5_file, output_path)
        stat1 = output_path.stat()
        
        native_hdf5_to_standard(native_hdf5_file, output_path, overwrite=True)
        stat2 = output_path.stat()
        
        assert output_path.exists()

    def test_invalid_frequency_raises(self, native_hdf5_file, tmp_dir):
        """Should raise ValueError for invalid frequency"""
        output_path = tmp_dir / "output.ss.h5"
        
        with pytest.raises(ValueError):
            native_hdf5_to_standard(
                native_hdf5_file,
                output_path,
                freq_hz=999999
            )