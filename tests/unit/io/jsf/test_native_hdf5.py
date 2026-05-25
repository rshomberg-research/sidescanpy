"""
Unit tests for JSF native HDF5 conversion

2026 rshomberg@uri.edu
"""

import pytest
import h5py
from pathlib import Path

from sidescanpy.io.jsf.native_hdf5 import jsf_to_native_hdf5
from sidescanpy.io.jsf.messages import SonarDataMessage


@pytest.mark.unit
class TestJSFToNativeHDF5:
    """Test JSF to native HDF5 conversion"""

    def test_convert_creates_output_file(self, sample_jsf_messages, tmp_dir):
        """Should create output HDF5 file from pickled messages"""
        output_path = tmp_dir / "output.jsf.h5"
        
        # For this test, we'll manually call the internal function
        # since we have pre-parsed messages
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        
        assert output_path.exists()

    def test_output_has_frequencies_group(self, sample_jsf_messages, tmp_dir):
        """Output should have 'frequencies' group"""
        output_path = tmp_dir / "output.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            assert "frequencies" in f  # type: ignore

    def test_sonar_messages_organized_by_frequency(self, sample_jsf_messages, tmp_dir):
        """Sonar messages should be in frequencies/hz_XXXXX/ groups"""
        output_path = tmp_dir / "output.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        assert len(sonar_msgs) > 0, "Need sonar messages in test data"
        
        with h5py.File(output_path, "r") as f:
            assert "frequencies" in f  # type: ignore
            # Should have at least one frequency group
            assert len(f["frequencies"]) > 0  # type: ignore

    def test_frequency_group_has_backscatter_data(self, sample_jsf_messages, tmp_dir):
        """Frequency groups should contain backscatter datasets"""
        output_path = tmp_dir / "output.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            freq_groups = list(f["frequencies"].keys())  # type: ignore
            assert len(freq_groups) > 0
            
            freq_group = f["frequencies"][freq_groups[0]]  # type: ignore
            assert "backscatter" in freq_group  # type: ignore
            assert "timestamp" in freq_group  # type: ignore
            assert "latitude" in freq_group  # type: ignore
            assert "longitude" in freq_group  # type: ignore

    def test_backscatter_has_correct_shape(self, sample_jsf_messages, tmp_dir):
        """Backscatter should be 3D: (pings, channels, samples)"""
        output_path = tmp_dir / "output.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            freq_groups = list(f["frequencies"].keys())  # type: ignore
            freq_group = f["frequencies"][freq_groups[0]]  # type: ignore
            backscatter = freq_group["backscatter"]  # type: ignore
            
            # Should be 3D
            assert len(backscatter.shape) == 3
            n_pings, n_channels, n_samples = backscatter.shape
            assert n_channels == 2
            assert n_pings > 0
            assert n_samples > 0


@pytest.mark.unit
class TestFileHandling:
    """Test file I/O and error handling"""

    def test_jsf_file_not_found_raises(self, tmp_dir):
        """Should raise FileNotFoundError if JSF doesn't exist"""
        with pytest.raises(FileNotFoundError):
            jsf_to_native_hdf5(
                tmp_dir / "nonexistent.jsf",
                tmp_dir / "output.jsf.h5"
            )

    def test_output_file_exists_raises(self, tmp_dir):
        """Should raise FileExistsError if output exists and overwrite=False"""
        jsf_file = tmp_dir / "dummy.jsf"
        jsf_file.touch()  # Create dummy JSF file
        
        output_path = tmp_dir / "output.jsf.h5"
        output_path.touch()  # Create existing output file
        
        with pytest.raises(FileExistsError):
            jsf_to_native_hdf5(jsf_file, output_path, overwrite=False)

    def test_overwrite_succeeds(self, sample_jsf_messages, tmp_dir):
        """Should overwrite existing file when flag is set"""
        output_path = tmp_dir / "output.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _write_messages_to_hdf5
        
        # Write first time
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        assert output_path.exists()
        size1 = output_path.stat().st_size
        
        # Write again - should not raise
        _write_messages_to_hdf5(sample_jsf_messages, output_path)
        assert output_path.exists()