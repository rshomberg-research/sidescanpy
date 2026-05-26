"""
Unit tests for JSF to native HDF5 archival

2026 rshomberg@uri.edu
"""

import pytest
import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from sidescanpy.io.jsf.native_hdf5 import jsf_to_native_hdf5, _store_field_as_dataset
from sidescanpy.io.jsf.messages import SonarDataMessage


@pytest.mark.unit
class TestArchiveCreation:
    """Test HDF5 archive creation"""

    def test_archive_creates_file(self, sample_jsf_messages, tmp_dir):
        """Should create output .jsf.h5 file"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        assert output_path.exists()

    def test_archive_has_message_types_group(self, sample_jsf_messages, tmp_dir):
        """Archive should have 'message_types' root group"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            assert "message_types" in f  # type: ignore


@pytest.mark.unit
class TestSonarMessageStorage:
    """Test sonar message archival organized by frequency"""

    def test_sonar_messages_grouped_by_frequency(self, sample_jsf_messages, tmp_dir):
        """SonarDataMessages should be in message_types/SonarDataMessage/hz_XXXXX/"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        if not sonar_msgs:
            pytest.skip("No sonar messages in sample data")
        
        with h5py.File(output_path, "r") as f:
            assert "message_types/SonarDataMessage" in f  # type: ignore
            sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
            assert len(sonar_group) > 0  # type: ignore

    def test_sonar_frequency_groups_named_correctly(self, sample_jsf_messages, tmp_dir):
        """Frequency groups should be named hz_XXXXX"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        if not sonar_msgs:
            pytest.skip("No sonar messages in sample data")
        
        with h5py.File(output_path, "r") as f:
            sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
            for freq_key in sonar_group.keys():  # type: ignore
                assert freq_key.startswith("hz_")
                freq_str = freq_key[3:]
                assert freq_str.isdigit()

    def test_sonar_fields_stored_as_datasets(self, sample_jsf_messages, tmp_dir):
        """Each field should be a dataset in frequency group"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        if not sonar_msgs:
            pytest.skip("No sonar messages in sample data")
        
        with h5py.File(output_path, "r") as f:
            sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
            freq_key = list(sonar_group.keys())[0]  # type: ignore
            freq_group = sonar_group[freq_key]  # type: ignore
            
            # Should have multiple fields
            fields = list(freq_group.keys())  # type: ignore
            assert len(fields) > 5


@pytest.mark.unit
class TestMessageTypeStorage:
    """Test non-sonar messages organized by type"""

    def test_other_message_types_stored(self, sample_jsf_messages, tmp_dir):
        """Non-sonar messages should be grouped by type"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            msg_types_group = f["message_types"]  # type: ignore
            
            # Should have at least SonarDataMessage
            assert len(msg_types_group) > 0  # type: ignore

    def test_message_type_groups_have_fields(self, sample_jsf_messages, tmp_dir):
        """Message type groups should contain field datasets"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        with h5py.File(output_path, "r") as f:
            msg_types = f["message_types"]  # type: ignore
            
            for msg_type in msg_types.keys():  # type: ignore
                type_group = msg_types[msg_type]  # type: ignore
                
                # Skip if it's SonarDataMessage (has frequency subgroups)
                if msg_type == "SonarDataMessage":
                    # Check frequency subgroups
                    for freq_key in type_group.keys():  # type: ignore
                        freq_group = type_group[freq_key]  # type: ignore
                        assert len(freq_group) > 0  # type: ignore
                else:
                    # Should have fields
                    if len(type_group) > 0:  # type: ignore
                        assert any(isinstance(v, h5py.Dataset) for v in type_group.values())  # type: ignore


@pytest.mark.unit
class TestFieldStorage:
    """Test individual field storage"""

    def test_store_numeric_fields(self, tmp_dir):
        """Should store numeric fields correctly"""
        output_path = tmp_dir / "test.h5"
        
        with h5py.File(output_path, "w") as f:
            group = f.create_group("test")
            
            # Integer
            _store_field_as_dataset(group, "int_field", [1, 2, 3, 4])
            assert "int_field" in group  # type: ignore
            assert group["int_field"].dtype in [np.int64, np.int32]  # type: ignore
            
            # Float
            _store_field_as_dataset(group, "float_field", [1.5, 2.5, 3.5])
            assert "float_field" in group  # type: ignore
            assert group["float_field"].dtype in [np.float64, np.float32]  # type: ignore

    def test_store_string_fields(self, tmp_dir):
        """Should store string fields correctly"""
        output_path = tmp_dir / "test.h5"
        
        with h5py.File(output_path, "w") as f:
            group = f.create_group("test")
            
            _store_field_as_dataset(group, "str_field", ["hello", "world"])
            assert "str_field" in group  # type: ignore

    def test_store_array_fields(self, tmp_dir):
        """Should store numpy array fields"""
        output_path = tmp_dir / "test.h5"
        
        with h5py.File(output_path, "w") as f:
            group = f.create_group("test")
            
            arrays = [
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                np.array([7, 8, 9]),
            ]
            _store_field_as_dataset(group, "array_field", arrays)
            assert "array_field" in group  # type: ignore

    def test_skip_all_none_fields(self, tmp_dir):
        """Should skip fields that are all None"""
        output_path = tmp_dir / "test.h5"
        
        with h5py.File(output_path, "w") as f:
            group = f.create_group("test")
            
            _store_field_as_dataset(group, "none_field", [None, None, None])
            assert "none_field" not in group  # type: ignore


@pytest.mark.unit
class TestFileHandling:
    """Test file I/O and error handling"""

    def test_jsf_file_not_found_raises(self, tmp_dir):
        """Should raise FileNotFoundError if JSF doesn't exist"""
        with pytest.raises(FileNotFoundError):
            jsf_to_native_hdf5(
                tmp_dir / "nonexistent.jsf",
                tmp_dir / "archive.jsf.h5"
            )

    def test_output_exists_raises(self, tmp_dir):
        """Should raise FileExistsError if output exists and overwrite=False"""
        jsf_file = tmp_dir / "dummy.jsf"
        jsf_file.touch()
        
        output_path = tmp_dir / "archive.jsf.h5"
        output_path.touch()
        
        with pytest.raises(FileExistsError):
            jsf_to_native_hdf5(jsf_file, output_path, overwrite=False)

    def test_overwrite_succeeds(self, sample_jsf_messages, tmp_dir):
        """Should overwrite when flag is set"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        assert output_path.exists()
        
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        assert output_path.exists()


@pytest.mark.unit
class TestDataFidelity:
    """Test that data is preserved accurately"""

    def test_sonar_message_count_preserved(self, sample_jsf_messages, tmp_dir):
        """Number of messages should be preserved"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        if not sonar_msgs:
            pytest.skip("No sonar messages in sample data")
        
        with h5py.File(output_path, "r") as f:
            sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
            
            total_archived = 0
            for freq_key in sonar_group.keys():  # type: ignore
                freq_group = sonar_group[freq_key]  # type: ignore
                # Get count from first field's dataset
                if len(freq_group) > 0:  # type: ignore
                    first_field = list(freq_group.keys())[0]  # type: ignore
                    total_archived += len(freq_group[first_field])  # type: ignore
            
            assert total_archived == len(sonar_msgs)

    def test_ping_numbers_preserved(self, sample_jsf_messages, tmp_dir):
        """Ping numbers should be preserved exactly per frequency"""
        output_path = tmp_dir / "archive.jsf.h5"
        
        from sidescanpy.io.jsf.native_hdf5 import _archive_messages_to_hdf5
        _archive_messages_to_hdf5(sample_jsf_messages, output_path)
        
        sonar_msgs = [m for m in sample_jsf_messages if isinstance(m, SonarDataMessage)]
        if not sonar_msgs:
            pytest.skip("No sonar messages in sample data")
        
        # Group original messages by frequency
        orig_by_freq = defaultdict(list)
        for msg in sonar_msgs:
            freq_hz = int(msg.transmit_freq_start)
            orig_by_freq[freq_hz].append(msg)
        
        with h5py.File(output_path, "r") as f:
            sonar_group = f["message_types/SonarDataMessage"]  # type: ignore
            
            # Check each frequency
            for freq_key in sonar_group.keys():  # type: ignore
                freq_hz = int(freq_key[3:])  # Extract hz_XXXXX
                freq_group = sonar_group[freq_key]  # type: ignore
                
                if "ping_number" in freq_group:  # type: ignore
                    archived_ping_numbers = freq_group["ping_number"][:]  # type: ignore
                    original_ping_numbers = [m.ping_number for m in orig_by_freq[freq_hz]]
                    
                    np.testing.assert_array_equal(
                        archived_ping_numbers,
                        original_ping_numbers,
                        err_msg=f"Ping numbers mismatch for frequency {freq_hz}"
                    )