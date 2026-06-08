"""
Unit tests for JSF reader module.

Tests core reading functionality, message inspection, and export utilities.
Uses fixtures from conftest.py for real message data.

2026 rshomberg@uri.edu
"""

import pytest
from pathlib import Path
import json
from typing import List
import numpy as np

from sidescanpy.io.jsf.messages import JSFMessage, SonarDataMessage
from sidescanpy.io.jsf.reader import (
    read_jsf,
    read_jsf_iter,
    get_messages_by_index,
    filter_messages_by_type,
    filter_messages_by_frequency,
    get_message_statistics,
    export_messages_to_json,
    export_messages_by_type_to_json,
    save_messages_to_pickle,
    load_messages_from_pickle,
)


# =============================================================================
# SECTION 1: Basic Reading Tests
# =============================================================================

class TestReadJSFBasic:
    """Test fundamental JSF reading functionality."""
    
    def test_read_jsf_returns_list(self, real_jsf_file_path):
        """read_jsf() should return a list of messages."""
        messages = read_jsf(real_jsf_file_path, N=100)
        assert isinstance(messages, list)
        assert len(messages) > 0
    
    def test_read_jsf_respects_limit(self, real_jsf_file_path):
        """read_jsf() with N parameter should limit message count."""
        messages_100 = read_jsf(real_jsf_file_path, N=100)
        messages_50 = read_jsf(real_jsf_file_path, N=50)
        
        assert len(messages_100) == 100
        assert len(messages_50) == 50
    
    def test_read_jsf_messages_are_jsf_message_type(self, real_jsf_file_path):
        """All returned messages should be JSFMessage instances."""
        messages = read_jsf(real_jsf_file_path, N=50)
        assert all(isinstance(msg, JSFMessage) for msg in messages)
    
    def test_read_jsf_file_not_found_raises(self):
        """read_jsf() should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_jsf(Path("nonexistent_file.jsf"))
    
    @pytest.mark.slow
    def test_read_jsf_no_limit_reads_all(self, real_jsf_file_path):
        """read_jsf() with N=None should read entire file."""
        messages = read_jsf(real_jsf_file_path, N=None)
        assert len(messages) > 100  # Assuming file has >100 messages


class TestReadJSFIterator:
    """Test streaming/iterator functionality."""
    
    def test_read_jsf_iter_yields_messages(self, real_jsf_file_path):
        """read_jsf_iter() should yield JSFMessage objects."""
        count = 0
        for msg in read_jsf_iter(real_jsf_file_path):
            assert isinstance(msg, JSFMessage)
            count += 1
            if count >= 50:
                break
        
        assert count == 50
    
    def test_read_jsf_iter_matches_read_jsf(self, real_jsf_file_path):
        """Messages from iterator should match bulk read."""
        bulk_messages = read_jsf(real_jsf_file_path, N=100)
        
        iter_messages = []
        for i, msg in enumerate(read_jsf_iter(real_jsf_file_path)):
            iter_messages.append(msg)
            if i >= 99:
                break
        
        # Should have same number of messages
        assert len(iter_messages) == len(bulk_messages)
        
        # Messages should be same type in same order
        for bulk, iter_msg in zip(bulk_messages, iter_messages):
            assert type(bulk).__name__ == type(iter_msg).__name__
    
    def test_read_jsf_iter_file_not_found_raises(self):
        """read_jsf_iter() should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            next(read_jsf_iter(Path("nonexistent_file.jsf")))
    
    def test_read_jsf_iter_memory_efficient(self, real_jsf_file_path):
        """Iterator should allow processing without loading all into memory."""
        # Just verify we can iterate and process without loading all
        processed_count = 0
        for msg in read_jsf_iter(real_jsf_file_path):
            processed_count += 1
            if processed_count >= 200:
                break  # Stop early
        
        assert processed_count == 200


# =============================================================================
# SECTION 2: Message Inspection & Filtering Tests
# =============================================================================

class TestGetMessagesByIndex:
    """Test message slicing/indexing."""
    
    def test_get_messages_by_index_range(self, real_jsf_file_path):
        """Should retrieve messages in specified index range."""
        messages = get_messages_by_index(real_jsf_file_path, start_idx=10, end_idx=20)
        assert len(messages) == 10
    
    @pytest.mark.slow
    def test_get_messages_by_index_from_start(self, real_jsf_file_path):
        """Should retrieve messages from start index to end of file."""
        messages = get_messages_by_index(real_jsf_file_path, start_idx=0, end_idx=None)
        assert len(messages) > 0
    
    def test_get_messages_by_index_single_message(self, real_jsf_file_path):
        """Should retrieve single message."""
        messages = get_messages_by_index(real_jsf_file_path, start_idx=0, end_idx=1)
        assert len(messages) == 1
    
    def test_get_messages_by_index_zero_range(self, real_jsf_file_path):
        """Should return empty list for zero-length range."""
        messages = get_messages_by_index(real_jsf_file_path, start_idx=5, end_idx=5)
        assert len(messages) == 0
    
    @pytest.mark.slow
    def test_get_messages_by_index_beyond_file(self, real_jsf_file_path):
        """Should return available messages when range exceeds file."""
        messages = get_messages_by_index(real_jsf_file_path, start_idx=0, end_idx=1000000)
        assert len(messages) > 0


class TestFilterMessagesByType:
    """Test message type filtering."""
    
    def test_filter_messages_by_type_returns_list(self, real_jsf_file_path):
        """Should return a list."""
        messages = read_jsf(real_jsf_file_path, N=100)
        filtered = filter_messages_by_type(messages, 'SonarDataMessage')
        assert isinstance(filtered, list)
    
    def test_filter_messages_by_type_sonar_data(self, real_jsf_file_path):
        """Should filter to only SonarDataMessage."""
        messages = read_jsf(real_jsf_file_path, N=100)
        filtered = filter_messages_by_type(messages, 'SonarDataMessage')
        assert all(type(msg).__name__ == 'SonarDataMessage' for msg in filtered)
    
    def test_filter_messages_by_type_empty_result(self, real_jsf_file_path):
        """Should return empty list if no matches."""
        messages = read_jsf(real_jsf_file_path, N=100)
        filtered = filter_messages_by_type(messages, 'NonexistentMessageType')
        assert filtered == []
    
    def test_filter_messages_by_type_subset(self, real_jsf_file_path):
        """Filtered list should be subset of original."""
        messages = read_jsf(real_jsf_file_path, N=100)
        filtered = filter_messages_by_type(messages, 'PitchRollDataMessage')
        assert len(filtered) <= len(messages)


class TestFilterMessagesByFrequency:
    """Test frequency filtering for sonar messages."""
    
    def test_filter_messages_by_frequency_returns_list(self, real_jsf_file_path):
        """Should return a list."""
        messages = read_jsf(real_jsf_file_path, N=100)
        sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
        
        if sonar_msgs:
            filtered = filter_messages_by_frequency(sonar_msgs, 1495000)
            assert isinstance(filtered, list)
    
    def test_filter_messages_by_frequency_all_same(self, real_jsf_file_path):
        """Filtered messages should all have requested frequency."""
        messages = read_jsf(real_jsf_file_path, N=100)
        sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
        
        if sonar_msgs:
            target_freq = int(sonar_msgs[0].transmit_freq_start)
            filtered = filter_messages_by_frequency(sonar_msgs, target_freq)
            
            assert all(
                int(msg.transmit_freq_start) == target_freq
                for msg in filtered
            )
    
    def test_filter_messages_by_frequency_empty_result(self, real_jsf_file_path):
        """Should return empty list for nonexistent frequency."""
        messages = read_jsf(real_jsf_file_path, N=100)
        sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
        
        if sonar_msgs:
            filtered = filter_messages_by_frequency(sonar_msgs, 999999999)
            assert filtered == []


# =============================================================================
# SECTION 3: Message Statistics Tests
# =============================================================================

class TestGetMessageStatistics:
    """Test message statistics computation."""
    
    def test_get_message_statistics_returns_dict(self, real_jsf_file_path):
        """Should return a dictionary."""
        messages = read_jsf(real_jsf_file_path, N=100)
        stats = get_message_statistics(messages)
        assert isinstance(stats, dict)
    
    def test_get_message_statistics_has_required_keys(self, real_jsf_file_path):
        """Should have 'total', 'by_type', 'frequencies' keys."""
        messages = read_jsf(real_jsf_file_path, N=100)
        stats = get_message_statistics(messages)
        
        assert 'total' in stats
        assert 'by_type' in stats
        assert 'frequencies' in stats
    
    def test_get_message_statistics_total_matches_count(self, real_jsf_file_path):
        """Total should match message count."""
        messages = read_jsf(real_jsf_file_path, N=100)
        stats = get_message_statistics(messages)
        
        assert stats['total'] == len(messages)
    
    def test_get_message_statistics_by_type_sum_matches_total(self, real_jsf_file_path):
        """Sum of by_type counts should equal total."""
        messages = read_jsf(real_jsf_file_path, N=100)
        stats = get_message_statistics(messages)
        
        type_sum = sum(stats['by_type'].values())
        assert type_sum == stats['total']
    
    def test_get_message_statistics_sonar_frequency_tracking(self, real_jsf_file_path):
        """Frequencies should track sonar message distribution."""
        messages = read_jsf(real_jsf_file_path, N=100)
        stats = get_message_statistics(messages)
        
        # Count sonar messages manually
        sonar_msgs = filter_messages_by_type(messages, 'SonarDataMessage')
        
        if sonar_msgs:
            # Frequencies should have entries
            assert len(stats['frequencies']) > 0
            
            # Sum of frequencies should equal sonar message count
            freq_sum = sum(stats['frequencies'].values())
            assert freq_sum == len(sonar_msgs)


# =============================================================================
# SECTION 4: JSON Export Tests
# =============================================================================

class TestExportMessagesToJSON:
    """Test JSON export functionality."""
    
    def test_export_messages_to_json_creates_file(self, tmp_path, real_jsf_file_path):
        """Should create a JSON file."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.json"
        
        export_messages_to_json(messages, output_file)
        
        assert output_file.exists()
    
    def test_export_messages_to_json_valid_json(self, tmp_path, real_jsf_file_path):
        """Exported file should be valid JSON."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.json"
        
        export_messages_to_json(messages, output_file)
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, list)
    
    def test_export_messages_to_json_preserves_count(self, tmp_path, real_jsf_file_path):
        """Exported JSON should have same number of messages."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.json"
        
        export_messages_to_json(messages, output_file)
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) == len(messages)
    
    def test_export_messages_to_json_has_message_type(self, tmp_path, real_jsf_file_path):
        """Each exported message should have _message_type field."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.json"
        
        export_messages_to_json(messages, output_file)
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert all('_message_type' in msg for msg in data)
    
    def test_export_messages_to_json_max_messages(self, tmp_path, real_jsf_file_path):
        """Should respect max_messages parameter."""
        messages = read_jsf(real_jsf_file_path, N=100)
        output_file = tmp_path / "messages.json"
        
        export_messages_to_json(messages, output_file, max_messages=25)
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 25
    
    def test_export_messages_to_json_pretty_format(self, tmp_path, real_jsf_file_path):
        """Pretty format should be readable."""
        messages = read_jsf(real_jsf_file_path, N=10)
        output_file = tmp_path / "messages_pretty.json"
        
        export_messages_to_json(messages, output_file, pretty=True)
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Pretty formatted JSON should have newlines and indentation
        assert '\n' in content
        assert '  ' in content


class TestExportMessagesByTypeToJSON:
    """Test type-organized JSON export."""
    
    def test_export_messages_by_type_creates_directory(self, tmp_path, real_jsf_file_path):
        """Should create output directory."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_dir = tmp_path / "by_type"
        
        export_messages_by_type_to_json(messages, output_dir)
        
        assert output_dir.exists()
    
    def test_export_messages_by_type_creates_files(self, tmp_path, real_jsf_file_path):
        """Should create one JSON file per message type."""
        messages = read_jsf(real_jsf_file_path, N=100)
        output_dir = tmp_path / "by_type"
        
        export_messages_by_type_to_json(messages, output_dir)
        
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) > 0
    
    def test_export_messages_by_type_file_naming(self, tmp_path, real_jsf_file_path):
        """JSON files should be named after message types."""
        messages = read_jsf(real_jsf_file_path, N=100)
        output_dir = tmp_path / "by_type"
        
        export_messages_by_type_to_json(messages, output_dir)
        
        json_files = list(output_dir.glob("*.json"))
        file_names = [f.stem for f in json_files]
        
        # Should have at least SonarDataMessage file
        assert any('Message' in name for name in file_names)
    
    def test_export_messages_by_type_content_separation(self, tmp_path, real_jsf_file_path):
        """Each file should contain only its message type."""
        messages = read_jsf(real_jsf_file_path, N=100)
        output_dir = tmp_path / "by_type"
        
        export_messages_by_type_to_json(messages, output_dir)
        
        json_files = list(output_dir.glob("*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # All messages in file should have same type
            msg_types = {msg['_message_type'] for msg in data}
            assert len(msg_types) == 1


# =============================================================================
# SECTION 5: Pickle Caching Tests
# =============================================================================

class TestPickleCaching:
    """Test pickle save/load functionality."""
    
    def test_save_messages_to_pickle_creates_file(self, tmp_path, real_jsf_file_path):
        """Should create a pickle file."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.pkl"
        
        save_messages_to_pickle(messages, output_file)
        
        assert output_file.exists()
    
    def test_load_messages_from_pickle_returns_list(self, tmp_path, real_jsf_file_path):
        """Should return a list of messages."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.pkl"
        
        save_messages_to_pickle(messages, output_file)
        loaded = load_messages_from_pickle(output_file)
        
        assert isinstance(loaded, list)
    
    def test_load_messages_from_pickle_preserves_count(self, tmp_path, real_jsf_file_path):
        """Loaded messages should match saved count."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.pkl"
        
        save_messages_to_pickle(messages, output_file)
        loaded = load_messages_from_pickle(output_file)
        
        assert len(loaded) == len(messages)
    
    def test_load_messages_from_pickle_preserves_types(self, tmp_path, real_jsf_file_path):
        """Loaded messages should have same types as saved."""
        messages = read_jsf(real_jsf_file_path, N=50)
        output_file = tmp_path / "messages.pkl"
        
        save_messages_to_pickle(messages, output_file)
        loaded = load_messages_from_pickle(output_file)
        
        for orig, loaded_msg in zip(messages, loaded):
            assert type(orig).__name__ == type(loaded_msg).__name__
    
    def test_load_messages_from_pickle_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing pickle."""
        with pytest.raises(FileNotFoundError):
            load_messages_from_pickle(tmp_path / "nonexistent.pkl")


    def test_save_and_load_roundtrip(self, tmp_path, real_jsf_file_path):
        """Save and load should preserve message data."""
        messages = read_jsf(real_jsf_file_path, N=30)
        output_file = tmp_path / "roundtrip.pkl"
        
        message_types = set([str(type(msg).__name__) for msg in messages])
        save_messages_to_pickle(messages, output_file)
        loaded = load_messages_from_pickle(output_file)
        
        # Check a few critical fields on first sonar message
        sonar_orig = filter_messages_by_type(messages, 'SonarDataMessage')
        sonar_loaded = filter_messages_by_type(loaded, 'SonarDataMessage')
        
        if sonar_orig and sonar_loaded:
            assert sonar_orig[0].ping_number == sonar_loaded[0].ping_number
            assert sonar_orig[0].transmit_freq_start == sonar_loaded[0].transmit_freq_start