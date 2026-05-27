"""
Fixtures for JSF Module

2026 rshomberg@uri.edu
"""

import pytest 

import struct

from sidescanpy.io.jsf.messages import (
    JSFMessageHeader,
    JSFMessage,
    SonarDataMessage,
    NavigationOffsetsMessage
)

import pytest

@pytest.fixture
def mock_jsf_header():
    """Generates a valid, minimal header for navigation offset messages."""
    # TODO add request to generate different header types
    return JSFMessageHeader(
        marker=0xAAAA,
        protocol_version=1,
        session_ID=5,
        message_type=181,  # Matches NavigationOffsetsMessage.MESSAGE_TYPE
        command_type=0,
        subsystem_num=1,
        channel=0,
        sequence_number=42,
        _reserved=0,
        message_size=64   # 16 floats * 4 bytes
    )

# No fixture of JSFMessage because abstract class

@pytest.fixture
def sample_sonar_data_message_bytes():
    # return struct.pack('<ffffffffffffffff', *[float(i) for i in range(1, 17)])
    raise NotImplementedError
    
@pytest.fixture
def sample_legacy_side_scan_data_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_navigation_offset_message_bytes():
    """16 sequential 32-bit floats packed as little-endian bytes."""
    # Packing numbers 1.0 through 16.0
    return struct.pack('<ffffffffffffffff', *[float(i) for i in range(1, 17)])

@pytest.fixture
def sample_system_information_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_file_padding_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_nmea_string_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_nmea_dbt_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_nmea_rmc_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_nmea_dpt_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_pitch_roll_data_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_pressure_sensor_reading_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_doppler_velocity_log_data_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_situation_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_situation_comprehensive_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_cable_counter_data_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_kilometer_of_pipe_data_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_container_timestamp_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_unknown_message_2040_message_bytes():
    raise NotImplementedError

@pytest.fixture
def sample_sonar_message_status_message_bytes():
    raise NotImplementedError