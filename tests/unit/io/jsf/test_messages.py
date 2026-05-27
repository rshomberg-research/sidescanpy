"""
Unit tests for JSF Messages

2026 rshomberg@uri.edu
"""

import pytest 

from sidescanpy.io.jsf.messages import (
    JSFMessageHeader,
    JSFMessage,
    SonarDataMessage,
    NavigationOffsetsMessage,
    SystemInformationMessage,
    FilePaddingMessage,
    NMEAStringMessage,
    NMEADBTMessage,
    NMEARMCMessage,
    NMEADPTMessage,
    PitchRollDataMessage,
    PressureSensorReadingMessage,
    DopplerVelocityLogDataMessage,
    SituationMessage,
    SituationComprehensiveMessage,
    CableCounterDataMessage,
    KilometerOfPipeDataMessage,
    ContainerTimeStampMessage,
    UnknownMessage2040,
    SonarMessageStatusMessage
)


@pytest.mark.unit 
class TestJSFMessageHeader:
    """Test JSF Message header"""

        
    def test_jsf_header_round_trip(self):
        # 1. Create a sample instance
        header = JSFMessageHeader(
            marker=0xAAAA, protocol_version=1, session_ID=42, 
            message_type=5, command_type=2, subsystem_num=1, 
            channel=0, sequence_number=100, _reserved=0, message_size=1024
        )
        
        # 2. Serialize to bytes
        binary_data = header.to_bytes()
        assert len(binary_data) == 16
        
        # 3. Deserialize back to object
        parsed_header = JSFMessageHeader.from_bytes(binary_data)
        
        # 4. Verify match
        assert header == parsed_header


@pytest.mark.unit
class TestJSFMessageBody:
    """Test JSF Message Body"""

    def test_cannot_be_initiated(self, mock_jsf_header):
        """Body abstract, should not be initiated"""
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class JSFMessage without an implementation for abstract methods '_parse_body', 'to_bytes'"):
            msg = JSFMessage(mock_jsf_header) # pyright: ignore[reportCallIssue, reportAbstractUsage]


    def test_unsupported_type_raises_error(self, mock_jsf_header):
        """Ensures factory fails gracefully on unknown message types."""
        mock_jsf_header.message_type = 999  # Fake/unregistered type
        
        with pytest.raises(ValueError, match="Unsupported message type: 999"):
            JSFMessage.from_bytes(mock_jsf_header, b'\x00' * 64)


@pytest.mark.unit 
class TestSonarDataMessage:
    """Tests for JSF Sonar Data Message"""

    def test_parsing(self, mock_jsf_header, sample_sonar_data_message_bytes):
        msg = SonarDataMessage.from_bytes(mock_jsf_header, sample_sonar_data_message_bytes)
        assert isinstance(msg, SonarDataMessage)

    def test_global_factory_routing(self, mock_jsf_header, sample_sonar_data_message_bytes):
        msg = JSFMessage.from_bytes(mock_jsf_header, sample_sonar_data_message_bytes)

        assert isinstance(msg, SonarDataMessage)
        assert msg.MESSAGE_TYPE == 80

    def test_serialization_round_trip(self, mock_jsf_header, sample_sonar_data_message_bytes):
        msg = SonarDataMessage.from_bytes(mock_jsf_header, sample_sonar_data_message_bytes)
        msg_bytes = msg.to_bytes()

        assert msg_bytes == sample_sonar_data_message_bytes

@pytest.mark.unit
class TestNavigationOffsetsMessage:
    """Tests for JSF Navigation Offsets message"""

    def test_parsing(self, mock_jsf_header, sample_navigation_offset_message_bytes):
        """Verifies that bytes are translated accurately into correct subclass fields."""
        msg = NavigationOffsetsMessage.from_bytes(mock_jsf_header, sample_navigation_offset_message_bytes)
        
        assert isinstance(msg, NavigationOffsetsMessage)
        assert msg._header == mock_jsf_header
        assert msg.x_offset == 1.0
        assert msg.y_offset == 2.0
        assert msg.reserved_3 == 16.0

    def test_global_factory_routing(self, mock_jsf_header, sample_navigation_offset_message_bytes):
        """Verifies the base class factory routes message type 181 to the right subclass."""
        # Call the abstract base class directly
        msg = JSFMessage.from_bytes(mock_jsf_header, sample_navigation_offset_message_bytes)
        
        # Assert polymorphic lookup worked flawlessly
        assert isinstance(msg, NavigationOffsetsMessage)
        assert msg.MESSAGE_TYPE == 181

    def test_serialization_round_trip(self, mock_jsf_header, sample_navigation_offset_message_bytes):
        """Ensures that serialization to_bytes produces identical inputs."""
        msg = NavigationOffsetsMessage.from_bytes(mock_jsf_header, sample_navigation_offset_message_bytes)
        
        serialized_bytes = msg.to_bytes()
        
        assert serialized_bytes == sample_navigation_offset_message_bytes
        assert len(serialized_bytes) == 64
