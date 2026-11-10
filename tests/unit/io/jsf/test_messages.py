"""
Unit tests for JSF message parsing and serialization.

Tests real message data extracted from actual JSF survey files.
"""


import pytest

# Import your message classes (adjust imports to match your project)
from sidescanpy.io.jsf.messages import (
    JSFMessage,
    JSFMessageHeader,
    NavigationOffsetsMessage,
    NMEARMCMessage,
    NMEAZDAMessage,
    PitchRollDataMessage,
    PressureSensorReadingMessage,
    SonarDataMessage,
)
from tests.unit.io.jsf.conftest import (
    PR_FLAG_HEADING,
    PR_FLAG_PITCH,
    PR_FLAG_ROLL,
    _make_real_pitch_roll_msg,
)

# ============================================================================
# SONAR DATA MESSAGE TESTS
# ============================================================================

@pytest.mark.unit
class TestSonarDataMessage:
    """Tests for SonarDataMessage parsing."""

    def test_parse_sonar_data_header(self, sample_sonar_data_message_header_bytes):
        """Test that we can parse a real sonar data message header."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 17
        assert header.message_type == 80
        assert header.command_type == 2
        assert header.subsystem_num == 21
        assert header.channel == 0
        assert header.message_size == 9824

    def test_parse_sonar_data_message(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
    ):
        """Test parsing a complete sonar data message."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert isinstance(message, SonarDataMessage)
        assert message.ping_number == 69
        assert message.channel == 0
        assert message.N_samples == 4792

    def test_sonar_data_location_data(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
        sample_sonar_data_message_dict,
    ):
        """Test that latitude/longitude are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert message.latitude == pytest.approx(
            sample_sonar_data_message_dict['latitude'],
            rel=1e-6
        )
        assert message.longitude == pytest.approx(
            sample_sonar_data_message_dict['longitude'],
            rel=1e-6
        )

    def test_sonar_data_attitude_data(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
        sample_sonar_data_message_dict,
    ):
        """Test that heading/pitch/roll are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message: SonarDataMessage = SonarDataMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert message.heading_deg == pytest.approx(
            sample_sonar_data_message_dict['heading_deg'],
            rel=1e-3
        )
        assert message.pitch_deg == pytest.approx(
            sample_sonar_data_message_dict['pitch_deg'],
            rel=1e-3
        )
        assert message.roll_deg == pytest.approx(
            sample_sonar_data_message_dict['roll_deg'],
            rel=1e-3
        )

    def test_sonar_data_depth_altitude(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
        sample_sonar_data_message_dict,
    ):
        """Test that depth and altitude are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message: SonarDataMessage = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert message.depth_m == pytest.approx(
            sample_sonar_data_message_dict['depth_m'],
            rel=1e-2
        )
        assert message.altitude_m == pytest.approx(
            sample_sonar_data_message_dict['altitude_m'],
            rel=1e-2
        )

    def test_sonar_data_frequencies(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
        sample_sonar_data_message_dict,
    ):
        """Test that transmit frequencies are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message: SonarDataMessage = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert message.transmit_freq_start == pytest.approx(
            sample_sonar_data_message_dict['transmit_freq_start'],
            rel=1e-4
        )
        assert message.transmit_freq_end == pytest.approx(
            sample_sonar_data_message_dict['transmit_freq_end'],
            rel=1e-4
        )

    def test_sonar_data_sample_params(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
        sample_sonar_data_message_dict,
    ):
        """Test that sampling parameters are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message: SonarDataMessage = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        assert message.sample_freq == sample_sonar_data_message_dict['sample_freq']
        assert message.sample_interval == pytest.approx(
            sample_sonar_data_message_dict['sample_interval'],
            rel=1e-6
        )
        assert message.N_samples == sample_sonar_data_message_dict['N_samples']

    def test_sonar_data_starting_depth(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
    ):
        """Bytes 4-7 (uint32) are the window offset of the first sample, in samples."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message: SonarDataMessage = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)
        assert message.starting_depth_samples == 0
        data = bytearray(sample_sonar_data_message_bytes)
        data[4:8] = (120).to_bytes(4, "little")
        message = JSFMessage.from_bytes(header, bytes(data))
        assert message.starting_depth_samples == 120




# ============================================================================
# PITCH/ROLL MESSAGE — SPEC LAYOUT, SYNTHETIC VALUES
# ============================================================================

@pytest.mark.unit
class TestPitchRollSpecDecoding:
    """Message 2020 decoded from synthetic bytes laid out per the JSF spec."""

    def test_values_populate_declared_fields_when_flags_set(self):
        msg = _make_real_pitch_roll_msg(
            time_s=1_700_000_000, ms_in_second=250,
            pitch_deg=3.5, roll_deg=-2.25, heave_m=0.125, heading_deg=271.5,
        )
        assert msg.pitch_deg == pytest.approx(3.5, abs=180 / 32768)
        assert msg.roll_deg == pytest.approx(-2.25, abs=180 / 32768)
        assert msg.heave_m == pytest.approx(0.125)
        assert msg.heading_deg == pytest.approx(271.5)

    def test_field_is_none_when_flag_clear(self):
        msg = _make_real_pitch_roll_msg(
            time_s=1_700_000_000, ms_in_second=0,
            pitch_deg=1.0, roll_deg=1.0, heave_m=0.5, heading_deg=10.0,
            flags=PR_FLAG_PITCH | PR_FLAG_ROLL | PR_FLAG_HEADING,
        )
        assert msg.heave_m is None
        assert msg.pitch_deg == pytest.approx(1.0, abs=180 / 32768)

    def test_all_fields_none_when_no_flags(self):
        msg = _make_real_pitch_roll_msg(
            time_s=1_700_000_000, ms_in_second=0, pitch_deg=1.0, flags=0,
        )
        assert msg.pitch_deg is None
        assert msg.roll_deg is None
        assert msg.heave_m is None
        assert msg.heading_deg is None

    def test_timestamp_combines_seconds_and_ms_in_second(self):
        msg = _make_real_pitch_roll_msg(time_s=1_700_000_000, ms_in_second=987)
        assert msg.timestamp == pytest.approx(1_700_000_000.987, abs=1e-6)

    def test_timestamp_ns_is_exact_integer(self):
        msg = _make_real_pitch_roll_msg(time_s=1_700_000_000, ms_in_second=987)
        assert msg.timestamp_ns == 1_700_000_000 * 10**9 + 987 * 10**6


# ============================================================================
# PITCH/ROLL MESSAGE TESTS
# ============================================================================

@pytest.mark.unit
class TestPitchRollMessage:
    """Tests for PitchRollDataMessage parsing."""

    def test_parse_pitch_roll_header(self, sample_pitch_roll_message_header_bytes):
        """Test that we can parse a pitch/roll message header."""
        header = JSFMessageHeader.from_bytes(sample_pitch_roll_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 17
        assert header.message_type == 2020
        assert header.subsystem_num == 101
        assert header.channel == 1
        assert header.message_size == 44

    def test_parse_pitch_roll_message(
        self,
        sample_pitch_roll_message_header_bytes,
        sample_pitch_roll_message_bytes,
    ):
        """Test parsing a complete pitch/roll message."""
        header = JSFMessageHeader.from_bytes(sample_pitch_roll_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_pitch_roll_message_bytes)

        assert isinstance(message, PitchRollDataMessage)

    def test_pitch_roll_values(
        self,
        sample_pitch_roll_message_header_bytes,
        sample_pitch_roll_message_bytes,
        sample_pitch_roll_message_dict,
    ):
        """Test that pitch and roll values are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_pitch_roll_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_pitch_roll_message_bytes)

        assert message.pitch_deg == pytest.approx(
            sample_pitch_roll_message_dict['pitch'],
            rel=1e-6
        )
        assert message.roll_deg == pytest.approx(
            sample_pitch_roll_message_dict['roll'],
            rel=1e-6
        )
        assert message.heading_deg == pytest.approx(
            sample_pitch_roll_message_dict['heading'],
            rel=1e-6
        )

    def test_pitch_roll_timestamp(
        self,
        sample_pitch_roll_message_header_bytes,
        sample_pitch_roll_message_bytes,
        sample_pitch_roll_message_dict,
    ):
        """Test that timestamp is parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_pitch_roll_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_pitch_roll_message_bytes)

        assert message.timestamp == pytest.approx(
            sample_pitch_roll_message_dict['timestamp'],
            rel=1e-6
        )


# ============================================================================
# PRESSURE SENSOR MESSAGE TESTS
# ============================================================================

@pytest.mark.unit
class TestPressureSensorMessage:
    """Tests for PressureSensorReadingMessage parsing."""

    def test_parse_pressure_sensor_header(self, sample_pressure_sensor_message_header_bytes):
        """Test that we can parse a pressure sensor message header."""
        header = JSFMessageHeader.from_bytes(sample_pressure_sensor_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 17
        assert header.message_type == 2060
        assert header.subsystem_num == 101
        assert header.message_size == 76

    def test_parse_pressure_sensor_message(
        self,
        sample_pressure_sensor_message_header_bytes,
        sample_pressure_sensor_message_bytes,
    ):
        """Test parsing a complete pressure sensor message."""
        header = JSFMessageHeader.from_bytes(sample_pressure_sensor_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_pressure_sensor_message_bytes)

        assert isinstance(message, PressureSensorReadingMessage)

    def test_pressure_sensor_values(
        self,
        sample_pressure_sensor_message_header_bytes,
        sample_pressure_sensor_message_bytes,
        sample_pressure_sensor_message_dict,
    ):
        """Test that pressure and temperature are parsed correctly."""
        header = JSFMessageHeader.from_bytes(sample_pressure_sensor_message_header_bytes)
        message: PressureSensorReadingMessage = JSFMessage.from_bytes(header, sample_pressure_sensor_message_bytes)

        assert message.pressure_psi == pytest.approx(
            sample_pressure_sensor_message_dict['pressure_psi'],
            rel=1e-3
        )
        assert message.temperature_C == pytest.approx(
            sample_pressure_sensor_message_dict['temperature_C'],
            rel=1e-6
        )


# ============================================================================
# NAVIGATION OFFSETS MESSAGE TESTS
# ============================================================================

@pytest.mark.unit
class TestNavigationOffsetsMessage:
    """Tests for NavigationOffsetsMessage parsing."""

    def test_parse_navigation_offsets_header(self, sample_navigation_offsets_message_header_bytes):
        """Test that we can parse a navigation offsets message header."""
        header = JSFMessageHeader.from_bytes(sample_navigation_offsets_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 16
        assert header.message_type == 181
        assert header.message_size == 64

    def test_parse_navigation_offsets_message(
        self,
        sample_navigation_offsets_message_header_bytes,
        sample_navigation_offsets_message_bytes,
    ):
        """Test parsing a complete navigation offsets message."""
        header = JSFMessageHeader.from_bytes(sample_navigation_offsets_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_navigation_offsets_message_bytes)

        assert isinstance(message, NavigationOffsetsMessage)

    def test_navigation_offsets_all_zero(
        self,
        sample_navigation_offsets_message_header_bytes,
        sample_navigation_offsets_message_bytes,
    ):
        """Test that all offsets are zero (no offsets applied)."""
        header = JSFMessageHeader.from_bytes(sample_navigation_offsets_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_navigation_offsets_message_bytes)

        assert message.x_offset == 0.0
        assert message.y_offset == 0.0
        assert message.latitude_offset == 0.0
        assert message.longitude_offset == 0.0
        assert message.depth_offset == 0.0


# ============================================================================
# SystemInformationMessage
# ============================================================================

# ============================================================================
# FilePaddingMessage
# ============================================================================


# ============================================================================
# DopplerVelocityLogDataMessage
# ============================================================================


# ============================================================================
# SituationMessage
# ============================================================================


# ============================================================================
# SituationComprehensiveMessage
# ============================================================================

# ============================================================================
# CableCounterDataMessage
# ============================================================================

# ============================================================================
# KilometerOfPipeDataMessage
# ============================================================================

# ============================================================================
# ContainerTimeStampMessage
# ============================================================================

# ============================================================================
# UnknownMessage2040
# ============================================================================

# ============================================================================
# SonarMessageStatusMessage
# ============================================================================

# ============================================================================
# NMEA MESSAGE TESTS
# ============================================================================

@pytest.mark.unit
class TestNMEAMessages:
    """Tests for NMEA message parsing."""

    # TODO: test generic NMEA message





    def test_parse_nmea_zda_header(self, sample_nmea_zda_message_header_bytes):
        """Test that we can parse a NMEA ZDA message header."""
        header = JSFMessageHeader.from_bytes(sample_nmea_zda_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 16
        assert header.message_type == 2002
        assert header.subsystem_num == 100
        assert header.channel == 7
        assert header.message_size == 42

    def test_parse_nmea_zda_message(
        self,
        sample_nmea_zda_message_header_bytes,
        sample_nmea_zda_message_bytes,
    ):
        """Test parsing a complete NMEA ZDA message."""
        header = JSFMessageHeader.from_bytes(sample_nmea_zda_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_nmea_zda_message_bytes)

        assert isinstance(message, NMEAZDAMessage)
        assert "$GPZDA" in str(message._raw_message)

    def test_parse_nmea_rmc_header(self, sample_nmea_rmc_message_header_bytes):
        """Test that we can parse a NMEA RMC message header."""
        header = JSFMessageHeader.from_bytes(sample_nmea_rmc_message_header_bytes)

        assert header.marker == 5633
        assert header.protocol_version == 16
        assert header.message_type == 2002
        assert header.subsystem_num == 100
        assert header.channel == 7
        assert header.message_size == 75

    def test_parse_nmea_rmc_message(
        self,
        sample_nmea_rmc_message_header_bytes,
        sample_nmea_rmc_message_bytes,
    ):
        """Test parsing a complete NMEA RMC message."""
        header = JSFMessageHeader.from_bytes(sample_nmea_rmc_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_nmea_rmc_message_bytes)

        assert isinstance(message, NMEARMCMessage)
        assert "$GPRMC" in str(message._raw_message)





# ============================================================================
# ROUND-TRIP TESTS
# ============================================================================

@pytest.mark.unit
class TestMessageRoundTrip:
    """Test that messages can be serialized and deserialized correctly."""

    def test_sonar_data_round_trip(
        self,
        sample_sonar_data_message_header_bytes,
        sample_sonar_data_message_bytes,
    ):
        """Test that sonar data can be parsed and re-serialized."""
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_sonar_data_message_bytes)

        # Verify key fields are intact
        assert message.ping_number == 69
        assert message.channel == 0
        assert message.N_samples == 4792
        assert message.latitude == pytest.approx(41.561791666666664, rel=1e-6)

    def test_pitch_roll_round_trip(
        self,
        sample_pitch_roll_message_header_bytes,
        sample_pitch_roll_message_bytes,
    ):
        """Test that pitch/roll can be parsed and re-serialized."""
        header = JSFMessageHeader.from_bytes(sample_pitch_roll_message_header_bytes)
        message = JSFMessage.from_bytes(header, sample_pitch_roll_message_bytes)

        # Verify key fields are intact
        assert message.pitch_deg == pytest.approx(-6.70166015625, rel=1e-6)
        assert message.roll_deg == pytest.approx(1.2469482421875, rel=1e-6)


@pytest.mark.unit
class TestSonarDataLayback:
    """Bytes 228-231 layback (float m), 236-239 cable out (decimetres), bit 13."""

    @staticmethod
    def _patched(body: bytes, *, layback: float, cable_dm: int, flags_or: int) -> bytes:
        import struct

        b = bytearray(body)
        struct.pack_into("<f", b, 228, layback)
        struct.pack_into("<H", b, 236, cable_dm)
        flags = struct.unpack_from("<H", b, 30)[0] | flags_or
        struct.pack_into("<H", b, 30, flags)
        return bytes(b)

    def test_cable_out_decimetres_when_parsed(
        self, sample_sonar_data_message_header_bytes, sample_sonar_data_message_bytes,
    ):
        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        body = self._patched(sample_sonar_data_message_bytes, layback=125.9, cable_dm=1300, flags_or=1 << 13)
        message = SonarDataMessage.from_bytes(header, body)
        assert message.cable_out_m == pytest.approx(130.0)
        assert message.layback_m == pytest.approx(125.9, rel=1e-6)
        assert message.position_interpolated is True

    def test_position_interpolated_false_when_bit_13_clear(
        self, sample_sonar_data_message_header_bytes, sample_sonar_data_message_bytes,
    ):
        import struct

        header = JSFMessageHeader.from_bytes(sample_sonar_data_message_header_bytes)
        b = bytearray(sample_sonar_data_message_bytes)
        flags = struct.unpack_from("<H", b, 30)[0] & ~(1 << 13)
        struct.pack_into("<H", b, 30, flags)
        message = SonarDataMessage.from_bytes(header, bytes(b))
        assert message.position_interpolated is False
