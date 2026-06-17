"""
Unit tests for sidescanpy.io.sdf.pages.

Validates SDFPageHeader parsing of the 512-byte typed header (excluding
the 4-byte marker, which is handled at the file/reader level).

Uses a synthetic header builder (conftest.make_sdf_header_bytes) to
construct headers with known values and verify each field path. Tests
target SDFPageHeader.from_bytes(bytes_512) directly.

2026 rshomberg@uri.edu
"""

import math
import struct

import numpy as np
import pytest

from sidescanpy.io.sdf.pages import (
    PAGE_HEADER_BYTES,
    PAGE_VERSIONS_SONAR,
    SDFPageHeader,
)
from tests.unit.io.sdf.conftest import (
    HEADER_TOTAL_BYTES,
    PAGE_HEADER_BYTES as CONFTEST_PAGE_HEADER_BYTES,
    make_sdf_header_bytes,
)


# ============================================================
# BASIC FIELDS (uint32, fields 1-26)
# ============================================================

@pytest.mark.unit
class TestBasicFields:
    def test_page_size_parsed(self):
        raw = make_sdf_header_bytes(page_size=27904)
        h = SDFPageHeader.from_bytes(raw)
        assert h.page_size == 27904

    def test_page_version_parsed(self):
        raw = make_sdf_header_bytes(page_version=3501)
        h = SDFPageHeader.from_bytes(raw)
        assert h.page_version == 3501

    def test_configuration_parsed(self):
        raw = make_sdf_header_bytes(configuration=0xABCD)
        h = SDFPageHeader.from_bytes(raw)
        assert h._configuration == 0xABCD

    def test_ping_number_parsed(self):
        raw = make_sdf_header_bytes(ping_number=42)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ping_number == 42

    def test_num_samples_parsed(self):
        raw = make_sdf_header_bytes(num_samples=3500)
        h = SDFPageHeader.from_bytes(raw)
        assert h.num_samples == 3500

    def test_beams_to_display_parsed(self):
        raw = make_sdf_header_bytes(beams_to_display=0xFF)
        h = SDFPageHeader.from_bytes(raw)
        assert h._beams_to_display == 0xFF

    def test_error_flags_parsed(self):
        raw = make_sdf_header_bytes(error_flags=0)
        h = SDFPageHeader.from_bytes(raw)
        assert h._error_flags == 0
        assert h.error_flags == 0

    def test_range_m_parsed(self):
        raw = make_sdf_header_bytes(range_m=75)
        h = SDFPageHeader.from_bytes(raw)
        assert h.range_m == 75


# ============================================================
# SPEED CONVERSIONS (cm/s → m/s)
# ============================================================

@pytest.mark.unit
class TestSpeedFields:
    def test_speed_cms_stored_raw(self):
        raw = make_sdf_header_bytes(speed_cms=300)
        h = SDFPageHeader.from_bytes(raw)
        assert h._speed_cms == 300

    def test_speed_ms_property(self):
        """speed_cms × 0.01 → speed_ms."""
        raw = make_sdf_header_bytes(speed_cms=200)
        h = SDFPageHeader.from_bytes(raw)
        assert h.speed_ms == pytest.approx(2.00, abs=1e-6)

    def test_sound_speed_cms_stored_raw(self):
        raw = make_sdf_header_bytes(sound_speed_cms=150000)
        h = SDFPageHeader.from_bytes(raw)
        assert h._sound_speed_cms == 150000

    def test_sound_speed_ms_property(self):
        """sound_speed_cms × 0.01 → sound_speed_ms."""
        raw = make_sdf_header_bytes(sound_speed_cms=150000)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sound_speed_ms == pytest.approx(1500.0, abs=1e-3)

    def test_zero_speed(self):
        raw = make_sdf_header_bytes(speed_cms=0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.speed_ms == 0.0


# ============================================================
# TIMESTAMP HANDLING (fields 18-24, post_init)
# ============================================================

@pytest.mark.unit
class TestTimestamp:
    def test_4digit_year_direct(self):
        """Year stored as 2026 → datetime year is 2026."""
        raw = make_sdf_header_bytes(year=2026, month=5, day=13,
                                   hour=14, minute=50, second=5,
                                   fseconds=0.0)
        h = SDFPageHeader.from_bytes(raw)
        from datetime import datetime, timezone
        expected = datetime(2026, 5, 13, 14, 50, 5,
                            tzinfo=timezone.utc).timestamp()
        assert h.timestamp == pytest.approx(expected, abs=1e-3)

    def test_year_offset_from_1900(self):
        """Year < 200 should be treated as offset from 1900."""
        raw = make_sdf_header_bytes(year=126, month=5, day=13,
                                   hour=14, minute=50, second=5,
                                   fseconds=0.0)
        h = SDFPageHeader.from_bytes(raw)
        from datetime import datetime, timezone
        expected = datetime(2026, 5, 13, 14, 50, 5,
                            tzinfo=timezone.utc).timestamp()
        assert h.timestamp == pytest.approx(expected, abs=1e-3)

    def test_year_normalized_in_post_init(self):
        """After parsing, _year should be 4-digit even if raw was offset."""
        raw = make_sdf_header_bytes(year=126)
        h = SDFPageHeader.from_bytes(raw)
        assert h._year == 2026

    def test_fseconds_added_to_timestamp(self):
        """fseconds (field 51) is the fractional component."""
        raw = make_sdf_header_bytes(year=2026, month=5, day=13,
                                   hour=14, minute=50, second=5,
                                   fseconds=0.123456)
        h = SDFPageHeader.from_bytes(raw)
        from datetime import datetime, timezone
        base = datetime(2026, 5, 13, 14, 50, 5,
                        tzinfo=timezone.utc).timestamp()
        assert h.timestamp == pytest.approx(base + 0.123456, abs=1e-5)

    def test_hsecond_stored_separately(self):
        """hsecond (field 24) is stored but timestamp uses fseconds."""
        raw = make_sdf_header_bytes(hsecond=50, fseconds=0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.hsecond == 50
        # timestamp uses fseconds, not hsecond
        from datetime import datetime, timezone
        base = datetime(2026, 5, 13, 14, 50, 5,
                        tzinfo=timezone.utc).timestamp()
        assert h.timestamp == pytest.approx(base + 0.5, abs=1e-5)

    def test_timestamp_ns(self):
        """timestamp_ns = int(timestamp * 1e9)."""
        raw = make_sdf_header_bytes()
        h = SDFPageHeader.from_bytes(raw)
        assert h.timestamp_ns == int(h.timestamp * 1e9)

    def test_invalid_date_logs_warning(self, caplog):
        """Invalid date components should log warning, not crash."""
        import logging
        raw = make_sdf_header_bytes(month=13, day=32)  # invalid
        with caplog.at_level(logging.WARNING):
            h = SDFPageHeader.from_bytes(raw)
        assert h.timestamp == 0.0
        assert h.timestamp_ns == 0
        assert any("invalid" in r.message.lower() or "date" in r.message.lower()
                   for r in caplog.records)


# ============================================================
# ATTITUDE FIELDS (float32, fields 28-30)
# ============================================================

@pytest.mark.unit
class TestAttitude:
    def test_heading_deg(self):
        raw = make_sdf_header_bytes(heading_deg=180.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.heading_deg == pytest.approx(180.5, abs=1e-4)

    def test_pitch_deg(self):
        raw = make_sdf_header_bytes(pitch_deg=-3.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pitch_deg == pytest.approx(-3.5, abs=1e-4)

    def test_roll_deg(self):
        raw = make_sdf_header_bytes(roll_deg=2.1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.roll_deg == pytest.approx(2.1, abs=1e-4)

    def test_zero_attitude(self):
        raw = make_sdf_header_bytes(pitch_deg=0.0, roll_deg=0.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pitch_deg == 0.0
        assert h.roll_deg == 0.0

    def test_negative_attitude(self):
        raw = make_sdf_header_bytes(pitch_deg=-15.5, roll_deg=-7.25)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pitch_deg == pytest.approx(-15.5, abs=1e-4)
        assert h.roll_deg == pytest.approx(-7.25, abs=1e-4)


# ============================================================
# DEPTH / ALTITUDE / TEMPERATURE (float32, fields 31-33)
# ============================================================

@pytest.mark.unit
class TestDepthAltitudeTemp:
    def test_depth_volts(self):
        """Spec stores depth in Volts (needs INI calibration to convert)."""
        raw = make_sdf_header_bytes(depth_volts=2.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.depth_volts == pytest.approx(2.5, abs=1e-4)

    def test_altitude_m(self):
        raw = make_sdf_header_bytes(altitude_m=4.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.altitude_m == pytest.approx(4.5, abs=1e-4)

    def test_temperature_c(self):
        raw = make_sdf_header_bytes(temperature_c=15.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.temperature_c == pytest.approx(15.5, abs=1e-4)

    def test_zero_depth_altitude(self):
        raw = make_sdf_header_bytes(depth_volts=0.0, altitude_m=0.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.depth_volts == 0.0
        assert h.altitude_m == 0.0


# ============================================================
# POSITION (float64 in radians, fields 37-40)
# ============================================================

@pytest.mark.unit
class TestPosition:
    def test_ship_latitude_radians_stored(self):
        raw = make_sdf_header_bytes(ship_latitude_rad=0.770159)
        h = SDFPageHeader.from_bytes(raw)
        assert h._ship_latitude_rad == pytest.approx(0.770159, abs=1e-9)

    def test_ship_latitude_deg_property(self):
        """0.770159 rad ≈ 44.13°N (site-b)."""
        raw = make_sdf_header_bytes(ship_latitude_rad=0.770159)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_latitude_deg == pytest.approx(44.1269, abs=0.001)

    def test_ship_longitude_deg_property(self):
        """-1.202396 rad ≈ -68.89°W."""
        raw = make_sdf_header_bytes(ship_longitude_rad=-1.202396)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_longitude_deg == pytest.approx(-68.8915, abs=0.001)

    def test_zero_lat_lon(self):
        raw = make_sdf_header_bytes(ship_latitude_rad=0.0,
                                   ship_longitude_rad=0.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_latitude_deg == 0.0
        assert h.ship_longitude_deg == 0.0

    def test_negative_latitude(self):
        """Southern hemisphere."""
        raw = make_sdf_header_bytes(ship_latitude_rad=-0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_latitude_deg == pytest.approx(-28.6479, abs=0.001)

    def test_pi_over_2_rad_is_90_deg(self):
        raw = make_sdf_header_bytes(ship_latitude_rad=math.pi / 2)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_latitude_deg == pytest.approx(90.0, abs=1e-6)

    def test_fish_lat_lon_separate_from_ship(self):
        raw = make_sdf_header_bytes(
            ship_latitude_rad=0.5,
            ship_longitude_rad=-1.0,
            fish_latitude_rad=0.6,
            fish_longitude_rad=-1.1,
        )
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_latitude_deg != h.fish_latitude_deg
        assert h.fish_latitude_deg == pytest.approx(math.degrees(0.6), abs=1e-6)
        assert h.fish_longitude_deg == pytest.approx(math.degrees(-1.1), abs=1e-6)

    def test_round_trip_via_struct(self):
        """Verify byte order: lo, hi (little-endian double)."""
        target_rad = 1.5
        raw = make_sdf_header_bytes(ship_latitude_rad=target_rad)
        h = SDFPageHeader.from_bytes(raw)
        assert h._ship_latitude_rad == pytest.approx(target_rad, abs=1e-9)


# ============================================================
# SHIP MOTION FIELDS (float32, fields 34-36)
# ============================================================

@pytest.mark.unit
class TestShipMotion:
    def test_ship_speed_ms(self):
        """ship_speed_ms is already in m/s; no conversion needed."""
        raw = make_sdf_header_bytes(ship_speed_ms=2.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_speed_ms == pytest.approx(2.5, abs=1e-4)

    def test_ship_heading_deg(self):
        raw = make_sdf_header_bytes(ship_heading_deg=270.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_heading_deg == pytest.approx(270.0, abs=1e-4)

    def test_ship_magnetic_variation(self):
        raw = make_sdf_header_bytes(ship_magnetic_variation=-15.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ship_magnetic_variation_deg == pytest.approx(-15.5, abs=1e-4)


# ============================================================
# AUXILIARY FIELDS (float32, fields 46-49)
# ============================================================

@pytest.mark.unit
class TestAuxFields:
    def test_aux_pitch_is_float(self):
        raw = make_sdf_header_bytes(aux_pitch=12.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.aux_pitch == pytest.approx(12.5, abs=1e-4)
        assert isinstance(h.aux_pitch, float)

    def test_aux_roll_is_float(self):
        raw = make_sdf_header_bytes(aux_roll=-3.25)
        h = SDFPageHeader.from_bytes(raw)
        assert h.aux_roll == pytest.approx(-3.25, abs=1e-4)

    def test_aux_depth_is_float(self):
        raw = make_sdf_header_bytes(aux_depth=15.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.aux_depth == pytest.approx(15.0, abs=1e-4)

    def test_aux_altitude_is_float(self):
        raw = make_sdf_header_bytes(aux_altitude=2.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.aux_altitude == pytest.approx(2.5, abs=1e-4)


# ============================================================
# NMEA FIX TIME (fields 25-27, 43-45)
# ============================================================

@pytest.mark.unit
class TestNMEAFixTime:
    def test_fix_time_hour_minute(self):
        raw = make_sdf_header_bytes(fix_time_hour=14, fix_time_minute=30)
        h = SDFPageHeader.from_bytes(raw)
        assert h._fix_time_hour == 14
        assert h._fix_time_minute == 30

    def test_fix_time_second_is_float(self):
        raw = make_sdf_header_bytes(fix_time_second=15.123)
        h = SDFPageHeader.from_bytes(raw)
        assert h.fix_time_second == pytest.approx(15.123, abs=1e-4)

    def test_fix_time_year_month_day(self):
        raw = make_sdf_header_bytes(
            fix_time_year=2026, fix_time_month=5, fix_time_day=13,
        )
        h = SDFPageHeader.from_bytes(raw)
        assert h.fix_time_year == 2026
        assert h.fix_time_month == 5
        assert h.fix_time_day == 13


# ============================================================
# SHIEVE / GPS OFFSETS (float32, fields 56-59)
# ============================================================

@pytest.mark.unit
class TestShieveAndGPS:
    def test_shieve_x_offset_is_float(self):
        raw = make_sdf_header_bytes(shieve_x_offset_m=2.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.shieve_x_offset_m == pytest.approx(2.5, abs=1e-4)

    def test_shieve_y_offset_is_float(self):
        raw = make_sdf_header_bytes(shieve_y_offset_m=-1.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.shieve_y_offset_m == pytest.approx(-1.0, abs=1e-4)

    def test_shieve_z_offset_is_float(self):
        raw = make_sdf_header_bytes(shieve_z_offset_m=0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.shieve_z_offset_m == pytest.approx(0.5, abs=1e-4)

    def test_gps_height_is_float(self):
        raw = make_sdf_header_bytes(gps_height_m=3.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.gps_height_m == pytest.approx(3.0, abs=1e-4)


# ============================================================
# FREQUENCY FIELDS (fields 53, 100)
# ============================================================

@pytest.mark.unit
class TestFrequencyFields:
    def test_sample_freq_hz(self):
        raw = make_sdf_header_bytes(sample_freq_hz=31250)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sample_freq_hz == 31250

    def test_sonar_freq_khz_stored(self):
        raw = make_sdf_header_bytes(sonar_freq_khz=455)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sonar_freq_khz == 455
        assert h._sonar_freq_khz == 455

    def test_sonar_freq_hz_property(self):
        """sonar_freq_khz × 1000 → sonar_freq_hz."""
        raw = make_sdf_header_bytes(sonar_freq_khz=900)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sonar_freq_hz == 900_000


# ============================================================
# LAYBACK FIELDS (fields 70-74)
# ============================================================

@pytest.mark.unit
class TestLaybackFields:
    def test_layback_method(self):
        raw = make_sdf_header_bytes(layback_method=1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.layback_method == 1

    def test_layback_fish_latitude_rad(self):
        raw = make_sdf_header_bytes(layback_fish_latitude_rad=0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h._layback_fish_latitude_rad == pytest.approx(0.5, abs=1e-9)

    def test_layback_fish_latitude_deg_property(self):
        raw = make_sdf_header_bytes(layback_fish_latitude_rad=0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.layback_fish_latitude_deg == pytest.approx(
            math.degrees(0.5), abs=1e-6
        )

    def test_layback_fish_longitude_deg_property(self):
        raw = make_sdf_header_bytes(layback_fish_longitude_rad=-1.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.layback_fish_longitude_deg == pytest.approx(
            math.degrees(-1.0), abs=1e-6
        )

    def test_fish_heading_offset_is_float(self):
        raw = make_sdf_header_bytes(fish_heading_offset_deg=2.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.fish_heading_offset_deg == pytest.approx(2.5, abs=1e-4)

    def test_pressure_sensor_offset_is_float(self):
        raw = make_sdf_header_bytes(pressure_sensor_offset_psi=14.7)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pressure_sensor_offset_psi == pytest.approx(14.7, abs=1e-4)


# ============================================================
# VERSION / SOFTWARE FIELDS (fields 75-79)
# ============================================================

@pytest.mark.unit
class TestVersionFields:
    def test_tpu_sw_version(self):
        # Version encoded as 0xVVNNMMDD
        raw = make_sdf_header_bytes(tpu_sw_version=0x06170A0F)
        h = SDFPageHeader.from_bytes(raw)
        assert h.tpu_sw_version == 0x06170A0F

    def test_capability_mask(self):
        raw = make_sdf_header_bytes(capability_mask=0xFF)
        h = SDFPageHeader.from_bytes(raw)
        assert h.capability_mask == 0xFF

    def test_tx_version(self):
        raw = make_sdf_header_bytes(tx_version=1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.tx_version == 1

    def test_num_samples_extra(self):
        raw = make_sdf_header_bytes(num_samples_extra=10)
        h = SDFPageHeader.from_bytes(raw)
        assert h.num_samples_extra == 10

    def test_post_processing_version(self):
        raw = make_sdf_header_bytes(post_processing_version=0x80000)
        h = SDFPageHeader.from_bytes(raw)
        assert h.post_processing_version == 0x80000


# ============================================================
# MOTION SENSOR + TIME REF COUNTS (uint16-in-uint32, fields 80-83)
# ============================================================

@pytest.mark.unit
class TestMotionAndRefCounts:
    def test_motion_sensor_type(self):
        """Spec: 0=TCM, 1=KMS-01, 2=POS MV. Stored in uint32 slot."""
        raw = make_sdf_header_bytes(motion_sensor_type=2)
        h = SDFPageHeader.from_bytes(raw)
        assert h.motion_sensor_type == 2

    def test_ping_time_ref_count(self):
        raw = make_sdf_header_bytes(ping_time_ref_count=35914791)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ping_time_ref_count == 35914791

    def test_ext_trig_time_ref_count(self):
        raw = make_sdf_header_bytes(ext_trig_time_ref_count=133160)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ext_trig_time_ref_count == 133160

    def test_one_pps_time_ref_count(self):
        raw = make_sdf_header_bytes(one_pps_time_ref_count=42)
        h = SDFPageHeader.from_bytes(raw)
        assert h.one_pps_time_ref_count == 42

    def test_time_ref_count_weight(self):
        """Weight in nanoseconds per LSB."""
        raw = make_sdf_header_bytes(time_ref_count_weight=1000)
        h = SDFPageHeader.from_bytes(raw)
        assert h.time_ref_count_weight == 1000


# ============================================================
# BATHYMETRY FIELDS (fields 85-87)
# ============================================================

@pytest.mark.unit
class TestBathymetryFields:
    def test_altitude_bathy_is_float(self):
        raw = make_sdf_header_bytes(altitude_bathy_m=5.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.altitude_bathy_m == pytest.approx(5.5, abs=1e-4)

    def test_ping_interval_s_is_float(self):
        """Field 86 is float32, NOT uint32."""
        raw = make_sdf_header_bytes(ping_interval_s=0.125)
        h = SDFPageHeader.from_bytes(raw)
        assert h.ping_interval_s == pytest.approx(0.125, abs=1e-4)
        assert isinstance(h.ping_interval_s, float)

    def test_sdf_extension_size(self):
        raw = make_sdf_header_bytes(sdf_extension_size=256)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sdf_extension_size == 256


# ============================================================
# SECONDS OF WEEK + SOUND SPEED (fields 88-89)
# ============================================================

@pytest.mark.unit
class TestSecondsOfWeek:
    def test_seconds_of_week_is_double(self):
        raw = make_sdf_header_bytes(seconds_of_week=345600.123456)
        h = SDFPageHeader.from_bytes(raw)
        assert h.seconds_of_week == pytest.approx(345600.123456, abs=1e-6)

    def test_seconds_of_week_zero(self):
        raw = make_sdf_header_bytes(seconds_of_week=0.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.seconds_of_week == 0.0

    def test_sound_speed_source(self):
        raw = make_sdf_header_bytes(sound_speed_source=1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.sound_speed_source == 1


# ============================================================
# PRESSURE SENSOR (fields 90-92)
# ============================================================

@pytest.mark.unit
class TestPressureSensor:
    def test_pressure_sensor_max(self):
        raw = make_sdf_header_bytes(pressure_sensor_max=300.0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pressure_sensor_max == pytest.approx(300.0, abs=1e-4)

    def test_pressure_sensor_voltage_min(self):
        raw = make_sdf_header_bytes(pressure_sensor_voltage_min=0.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pressure_sensor_voltage_min == pytest.approx(0.5, abs=1e-4)

    def test_pressure_sensor_voltage_max(self):
        raw = make_sdf_header_bytes(pressure_sensor_voltage_max=4.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.pressure_sensor_voltage_max == pytest.approx(4.5, abs=1e-4)


# ============================================================
# PROCESSED PING (fields 93-95)
# ============================================================

@pytest.mark.unit
class TestProcessedPing:
    # def test_processed_ping_number(self):
    #     raw = make_sdf_header_bytes(processed_ping_number=12345)
    #     h = SDFPageHeader.from_bytes(raw)
    #     assert h.processed_ping_number == 12345

    # def test_processed_ping_spacing_m(self):
    #     raw = make_sdf_header_bytes(processed_ping_spacing_m=0.05)
    #     h = SDFPageHeader.from_bytes(raw)
    #     assert h.processed_ping_spacing_m == pytest.approx(0.05, abs=1e-4)

    def test_temperature_ambient_c(self):
        raw = make_sdf_header_bytes(temperature_ambient_c=18.5)
        h = SDFPageHeader.from_bytes(raw)
        assert h.temperature_ambient_c == pytest.approx(18.5, abs=1e-4)


# ============================================================
# SATURATION DETECT THRESHOLD (signed int32, field 99)
# ============================================================

@pytest.mark.unit
class TestSaturationThreshold:
    def test_signed_minus_1(self):
        """-1 means 'off'. Must read as signed int32."""
        raw = make_sdf_header_bytes(saturation_detect_threshold=-1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.saturation_detect_threshold == -1

    def test_zero(self):
        raw = make_sdf_header_bytes(saturation_detect_threshold=0)
        h = SDFPageHeader.from_bytes(raw)
        assert h.saturation_detect_threshold == 0

    def test_positive_value(self):
        raw = make_sdf_header_bytes(saturation_detect_threshold=4)
        h = SDFPageHeader.from_bytes(raw)
        assert h.saturation_detect_threshold == 4

    def test_max_positive(self):
        """Max valid threshold = 7."""
        raw = make_sdf_header_bytes(saturation_detect_threshold=7)
        h = SDFPageHeader.from_bytes(raw)
        assert h.saturation_detect_threshold == 7


# ============================================================
# RESERVED FIELDS (24 uint32, fields 101-124)
# ============================================================

@pytest.mark.unit
class TestReservedFields:
    def test_reserved_default_all_zero(self):
        raw = make_sdf_header_bytes()
        h = SDFPageHeader.from_bytes(raw)
        assert h._reserved == tuple(0 for _ in range(26))

    def test_reserved_24_values(self):
        custom = tuple(range(100, 126))
        raw = make_sdf_header_bytes(reserved=custom)
        h = SDFPageHeader.from_bytes(raw)
        assert h._reserved == custom

    def test_reserved_length_is_24(self):
        raw = make_sdf_header_bytes()
        h = SDFPageHeader.from_bytes(raw)
        assert len(h._reserved) == 26

    def test_reserved_invalid_length_raises(self):
        """Builder rejects wrong-length reserved tuple."""
        with pytest.raises(ValueError, match="reserved"):
            make_sdf_header_bytes(reserved=tuple(range(20)))


# ============================================================
# ERROR FLAGS (field 7)
# ============================================================

@pytest.mark.unit
class TestErrorFlags:
    def test_no_errors(self):
        raw = make_sdf_header_bytes(error_flags=0)
        h = SDFPageHeader.from_bytes(raw)
        for bit in range(13):
            assert not h.has_error(bit)

    def test_invalid_speed_flag(self):
        raw = make_sdf_header_bytes(error_flags=0x1)
        h = SDFPageHeader.from_bytes(raw)
        assert h.has_error(0)
        assert not h.has_error(1)

    def test_compass_error_flag(self):
        raw = make_sdf_header_bytes(error_flags=0x80)  # bit 7
        h = SDFPageHeader.from_bytes(raw)
        assert h.has_error(7)

    def test_multiple_errors(self):
        # Bits 0, 7, 12 set = 0x1081
        raw = make_sdf_header_bytes(error_flags=0x1081)
        h = SDFPageHeader.from_bytes(raw)
        assert h.has_error(0)
        assert h.has_error(7)
        assert h.has_error(12)
        assert not h.has_error(1)


# ============================================================
# SONAR PAGE WRAPPING (SDFSonarPage)
# ============================================================

@pytest.mark.unit
class TestSonarPageWrapping:
    def test_constructor_takes_header(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes()
        h = SDFPageHeader.from_bytes(raw)
        page = SDFSonarPage(header=h)
        assert page.header is h

    def test_delegates_ping_number(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(ping_number=42)
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.ping_number == 42

    def test_delegates_num_samples(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(num_samples=2500)
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.num_samples == 2500

    def test_delegates_timestamp(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(year=2026, month=5, day=13,
                                   hour=14, minute=50, second=5,
                                   fseconds=0.0)
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        from datetime import datetime, timezone
        expected = datetime(2026, 5, 13, 14, 50, 5,
                            tzinfo=timezone.utc).timestamp()
        assert page.timestamp == pytest.approx(expected, abs=1e-3)

    def test_delegates_center_freq_hz(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(sonar_freq_khz=455)
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.center_freq_hz == 455_000

    def test_latitude_prefers_fish_when_set(self):
        """If fish_latitude is non-zero, prefer it over ship."""
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(
            ship_latitude_rad=0.7,
            fish_latitude_rad=0.6,
        )
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.latitude == pytest.approx(math.degrees(0.6), abs=1e-6)

    def test_latitude_falls_back_to_ship_when_fish_zero(self):
        """When fish_latitude is 0, fall back to ship lat."""
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(
            ship_latitude_rad=0.7,
            fish_latitude_rad=0.0,
        )
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.latitude == pytest.approx(math.degrees(0.7), abs=1e-6)

    def test_longitude_prefers_fish_when_set(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(
            ship_longitude_rad=-1.2,
            fish_longitude_rad=-1.3,
        )
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.longitude == pytest.approx(math.degrees(-1.3), abs=1e-6)

    def test_fathometer_reading_m(self):
        from sidescanpy.io.sdf.pages import SDFSonarPage
        raw = make_sdf_header_bytes(altitude_m=4.5)
        page = SDFSonarPage(header=SDFPageHeader.from_bytes(raw))
        assert page.fathometer_reading_m == pytest.approx(4.5, abs=1e-4)


# ============================================================
# ROUND-TRIP (to_bytes ↔ from_bytes)
# ============================================================

@pytest.mark.unit
class TestRoundTrip:
    def test_byte_count_is_512(self):
        raw = make_sdf_header_bytes()
        assert len(raw) == 512
        assert SDFPageHeader.STRUCT.size == PAGE_HEADER_BYTES

    def test_full_round_trip_preserves_uint32(self):
        raw = make_sdf_header_bytes(ping_number=99999, num_samples=1234)
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        h2 = SDFPageHeader.from_bytes(round_trip)
        assert h2.ping_number == h.ping_number
        assert h2.num_samples == h.num_samples

    def test_full_round_trip_preserves_float32(self):
        raw = make_sdf_header_bytes(heading_deg=180.5, pitch_deg=-3.5)
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        h2 = SDFPageHeader.from_bytes(round_trip)
        assert h2.heading_deg == pytest.approx(180.5, abs=1e-4)
        assert h2.pitch_deg == pytest.approx(-3.5, abs=1e-4)

    def test_full_round_trip_preserves_float64(self):
        raw = make_sdf_header_bytes(ship_latitude_rad=0.770159)
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        h2 = SDFPageHeader.from_bytes(round_trip)
        assert h2._ship_latitude_rad == pytest.approx(0.770159, abs=1e-9)

    def test_full_round_trip_preserves_int32(self):
        """Negative saturation_detect_threshold round-trips."""
        raw = make_sdf_header_bytes(saturation_detect_threshold=-1)
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        h2 = SDFPageHeader.from_bytes(round_trip)
        assert h2.saturation_detect_threshold == -1

    def test_full_round_trip_preserves_reserved(self):
        custom = tuple(range(100, 126))
        raw = make_sdf_header_bytes(reserved=custom)
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        h2 = SDFPageHeader.from_bytes(round_trip)
        assert h2._reserved == custom

    def test_round_trip_preserves_bytes_exactly(self):
        """Byte-level: parsing then reserializing produces identical bytes."""
        raw = make_sdf_header_bytes(
            ping_number=42, heading_deg=180.5,
            ship_latitude_rad=0.5, saturation_detect_threshold=-1,
        )
        h = SDFPageHeader.from_bytes(raw)
        round_trip = h.to_bytes()
        # Note: timestamp/timestamp_ns are derived; they're not in to_bytes output
        # So we compare only the parseable parts
        assert round_trip == raw


# ============================================================
# EDGE CASES
# ============================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_short_input_raises(self):
        raw = make_sdf_header_bytes()[:200]
        with pytest.raises(ValueError, match="too short|bytes"):
            SDFPageHeader.from_bytes(raw)

    def test_extra_bytes_ignored(self):
        """Passing more than 512 bytes should still parse correctly."""
        raw = make_sdf_header_bytes() + b"\x00" * 100
        h = SDFPageHeader.from_bytes(raw)
        assert h.page_size == 27904  # default

    def test_struct_size_matches_constant(self):
        """STRUCT byte count = 512."""
        assert SDFPageHeader.STRUCT.size == 512
        assert SDFPageHeader.STRUCT.size == CONFTEST_PAGE_HEADER_BYTES
        assert SDFPageHeader.STRUCT.size == PAGE_HEADER_BYTES