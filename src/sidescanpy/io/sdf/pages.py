"""
Klein SDF page types and parser.

A Klein SDF file is a sequence of pages. Each page has:
- A 4-byte sync marker (always 0xFFFFFFFF)
- A 512-byte typed header (described by SDFPageHeader)
- Variable-length data section (sonar samples for sonar page versions)

The header is a binary structure of 128 uint32-equivalent slots.
Most fields are uint32 or float32 (4 bytes); seven fields are
float64 (8 bytes, taking 2 slots). All values are little-endian.

Spec source: Klein SDF/SDFX Data Page Definitions Specification.

Conventions encoded here:

- Latitude/longitude stored as float64 in radians; converted to
  degrees via @property
- Heading/pitch/roll stored as float32 in degrees; no conversion
- speedFish stored as cm/s; converted to m/s via @property
- speedSound stored as cm/s; converted to m/s via @property
- sonarFreq stored in kHz; converted to Hz via @property
- Year normally 4-digit; older firmware uses offset-from-1900,
  auto-detected via < 200 check
- Time fractional component (`fseconds` field 51) preferred over
  hSecond field 24 when both are valid

2026 rshomberg@uri.edu
"""

import logging
import math
import struct
from dataclasses import astuple, dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

MARKER = 0xFFFFFFFF
MARKER_BYTES = 4
PAGE_HEADER_BYTES = 512                          # excluding marker
HEADER_TOTAL_BYTES = MARKER_BYTES + PAGE_HEADER_BYTES   # 516

# Page versions for sonar data (port + starboard sample arrays in body)
PAGE_VERSIONS_SONAR = frozenset({3501, 3502})

# Page version → towfish description
PAGE_VERSIONS_TOWFISH_TYPE = {
    3000: "System 3000",
    3001: "System 3000",
    3500: "UUV-3500 (obsolete)",
    3501: "UUV-3500 Low Frequency",
    3502: "UUV-3500 High Frequency",
    # TODO: complete from spec
}


# ============================================================
# HEADER DATACLASS
# ============================================================

@dataclass
class SDFPageHeader:
    """Klein SDF page header (512 bytes after the 4-byte marker).

    All fields are little-endian. The dataclass holds the raw spec values;
    transformed (unit-converted) versions are exposed via @property.

    Field numbers in comments correspond to the Klein spec table
    (1-100 named, 101-126 reserved). Fields prefixed with underscore
    are internal raw forms with corresponding public properties.

    Note on uint16 fields (80-83): the spec says U16 but Klein stores
    them in 32-bit slots with values fitting in the lower 16 bits.
    They're parsed as uint32 here for layout purity.
    """

    STRUCT: ClassVar[struct.Struct] = struct.Struct(
        '<'
        '26I'      # 1-26: uint32 (page_size, version, config, ping_num, num_samples,
                   #   beams_to_display, error_flags, range_m, speed_cms, sound_speed_cms,
                   #   res_mode, tx_waveform, responder_div, responder_freq, manual_speed_sw,
                   #   despeckle_sw, speed_filter_sw, year, month, day, hour, minute, second,
                   #   frac_second, fix_time_hour, fix_time_minute)
        'f'        # 27: fix_time_second
        '6f'       # 28-33: heading, pitch, roll, depth_V, altitude, temperature
        'f'        # 34: ship_speed_ms
        'f'        # 35: ship_heading_deg
        'f'        # 36: ship_magnetic_variation
        '4d'       # 37-40: ship_lat, ship_lon, fish_lat, fish_lon (radians, doubles)
        '5I'       # 41-45: tvg_page, header_size, fix_time_year, fix_time_month, fix_time_day
        '4f'       # 46-49: aux_pitch, aux_roll, aux_depth, aux_altitude
        '2f'       # 50-51: cable_out_m, fseconds
        '4I'       # 52-55: altimeter, sample_freq_hz, depressor_type, cable_type
        '4f'       # 56-59: shieve_x, shieve_y, shieve_z, gps_height
        '8I'       # 60-67: raw_data_config, header_3_ext_size,
                   #   sbp_tx_waveform, sbp_preamp_gain, sbp_data_raw,
                   #   sbp_samples, sbp_sample_freq, sbp_tx_waveform_version
        'f'        # 68: wing_angle
        '2I'       # 69-70: emergency_switch_state, layback_method
        '2d'       # 71-72: layback_fish_lat, layback_fish_lon (doubles)
        '2f'       # 73-74: fish_heading_offset, pressure_sensor_offset
        '5I'       # 75-79: tpu_sw_version, capability_mask, tx_version,
                   #   num_samples_extra, post_processing_version
        '4I'       # 80-83: motion_sensor_type, ping_time_ref, ext_trig_ref, one_pps_ref
                   #   (spec says U16, stored in U32 slots)
        'I'        # 84: time_ref_count_weight
        '2f'       # 85-86: altitude_bathy, ping_interval_s
        'I'        # 87: sdf_extension_size
        'd'        # 88: seconds_of_week (double)
        'I'        # 89: sound_speed_source
        '3f'       # 90-92: pressure_sensor_max, pressure_sensor_voltage_min,
                   #   pressure_sensor_voltage_max
        # 'I'        # 93: processed_ping_number
        # '2f'       # 94-95: processed_ping_spacing, temperature_ambient
        # 'i'        # 99: saturation_detect_threshold (signed int32)
        #            # NOTE: spec skips fields 96, 97, 98 — they don't exist
        #            #   as separate fields. Saturation comes right after temperature.
        # 'I'        # 100: sonar_freq_khz
        # '24I'      # 101-124: 24 reserved uint32 fields
        #            # NOTE: spec text says "26 reserved" but byte counting
        #            #   gives 24 to make the header exactly 512 bytes
        'f'        # 95: temperature_ambient (3500 layout — fields 93, 94 not present)
        'i'        # 99: saturation_detect_threshold
        'I'        # 100: sonar_freq_khz
        '26I'      # reserved
    )

    # Fields 1-26 (uint32)
    _page_size: int                  # 1: total bytes in page (excludes marker)
    _page_version: int               # 2: data page structure / system type
    _configuration: int              # 3: bit field of populated channels
    ping_number: int                 # 4: increments per sonar ping
    num_samples: int                 # 5: count of samples per channel
    _beams_to_display: int           # 6: bit field for sidescan channel selection
    _error_flags: int                # 7: bit field of error states
    range_m: int                     # 8: range in meters
    _speed_cms: int                  # 9: towfish speed (cm/s)
    _sound_speed_cms: int            # 10: speed of sound (cm/s)
    _res_mode: int                   # 11: resolution mode (system-specific)
    _tx_waveform: int                # 12: transmit waveform number
    _responder_divisor: int          # 13: responder divisor
    _responder_frequency: int        # 14: responder frequency enum
    _manual_speed_switch: int        # 15: 0=sensor, 1=manual master value
    _despeckle_switch: int           # 16: 0=off, 1=low, 2=med, 3=high
    _speed_filter_switch: int        # 17: bit 0 = filter on/off
    _year: int                       # 18: TPU calendar year (4-digit, sometimes offset-1900)
    _month: int                      # 19: 1-12
    _day: int                        # 20: 1-31
    _hour: int                       # 21: 0-23
    _minute: int                     # 22: 0-59
    _second: int                     # 23: 0-59
    _hsecond: int                    # 24: hundredths of second (1-99)
    _fix_time_hour: int              # 25: NMEA fix hour
    _fix_time_minute: int            # 26: NMEA fix minute

    # Fields 27 (float32)
    _fix_time_second: float          # 27: NMEA fix second

    # Fields 28-33 (float32)
    heading_deg: float               # 28: towfish compass heading
    pitch_deg: float                 # 29: pitch from compass
    roll_deg: float                  # 30: roll from compass
    depth_volts: float               # 31: depth (Volts; needs conversion via INI calibration)
    altitude_m: float                # 32: altitude from towfish (meters)
    temperature_c: float             # 33: towfish temperature (degC)

    # Fields 34-36 (float32)
    _ship_speed_ms: float            # 34: NMEA speed (m/s)
    _ship_heading_deg: float         # 35: NMEA Course Over Ground (degrees)
    _ship_magnetic_variation: float  # 36: NMEA magnetic variation

    # Fields 37-40 (float64, radians)
    _ship_latitude_rad: float        # 37: ship lat from NMEA
    _ship_longitude_rad: float       # 38: ship lon from NMEA
    _fish_latitude_rad: float        # 39: fish lat (computed)
    _fish_longitude_rad: float       # 40: fish lon (computed)

    # Fields 41-45 (uint32)
    _tvg_page: int                   # 41: TVG page number (system-specific)
    _header_size: int                # 42: number of bytes in header
    fix_time_year: int               # 43: NMEA fix year
    fix_time_month: int              # 44: NMEA fix month
    fix_time_day: int                # 45: NMEA fix day

    # Fields 46-49 (float32, "variable units")
    aux_pitch: float                 # 46: auxiliary sensor pitch
    aux_roll: float                  # 47: auxiliary sensor roll
    aux_depth: float                 # 48: auxiliary sensor depth
    aux_altitude: float              # 49: auxiliary sensor altitude

    # Fields 50-51 (float32)
    cable_out_m: float               # 50: cable out (meters)
    _fseconds: float                 # 51: TPU fractional seconds (preferred over hsecond)

    # Fields 52-55 (uint32)
    altimeter_on: int                # 52: 0=off, 1=on
    sample_freq_hz: int              # 53: sample frequency (Hz)
    depressor_type: int              # 54: towfish wing type enum
    cable_type: int                  # 55: towfish cable type enum

    # Fields 56-59 (float32, meters)
    shieve_x_offset_m: float         # 56: X winch offset from datum
    shieve_y_offset_m: float         # 57: Y winch offset from datum
    shieve_z_offset_m: float         # 58: Z winch offset from datum
    gps_height_m: float              # 59: vertical GPS offset from datum

    # Fields 60-67 (uint32)
    _raw_data_config: int            # 60: raw data channel bit field
    _header_3_extension_size: int    # 61: ext size, must be 256
    sbp_tx_waveform: int             # 62: Sub Bottom Profiler tx waveform
    sbp_preamp_gain: int             # 63: 0=low, 1=high
    sbp_data_raw: int                # 64: 0=processed, 1=raw
    sbp_samples: int                 # 65: SBP sample count (may differ from sidescan)
    sbp_sample_frequency: int        # 66: SBP sample frequency
    sbp_tx_waveform_version: int     # 67: SBP tx waveform version

    # Field 68 (float32)
    wing_angle_deg: float            # 68: actuated wing angle in degrees

    # Fields 69-70 (uint32)
    emergency_switch_state: int      # 69: 0=off, 1=on
    layback_method: int              # 70: 0=Pythagorean, 1=hull mount

    # Fields 71-72 (float64, radians)
    _layback_fish_latitude_rad: float    # 71: layback-computed fish lat
    _layback_fish_longitude_rad: float   # 72: layback-computed fish lon

    # Fields 73-74 (float32)
    fish_heading_offset_deg: float   # 73: magnetic heading offset (degrees)
    pressure_sensor_offset_psi: float    # 74: pressure sensor offset (PSI)

    # Fields 75-79 (uint32)
    tpu_sw_version: int              # 75: 0xVVNNMMDD (major, minor, month, day)
    capability_mask: int             # 76: bit mask of system capabilities
    tx_version: int                  # 77: 0=5000, 1=5000 V2
    num_samples_extra: int           # 78: extra samples for chirp Tx
    post_processing_version: int     # 79: post-processing software version + bit mask

    # Fields 80-83 (uint32; spec says U16, fits in low 16 bits)
    motion_sensor_type: int          # 80: 0=TCM compass, 1=KMS-01, 2=POS MV
    ping_time_ref_count: int         # 81: TPU counter at ping trigger
    ext_trig_time_ref_count: int     # 82: TPU counter at external trigger
    one_pps_time_ref_count: int      # 83: TPU counter at 1PPS signal

    # Field 84 (uint32)
    time_ref_count_weight: int       # 84: 1 LSB of TPU counter in nanoseconds

    # Fields 85-86 (float32)
    altitude_bathy_m: float          # 85: altitude used for bathymetric solution
    ping_interval_s: float           # 86: ping interval in seconds

    # Field 87 (uint32)
    sdf_extension_size: int          # 87: SDF extension area size in bytes (0 = none)

    # Field 88 (float64)
    seconds_of_week: float           # 88: seconds-of-week ping trigger time

    # Field 89 (uint32)
    sound_speed_source: int          # 89: 0=manual, 1=array sensor

    # Fields 90-92 (float32)
    pressure_sensor_max: float       # 90: max pressure reading
    pressure_sensor_voltage_min: float    # 91: min pressure voltage
    pressure_sensor_voltage_max: float    # 92: max pressure voltage

    # Field 93 (uint32)
    # processed_ping_number: int       # 93: increments per processed ping

    # Fields 94-95 (float32)
    # processed_ping_spacing_m: float  # 94: along-track spacing (meters)
    temperature_ambient_c: float     # 95: towfish internal temperature (degC)

    # Field 99 (int32 — signed!)
    saturation_detect_threshold: int # 99: -1=off, 0-7=12%-88% steps

    # Field 100 (uint32)
    _sonar_freq_khz: int             # 100: operating frequency (kHz)

    # Reserved fields (101-124)
    # 24 uint32 fields kept as a tuple for future-proofing.
    # If needed, individual access can be added.
    _reserved: tuple[int, ...] = field(default_factory=tuple)

    # Derived fields populated in __post_init__
    timestamp: float = field(init=False, default=0.0)
    timestamp_ns: int = field(init=False, default=0)

    # ============================================================
    # PARSING
    # ============================================================

    @classmethod
    def from_bytes(cls, data: bytes) -> "SDFPageHeader":
        """Parse 512 bytes of header data into an SDFPageHeader.

        The 4-byte marker is NOT included; caller should validate and
        consume it separately.
        """
        if len(data) < cls.STRUCT.size:
            raise ValueError(
                f"Input too short: got {len(data)} bytes, "
                f"need {cls.STRUCT.size}"
            )

        unpacked = cls.STRUCT.unpack_from(data, 0)
        # Last 24 fields go into _reserved tuple
        n_named = len(unpacked) - 26
        named_args = unpacked[:n_named]
        reserved_args = tuple(unpacked[n_named:])
        return cls(*named_args, _reserved=reserved_args)

    def to_bytes(self) -> bytes:
        """Pack the dataclass back into 512 bytes."""
        # Flatten _reserved tuple into the args
        all_fields = astuple(self)
        # Reserved is at index -3 (followed by timestamp and timestamp_ns)
        # Skip the derived fields; they're not in STRUCT
        named_fields = list(all_fields[:-3])
        reserved = all_fields[-3]
        return self.STRUCT.pack(*named_fields, *reserved)

    # ============================================================
    # POST-INIT (timestamp computation, year normalization)
    # ============================================================

    def __post_init__(self) -> None:
        # Year handling: older firmware stores year as offset-from-1900
        if self._year < 200:
            self._year += 1900

        # Build timestamp from date fields + fseconds (preferred over hsecond)
        try:
            dt = datetime(
                self._year, self._month, self._day,
                self._hour, self._minute, self._second,
                tzinfo=UTC,
            )
            self.timestamp = dt.timestamp() + self._fseconds
            self.timestamp_ns = int(self.timestamp * 1e9)
        except (ValueError, OverflowError) as e:
            logger.warning(
                f"Invalid date in SDF header: "
                f"{self._year}-{self._month}-{self._day} "
                f"{self._hour}:{self._minute}:{self._second}: {e}"
            )
            self.timestamp = 0.0
            self.timestamp_ns = 0

    # ============================================================
    # PROPERTIES — TRANSFORMED VALUES
    # ============================================================

    @property
    def page_size(self) -> int:
        """Page size in bytes (excludes the 4-byte marker)."""
        return self._page_size

    @property
    def page_version(self) -> int:
        return self._page_version

    @property
    def speed_ms(self) -> float:
        """Towfish speed in m/s (converted from cm/s)."""
        return self._speed_cms * 0.01

    @property
    def sound_speed_ms(self) -> float:
        """Speed of sound at transducer in m/s (converted from cm/s)."""
        return self._sound_speed_cms * 0.01

    @property
    def ship_speed_ms(self) -> float:
        """Ship speed from NMEA in m/s (already in m/s in spec)."""
        return self._ship_speed_ms

    @property
    def ship_heading_deg(self) -> float:
        """Ship Course Over Ground from NMEA, degrees."""
        return self._ship_heading_deg

    @property
    def ship_magnetic_variation_deg(self) -> float:
        """Magnetic variation from NMEA $--RMC, degrees."""
        return self._ship_magnetic_variation

    @property
    def ship_latitude_deg(self) -> float:
        """Ship latitude in degrees (converted from radians)."""
        return math.degrees(self._ship_latitude_rad)

    @property
    def ship_longitude_deg(self) -> float:
        """Ship longitude in degrees (converted from radians)."""
        return math.degrees(self._ship_longitude_rad)

    @property
    def fish_latitude_deg(self) -> float:
        """Towfish latitude in degrees (converted from radians)."""
        return math.degrees(self._fish_latitude_rad)

    @property
    def fish_longitude_deg(self) -> float:
        """Towfish longitude in degrees (converted from radians)."""
        return math.degrees(self._fish_longitude_rad)

    @property
    def layback_fish_latitude_deg(self) -> float:
        """Layback-corrected towfish latitude in degrees."""
        return math.degrees(self._layback_fish_latitude_rad)

    @property
    def layback_fish_longitude_deg(self) -> float:
        """Layback-corrected towfish longitude in degrees."""
        return math.degrees(self._layback_fish_longitude_rad)

    @property
    def sonar_freq_hz(self) -> int:
        """Operating frequency in Hz (converted from kHz)."""
        return self._sonar_freq_khz * 1000

    @property
    def sonar_freq_khz(self) -> int:
        """Operating frequency in kHz (raw spec value)."""
        return self._sonar_freq_khz

    @property
    def fix_time_second(self) -> float:
        """NMEA fix second."""
        return self._fix_time_second

    @property
    def hsecond(self) -> int:
        """Hundredths of second from `hSecond` field (legacy; prefer `fseconds`)."""
        return self._hsecond

    @property
    def fseconds(self) -> float:
        """TPU fractional seconds (preferred over `hSecond`)."""
        return self._fseconds

    # ============================================================
    # ERROR FLAG ACCESSORS
    # ============================================================

    @property
    def error_flags(self) -> int:
        """Raw error flags bit field."""
        return self._error_flags

    def has_error(self, bit: int) -> bool:
        """Check if a specific error flag bit is set.

        Bits per spec:
            0: Invalid speed
            1: GPS data error
            2: Telemetry error
            3: Sensor checksum error
            4: Towfish leak
            5: Bad data
            6: Watchdog
            7: Compass error
            8: No GPS Lat/Lon sentence input
            9: No GPS speed sentence input
            10: No GPS ZDA sentence input
            11: No data from MRU or MRU error
            12: No 1 PPS input
        """
        return bool(self._error_flags & (1 << bit))


# ============================================================
# PAGE-LEVEL PARSER (DISPATCHER)
# ============================================================

@dataclass
class SDFSonarPage:
    """A parsed Klein SDF sonar data page (versions 3501/3502).

    Wraps the SDFPageHeader plus the per-channel sample arrays.
    The reader populates port_data, stbd_data, and extension_bytes
    after parsing the header.
    """
    header: SDFPageHeader
    port_data: np.ndarray = field(default=None)
    stbd_data: np.ndarray = field(default=None)
    extension_bytes: bytes = field(default=b"")
    # TODO: parse extension_bytes when spec is available

    # Convenience accessors that delegate to header
    @property
    def page_version(self) -> int:
        return self.header.page_version

    @property
    def page_size(self) -> int:
        return self.header.page_size

    @property
    def ping_number(self) -> int:
        return self.header.ping_number

    @property
    def num_samples(self) -> int:
        return self.header.num_samples

    @property
    def timestamp(self) -> float:
        return self.header.timestamp

    @property
    def timestamp_ns(self) -> int:
        return self.header.timestamp_ns

    @property
    def latitude(self) -> float:
        """Towfish latitude (degrees), preferring fish_latitude over ship."""
        fish = self.header.fish_latitude_deg
        if fish != 0.0 and not math.isnan(fish):
            return fish
        return self.header.ship_latitude_deg

    @property
    def longitude(self) -> float:
        """Towfish longitude (degrees)."""
        fish = self.header.fish_longitude_deg
        if fish != 0.0 and not math.isnan(fish):
            return fish
        return self.header.ship_longitude_deg

    @property
    def heading_deg(self) -> float:
        return self.header.heading_deg

    @property
    def pitch_deg(self) -> float:
        return self.header.pitch_deg

    @property
    def roll_deg(self) -> float:
        return self.header.roll_deg

    @property
    def depth_m(self) -> float:
        """Depth in meters. Note: spec stores as Volts; needs INI calibration."""
        # TODO: apply INI calibration. For now, return raw volts as meters.
        return self.header.depth_volts

    @property
    def fathometer_reading_m(self) -> float:
        """Raw altimeter reading from towfish."""
        return self.header.altitude_m

    @property
    def speed_ms(self) -> float:
        return self.header.speed_ms

    @property
    def range_m(self) -> float:
        return float(self.header.range_m)

    @property
    def center_freq_hz(self) -> int:
        return self.header.sonar_freq_hz


@dataclass
class SDFUnknownPage:
    """A parsed Klein SDF page of unknown/unsupported version.

    The header is fully parsed (since all SDF pages share the same
    header structure), but the body is preserved as raw bytes for
    later analysis. Use the page version to identify what kind of
    page it is and decide whether to add typed parsing.

    Attributes
    ----------
    header : SDFPageHeader
        Parsed page header (always valid).
    body_bytes : bytes
        Raw body content (size = header.page_size - PAGE_HEADER_BYTES).
        Includes everything between the header and the next page's marker.
    """
    header: SDFPageHeader
    body_bytes: bytes = field(default=b"")

    @property
    def page_version(self) -> int:
        return self.header.page_version

    @property
    def page_size(self) -> int:
        return self.header.page_size

    @property
    def ping_number(self) -> int:
        return self.header.ping_number

    @property
    def timestamp(self) -> float:
        return self.header.timestamp
