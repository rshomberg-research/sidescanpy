#!/usr/bin/env python3
'''jsf messages - define message types for jsf files

JSF files are from EdgeTech Side Scan Sonars. They are binary formatted and
contain a series of messages. Each message starts with a header that contains
basic information about the message. 

    
2025 rshomberg@uri.edu
'''

# TODO: organize and document  
# This file is very very long. It would probably be helpful to break it up or
# something

import logging
from pathlib import Path
import struct
import pynmea2

from dataclasses import dataclass, field

import struct
from typing import ClassVar, Tuple, Optional
from typing import ClassVar, Type, Dict

from datetime import datetime, timezone, timedelta
import numpy as np

@dataclass
class JSFMessageHeader:
    # Static struct format
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<HBBHBBBBHI')

    marker: int
    protocol_version: int
    session_ID: int
    message_type: int
    command_type: int
    subsystem_num: int
    channel: int
    sequence_number: int
    _reserved: int = field(repr=False)
    message_size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "JSFMessageHeader":
        unpacked = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*unpacked)
    
    # TODO: add a post_init for error checking


@dataclass
class JSFMessageBody:
    _registry: ClassVar[Dict[int, Type['JSFMessageBody']]] = {}
    MESSAGE_TYPE: int   # FIXME: this should be abstract and implemented later

    # Automatically register subclasses
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'MESSAGE_TYPE'):
            JSFMessageBody._registry[cls.MESSAGE_TYPE] = cls    

    @classmethod
    def from_bytes(cls, message_type, data: bytes) -> 'JSFMessageBody':
        msg_cls = cls._registry.get(message_type)
        if not msg_cls:
            raise ValueError(f"Unsupported message type: {message_type}")
        return msg_cls.from_bytes(data) # FIXME: fix the cause of the ide warning here
    
    def set_context(self, header: JSFMessageHeader):
        pass  # default: do nothing    


# Acoustic Messages
@dataclass
class SonarDataMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 80
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<'
        + 'iIIhhHHHhhhhHHhhhhh'                     # Data format block
        + 'fhhhhhhhhhhhhhhhhiih'                    # Navigation data block
        + 'BBBBBBBBBBBBBBBBBBBBBBBBHIHhhHHHiiHHiff' # Pulse information block
        + 'hhhhhh'                                  # CPU Time Block
        + 'hh'                                      # Weighting Factor Block
        + 'Hhhh'                                    # Orientation Sensor Data Block
        + 'hhH'                                     # Trigger Information Block
        + 'hhhhhhh'                                 # NMEA Navigation Block
        + 'IHhhBBBBBBiHhhhfiHH'                     # Miscellaneous Data Block
        )

    # Data format block
    _raw_timestamp_sec: int = field(repr=False)     # seconds since 1970
    _starting_depth:    int = field(repr=False)     # ???: "starting dep (window offset) in samples"
    ping_number:        int = field(repr=True)      # sequential ping number, matched for channels
    
    _reserved01:  int = field(repr=False)
    _reserved02:  int = field(repr=False)

    _most_significant_bits:     int = field(repr=False) # used to extend other values to greater values
    _least_significant_bits:    int = field(repr=False) # used to extend other values to greater precision
    _least_significant_bits2:   int = field(repr=False) # used to extend other values to greater precision

    _reserved03:    int = field(repr=False)
    _reserved04:    int = field(repr=False)
    _reserved05:    int = field(repr=False)

    _id_code:           int = field(repr=False) # always 1 (1=Seismic Data)
    _validity_flags:    int = field(repr=False) # validity flags for various information
    _reserved06:        int = field(repr=False)
    _data_format:       int = field(repr=False) # code representing how bytes are returned in sonar trace

    _raw_y_distance_antenna_to_tow_point:   int = field(repr=False) # cm positive aft
    _raw_x_distance_antenna_to_tow_point:   int = field(repr=False) # cm positive starboard

    _reserved07:  int = field(repr=False)
    _reserved08:  int = field(repr=False)

    # Navigation data block
    _raw_pipe_dist_km: float = field(repr=False) # ???: unknown use, possibly for specific applications
    _reserved09: int = field(repr=False)
    _reserved10: int = field(repr=False)
    _reserved11: int = field(repr=False)
    _reserved12: int = field(repr=False)
    _reserved13: int = field(repr=False)
    _reserved14: int = field(repr=False)
    _reserved15: int = field(repr=False)
    _reserved16: int = field(repr=False)
    _reserved17: int = field(repr=False)
    _reserved18: int = field(repr=False)
    _reserved19: int = field(repr=False)
    _reserved20: int = field(repr=False)
    _reserved21: int = field(repr=False)
    _reserved22: int = field(repr=False)
    _reserved23: int = field(repr=False)
    _reserved24: int = field(repr=False)

    _raw_longitude:     int = field(repr=False) # longitude (see coord units code)
    _raw_latitude:      int = field(repr=False) # latitude  (see coord units code)
    _coord_units_code:  int = field(repr=False) # sets format for lat/lon

    # Pulse information block
    _annotation_str01: bytes = field(repr=False)    # ???: seems empty
    _annotation_str02: bytes = field(repr=False)    # FIXME: find a way to collapse this to 1 attribute
    _annotation_str03: bytes = field(repr=False)
    _annotation_str04: bytes = field(repr=False)
    _annotation_str05: bytes = field(repr=False)
    _annotation_str06: bytes = field(repr=False)
    _annotation_str07: bytes = field(repr=False)
    _annotation_str08: bytes = field(repr=False)
    _annotation_str09: bytes = field(repr=False)
    _annotation_str10: bytes = field(repr=False)
    _annotation_str11: bytes = field(repr=False)
    _annotation_str12: bytes = field(repr=False)
    _annotation_str13: bytes = field(repr=False)
    _annotation_str14: bytes = field(repr=False)
    _annotation_str15: bytes = field(repr=False)
    _annotation_str16: bytes = field(repr=False)
    _annotation_str17: bytes = field(repr=False)
    _annotation_str18: bytes = field(repr=False)
    _annotation_str19: bytes = field(repr=False)
    _annotation_str20: bytes = field(repr=False)
    _annotation_str21: bytes = field(repr=False)
    _annotation_str22: bytes = field(repr=False)
    _annotation_str23: bytes = field(repr=False)
    _annotation_str24: bytes = field(repr=False)

    _raw_num_samples_lsb:           int = field(repr=False) # number of samples in trace (see MSB)
    _raw_sampling_interval_ns_msb:  int = field(repr=False) # sampling interval (nanoseconds) (see MSB)

    _gain_factor:            int = field(repr=False)    # ???: not sure if needed
    _user_transmit_level:    int = field(repr=False)    # ???: not sure if needed
    _reserved25: int = field(repr=False)

    _raw_transmit_freq_start_lsb:   int = field(repr=False) # pulse starting frequency (daHz) (see MSB)
    _raw_transmit_freq_end_lsb:     int = field(repr=False) # pulse ending frequency (daHz) (see MSB)
    _raw_sweep_length_ms:           int = field(repr=False)  # pulse sweep length (ms) (see LSB)
    # TODO: assert sampling interval*num_samples = sweep length

    _raw_pressure:  int = field(repr=False) # pressure (milli PSI) (see validity flags)
    _raw_depth:     int = field(repr=False) # depth (mm) (see validity flags)

    _raw_sample_freq:   int = field(repr=False) # sample freqency (Hz) usually (see Data Format)
    _pulse_id:          int = field(repr=False) # ???: outgoing pulse identifier (not used?)
    
    _raw_altitude:  int = field(repr=False) # altitude (mm (see validity flags))
    sound_speed:    Optional[float]         # sound speed (m/s) (see validity flags)
    mixer_freq:     float                   # mixer frequency (Hz), should be close to center frequency

    # CPU Time Block
    _cpu_time_year:  int = field(repr=False)    # UTC year data recorded on CPU (generally do not use)
    _cpu_time_day:   int = field(repr=False)    # UTC month data recorded on CPU (generally do not use)
    _cpu_time_hour:  int = field(repr=False)    # UTC day data recorded on CPU (generally do not use)
    _cpu_time_min:   int = field(repr=False)    # UTC minute data recorded on CPU (generally do not use)
    _cpu_time_sec:   int = field(repr=False)    # UTC second data recorded on CPU (generally do not use)
    _cpu_time_basis: int = field(repr=False)    # Time Basis (always 3)

    # Weighting Factor Block
    _weighting_factor:  int = field(repr=False) # Weighting factor (N) for block floating point expansion (2-N Volts for LSB) All data MUST be scaled by 2**-N (see Eq. 2-1 in docs)
    _number_of_pulses:  int = field(repr=False) # Number of pulses in the water (should be 1)

    # Orientation Sensor Data Block
    _raw_compass_heading:   int = field(repr=False) # Compass heading (1/100 deg) (see validity flag) 
    # FIXME: unclear if compass is magnetic or True
    _raw_pitch:             int = field(repr=False) # Pitch (degrees/180)&32768.0 positive bow up   (see validity flags)
    _raw_roll:              int = field(repr=False) # Roll  (degrees/180)&32768.0 positive port up  (see validity flags)
    _reserved26:    int = field(repr=False)

    # Trigger Information Block
    _reserved27:            int = field(repr=False)
    _trigger_source_code:   int = field(repr=False) # ???: Trigger source (see docs)
    _mark_number_code:      int = field(repr=False) # ???: Mark number (see MSB)

    # NMEA Navigation Data Block  
    _position_fix_hr:    int = field(repr=False) # time (hour) of lat/lon fix (see validity flags)
    _position_fix_min:   int = field(repr=False) # time (minute) of lat/lon fix (see validity flags)
    _position_fix_sec:   int = field(repr=False) # time (second) of lat/lon fix (see validity flags)
    _raw_course_msb:     int = field(repr=False) # Course 0-359.9 (see LSB)
    _raw_speed_msb:      int = field(repr=False) # Speed (1/10 knots) (see LSB)
    _position_fix_day:   int = field(repr=False) # time (day) of lat/lon fix (1-366) (see validity flags)
    _position_fix_yr:    int = field(repr=False) # time (year) of lat/lon fix (see validity flags)

    # Miscellaneous Data Block
    _timestamp_day_offset_ms:   int = field(repr=False) # milliseconds today since midnight for timestamp
    _max_abs_adc_val:           int = field(repr=False) # ???: maybe use for normalizing?
    _reserved28:                int = field(repr=False)
    _reserved29:                int = field(repr=False)
    _sonar_software_version1:   int = field(repr=False) # ???: unknown but generally blank
    _sonar_software_version2:   int = field(repr=False) # FIXME: collapse to single attribute
    _sonar_software_version3:   int = field(repr=False)
    _sonar_software_version4:   int = field(repr=False)
    _sonar_software_version5:   int = field(repr=False)
    _sonar_software_version6:   int = field(repr=False)
    _init_spherical_correction_factor:   int = field(repr=False) # ???: Initial spherical correction factor (samples*100) or -1 if spherical spreading disabled
    _packet_number:             int = field(repr=False) # packet number each ping starts with packet 1
    _raw_adc_decimation:        int = field(repr=False) # ???: ADC decimation * 100 times
    _reserved30:                int = field(repr=False)
    _raw_water_temp:            int = field(repr=False) # Water temperature convert 1/10 deg C (see validity flags)
    _layback_distance:        float = field(repr=False) # distance to sonar in meters
    _reserved31:                int = field(repr=False)
    _raw_cable_out:             int = field(repr=False) # ???: cable out (decimeters) (see validity flags)
    _reserved32:                int = field(repr=False)

    _raw_trace_data: list[int] = field(repr=False)    # Raw data output of trace

    # Converted fields
    timestamp:              np.datetime64 = field(init=False) # seconds since 1970
    channel:                  int = field(init=False) # Sonar channel (0=port, 1=stbd) (should have corresponding pings)
    course:       Optional[float] = field(init=False) # deg (0-259.9)
    speed:                  float = field(init=False) # speed (knots) # ???: not sure if over ground or water
    sample_interval:        float = field(init=False) # sec
    sweep_length:           float = field(init=False) # sec

    gps_offset_x:            float = field(init=False, repr=False) # Distance from antenna to tow points (starboard positive)
    gps_offset_y:            float = field(init=False, repr=False) # Distance from antenna to tow points (aft positive)
    latitude:      Optional[float] = field(init=False)   # decimal degrees
    longitude:     Optional[float] = field(init=False)   # decimal degrees

    transmit_freq_start:    float = field(init=False)   # Hz
    transmit_freq_end:      float = field(init=False)   # Hz
    sweep_length:           float = field(init=False)   # sec
    pressure_psi: Optional[float] = field(init=False)   # psi
    depth:        Optional[float] = field(init=False)   # meters
    sample_freq:            float = field(init=False)   # Hz

    altitude:     Optional[float] = field(init=False)   # meters
    heading:      Optional[float] = field(init=False)   # degrees - towfish compass heading
    pitch:        Optional[float] = field(init=False)   # degrees - positive bow up
    roll:         Optional[float] = field(init=False)   # degrees - positive port up
    mark_number:    Optional[int] = field(init=False)   # ???: possibly something the user can add by marking a location

    water_temp:      Optional[float] = field(init=False, repr=False) # deg C
    cable_out:       Optional[float] = field(init=False, repr=False) # meters
    pipe_kilometers: Optional[float] = field(init=False, repr=False) # TODO

    annotation_string:  Optional[str] = field(init=False, repr=False)

    backscatter_intensity:  list[float] = field(init=False, repr=False) # Sonar trace data # ???: dB?
    N_samples:              int = field(init=False)

    @staticmethod
    def parse_validity_flags(flags: int) -> dict:
        logging.debug(f"Validity Flags: {bin(flags)}")
        return {
            "lat_lon":            bool(flags & (1 << 0)),
            "course":             bool(flags & (1 << 1)),
            "speed":              bool(flags & (1 << 2)),
            "heading":            bool(flags & (1 << 3)),
            "pressure":           bool(flags & (1 << 4)),
            "pitch_roll":         bool(flags & (1 << 5)),
            "altitude":           bool(flags & (1 << 6)),
            "reserved":           bool(flags & (1 << 7)),
            "water_temp":         bool(flags & (1 << 8)),
            "depth":              bool(flags & (1 << 9)),
            "annotation":         bool(flags & (1 << 10)),
            "cable_counter":      bool(flags & (1 << 11)),
            "kp":                 bool(flags & (1 << 12)),
            "position_interpolated":    bool(flags & (1 << 13)),
            "water_sound_speed":  bool(flags & (1 << 14)),
            # Bit 15 unused or undefined
        }

    def __post_init__(self) -> None:

        # NOTE: from docs -   
        # For higher resolution (milliseconds) use the Year, and Day values of
        # bytes 156 to 159, and then use the milliSecondsToday value of bytes
        # 200-203 to complete the timestamp
        base_date = datetime(self._cpu_time_year, 1, 1, 
                            #  tzinfo=timezone.utc,
                             ) 
        dt = base_date + timedelta(days=self._cpu_time_day - 1, milliseconds=self._timestamp_day_offset_ms)
        logging.debug((dt, dt.timestamp(), float(dt.timestamp())))
        self.timestamp = np.datetime64(dt, "ns") # as unix timestamp

        # TODO: why does this not work
        # self.timestamp = self._raw_timestamp_sec + self._timestamp_day_offset_ms % 1e3

        # NOTE: from docs -  
        # MSBs – Most Significant Bits – High order bits to extend 16 bits
        # unsigned short values to 20 bits. The 4MSB bits become the most
        # significant portion of the new 20 bit value
        combine_20bit = lambda msb,lsb: (msb << 16) | lsb

        msb_field = self._most_significant_bits
        start_freq_msb  = (msb_field >> 0) & 0xF    # bits  0- 3
        end_freq_msb    = (msb_field >> 4) & 0xF    # bits  4- 7
        samples_msb     = (msb_field >> 8) & 0xF    # bits  8-11
        mark_number     = (msb_field >> 12) & 0xF   # bits 12-15

        # LSB – Extended precision - Low order bits for fields requiring greater
        # precision.
        lsb_field = self._least_significant_bits
        sample_interval_frac    = lsb_field & 0xFF        # bits 0- 7
        course_frac             = (lsb_field >> 8) & 0xFF # bits 8-11

        lsb2_field = self._least_significant_bits2
        speed_sub_frac  = (lsb2_field) & 0xF          # bits  0- 3
        sweep_length_us = (lsb2_field >> 4) & 0x3FF   # bits  4-13
        reserved        = (lsb2_field >> 14) & 0x3    # bits 14-15
    
        self.course = self._raw_course_msb + course_frac / 256.0      # FIXME: check math
        self.speed  = self._raw_speed_msb/10 + speed_sub_frac/100.0   # FIXME: check math

        # Validity flags
        validity_flags = self.parse_validity_flags(self._validity_flags)
        # FIXME: do validity flags better (see DVL message)
        if validity_flags['lat_lon'] is not True:
            logging.debug(f"Invalid lat/lon")
            self.latitude   = None
            self.longitude  = None
        if validity_flags['course'] is not True:
            logging.debug(f"Invalid course")
            self.course = None
        if validity_flags['heading'] is not True:
            logging.debug(f"Invalid heading")
            self.heading = None
        if validity_flags['pressure'] is not True:
            logging.debug(f"Invalid pressure value")
            self.pressure_psi = None
        if validity_flags['pitch_roll'] is not True:
            logging.debug(f"Invalid pitch/role value")
            self.pitch  = None
            self.roll   = None
        if validity_flags['altitude'] is not True:
            logging.debug(f"Invalid altitude value")
            self.altitude = None
        if validity_flags['water_temp'] is not True:
            logging.debug(f"Invalid water temperature")
            self.water_temp = None
        if validity_flags['depth'] is not True:
            logging.debug(f"Invalid depth value")
            self.depth = None
        if validity_flags['annotation'] is not True:
            logging.debug(f"Invalid annotation value")
            self.annotation_string = None
        if validity_flags['cable_counter'] is not True:
            logging.debug(f"Invalid cable out value")
            self.cable_out = None
        if validity_flags['kp'] is not True:
            logging.debug(f"Invalid kilometers of pipe value")
            self.pipe_kilometers = None
        if validity_flags['position_interpolated'] is not True:
            # TODO: figure out position
            logging.debug(f"Position not interpolated")
            # NOTE: Unless the Validity Flag Bit 13 “Position Interpolated” is
            # set, the position stored in message 80 is the value recorded by
            # Discover and is not the sonar’s position. It is the last
            # navigation position received prior to pinging and Layback is not
            # applied.
        if validity_flags['water_sound_speed'] is not True:
            logging.debug(f"Invalid sound speed value")
            self.sound_speed = None

        # Get position information
        self.gps_offset_x = self._raw_x_distance_antenna_to_tow_point/1e2
        self.gps_offset_y = self._raw_y_distance_antenna_to_tow_point/1e2

        self._raw_pipe_dist_km  # TODO: unsure what this is for

        if self._coord_units_code == 1:
            raise NotImplementedError(f"Coordinate Units ({self._coord_units_code} (mm) not implemented)")    # TODO
        elif self._coord_units_code == 2:
            self.latitude   = self._raw_latitude    / (60*10e3)
            self.longitude  = self._raw_longitude   / (60*10e3)
        elif self._coord_units_code == 3:
            raise NotImplementedError(f"Coordinate Units ({self._coord_units_code} (dm) not implemented)")    # TODO
        else:
            raise ValueError(f"Invalid coordinate code: {self._coord_units_code}")
        # TODO: check validity bit 13 "Position Interpolated"
        # TODO: adjust units
        

        # Pulse Information
        byte_list = [
            self._annotation_str01,
            self._annotation_str02,
            self._annotation_str03,
            self._annotation_str04,
            self._annotation_str05,
            self._annotation_str06,
            self._annotation_str07,
            self._annotation_str08,
            self._annotation_str09,
            self._annotation_str10,
            self._annotation_str11,
            self._annotation_str12,
            self._annotation_str13,
            self._annotation_str14,
            self._annotation_str15,
            self._annotation_str16,
            self._annotation_str17,
            self._annotation_str18,
            self._annotation_str19,
            self._annotation_str20,
            self._annotation_str21,
            self._annotation_str22,
            self._annotation_str23,
            self._annotation_str24,
        ]   # ???: always seems to be empty
        annotation_string = ''.join(bytes(byte).decode() for byte in byte_list)

        num_samples = combine_20bit(samples_msb, self._raw_num_samples_lsb)
        # logging.debug((samples_msb, self._raw_num_samples_lsb))
        # logging.debug(f"Number of samples {num_samples}")
        # FIXME: this value is lower than the number of bytes, and the MSB seems to be 0
        # NOTE: it is half. I think you multiply by two to get number of bytes
        self.sample_interval = (self._raw_sampling_interval_ns_msb + sample_interval_frac / 256.0) / 1e9
        self._gain_factor           # ???: do we need to use this
        self._user_transmit_level   # ???: do we need to use this
        self.transmit_freq_start    = combine_20bit(start_freq_msb,self._raw_transmit_freq_start_lsb)   * 1e1   # daHz -> Hz
        self.transmit_freq_end      = combine_20bit(end_freq_msb,self._raw_transmit_freq_end_lsb)       * 1e1   # daHz -> Hz

        self.sweep_length = self._raw_sweep_length_ms / 1e3 + sweep_length_us / 1e6 # -> sec

        self.pressure_psi   = self._raw_pressure / 1e3    # milli-PSI -> PSI
        # TODO: convert to Pa or kPa?
        # TODO: see validity flags
        self.depth          = self._raw_depth / 1e3 # mm -> m
        # TODO: see validity flags
        
        if self._data_format == 0:      # one short per sample (total bytes = 2 * samples)
            self.sample_freq = self._raw_sample_freq
        elif self._data_format == 1:    # two shorts per sample (real, imag) (total bytes = 4 * samples)
            self.sample_freq = self._raw_sample_freq
            raise NotImplementedError(f"Unimplemented data format: {self._data_format}")    # TODO
        elif self._data_format == 2:    # one short per sample (total bytes = 2 * samples)
            raise NotImplementedError(f"Unimplemented data format: {self._data_format}")    # TODO
        elif self._data_format == 9:    # two shorts per sample (real, imag) (total bytes = 4 * samples)
            self.sample_freq = self._raw_sample_freq
            raise NotImplementedError(f"Unimplemented data format: {self._data_format}")    # TODO
        else:   # NOTE: values > 255 indicate compressed data
            raise ValueError(f"Unknown data format: {self._data_format}")
        # TODO: Use this in conjunction with the Sample Interval (bytes 114-115) to calculate correct sample rate
        # FIXME: "All values are modulo 65536"
        self.altitude = self._raw_altitude / 1e3    # mm -> m

        # Weighting Factor
        # TODO: all data MUST be scaled by 2**(-N)
        self._weighting_factor  # Weighting Factor for Block Floating Point Expansion -- defined as 2 to N Volts for LSB
        assert self._number_of_pulses==1, f"Number of pulses ({self._number_of_pulses}) != 1"

        # Orientation Sensor Block
        self.heading = self._raw_compass_heading / 1e2
        # FIXME: See Validity Flag (bytes 30-31). The Compass heading is the
        # magnetic heading of the towfish. If a Gyro sensor is properly
        # interfaced to the DISCOVER Topside Acquisition Unit with a valid NMEA
        # HDT message, this field will contain the Gyro heading, relative to
        # True North.
        self.pitch  = self._raw_pitch * 180/32768.0 # TODO: see validity flag
        self.roll  = self._raw_roll * 180/32768.0   # TODO: see validity flag

        # TODO: trigger source
        if self._mark_number_code == 0:
            self.mark_number = None
        else:
            self.mark_number = mark_number  # TODO: check this

        # TODO: check position fix compared to cpu time
        # print((self._position_fix_sec, self._cpu_time_sec))
        

        self._max_abs_adc_val   # ???: error checking?

        byte_list = [
            self._sonar_software_version1,
            self._sonar_software_version2,
            self._sonar_software_version3,
            self._sonar_software_version4,
            self._sonar_software_version5,
            self._sonar_software_version6,
        ]
        sonar_software_version = ''.join(bytes(byte).decode() for byte in byte_list)    # ???: appears to be empty

        if self._init_spherical_correction_factor == -1:
            raise NotImplementedError(f"Initial spherical correction factor (-1) is disabled")
        self._init_spherical_correction_factor/1e2  # ???: not sure what this does but it is 0

        if self._packet_number > 1:
            raise NotImplementedError(f"Multiple packets not implemented")
        
        self._raw_adc_decimation/1e2 # ???: always 1.0?

        self.water_temp = self._raw_water_temp / 1e1    # TODO: check validity flags
        self.cable_out = self._raw_cable_out / 1e3      # TODO: check validity flags

        # Handle trace data
        self.backscatter_intensity = [float(val)*2**(-self._weighting_factor) for val in self._raw_trace_data]  # TODO: do some math on the trace data samples
        self.N_samples = len(self.backscatter_intensity)
        # TODO: error checking

    @classmethod
    def from_bytes(cls, data: bytes) -> "SonarDataMessage":
        fixed_fields = cls.STRUCT.unpack(data[:cls.STRUCT.size])

        # Pull out required fields using slicing
        num_samples = struct.unpack_from('<H', data, 114)[0]    # FIXME: requires MSB1 field
        data_format = struct.unpack_from('<H', data, 34)[0]

        # FIXME: document this and figure out details and refactor to function (see above)
        if data_format == 0:
            ints_per_sample = 1
        elif data_format == 1:
            ints_per_sample = 2
        elif data_format == 2:
            ints_per_sample = 1
        elif data_format ==9:
            ints_per_sample = 2
        elif data_format > 255:
            raise ValueError(f"Data is compressed and must be decompressed (see docs)")
        else:
            raise ValueError(f"Invalid data format: {data_format} (see docs)")
        
        # logging.debug(f"Num Samples: {num_samples}")
        # logging.debug(f"Ints per sample: {ints_per_sample}")

        # Compute expected data length
        bytes_per_sample = 2*ints_per_sample
        expected_data_bytes = num_samples * bytes_per_sample
        actual_data_bytes = len(data) - 240
        logging.debug((expected_data_bytes, actual_data_bytes))

        if actual_data_bytes != expected_data_bytes:
            raise ValueError(
                f"Mismatch in trace data length: expected {expected_data_bytes} bytes, "
                f"got {actual_data_bytes}"
            )

        # Extract and decode trace data as list of int16
        raw_trace_data = list(struct.unpack(f'<{num_samples * ints_per_sample}h', data[240:]))
        # logging.debug(trace_data)

        return cls(*fixed_fields, _raw_trace_data=raw_trace_data)
    
    def set_context(self, header: JSFMessageHeader):
        self.channel = header.channel    

@dataclass
class LegacySideScanDataMessage(JSFMessageBody):    # TODO: low priority
    MESSAGE_TYPE: ClassVar[int] = 82
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<')

    @classmethod
    def from_bytes(cls, data: bytes) -> "LegacySideScanDataMessage":
        raise NotImplementedError
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))

# Other Messages

# TODO:NavigationOffsetsMessage (181)
@dataclass
class NavigationOffsetsMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 181
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<ffffffffffffffff')  # 16 floats

    x_offset: float                      # meters
    y_offset: float                      # meters
    latitude_offset: float               # degrees
    longitude_offset: float              # degrees
    aft_offset: float                    # meters (forward is negative)
    starboard_offset: float              # meters (port is negative)
    depth_offset: float                  # meters (up is negative)
    altitude_offset: float               # meters (down is negative)
    heading_offset: float                # degrees
    pitch_offset: float                  # degrees (nose up is positive)
    roll_offset: float                   # degrees (port side up is positive)
    yaw_offset: float                    # degrees (toward port is negative)
    tow_point_elevation_offset: float    # meters (up is positive)
    reserved_1: float
    reserved_2: float
    reserved_3: float

    @classmethod
    def from_bytes(cls, data: bytes) -> "NavigationOffsetsMessage":
        values = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*values)
    
# TODO: SystemInformationMessage (182)
@dataclass
class SystemInformationMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 182
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiiiii')

    system_type: int
    low_rate_io_enabled_option: int     # 0 = disabled
    sonar_software_version: int
    num_subsystems: int
    num_serial_devices: int
    tow_vehicle_serial: int
    # _reserved
    # FIXME: changes size based on unknown values

    @classmethod
    def from_bytes(cls, data: bytes) -> "SystemInformationMessage":
        values = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*values)    

@dataclass
class FileTimeStampMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 426
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<ii')

    timestamp:  float = field(init=False)
    _raw_time_seconds:  int = field(repr=False)
    _raw_offset_ms:     int = field(repr=False)

    def __post_init__(self) -> None:
        self.timestamp = self._raw_time_seconds + self._raw_offset_ms/1e3

    @classmethod
    def from_bytes(cls, data: bytes) -> "FileTimeStampMessage":
        values = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*values)
    
@dataclass
class FilePaddingMessage(JSFMessageBody):
    # TODO: document
    MESSAGE_TYPE: ClassVar[int] = 428
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<')

    @classmethod
    def from_bytes(cls, data: bytes) -> "FilePaddingMessage":
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))

# Auxiliary Messages

@dataclass
class NMEAStringMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 2002
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiBBBB')
    _registry: ClassVar[dict[str, type]] = {}

    timestamp: float = field(init=False)
    _raw_time_seconds: int = field(repr=False)
    _raw_offset_ms:    int = field(repr=False)
    
    source: int

    _reserved01: int = field(repr=False)
    _reserved02: int = field(repr=False)
    _reserved03: int = field(repr=False)

    _raw_message: str = field(repr=False)

    def __post_init__(self) -> None:
        self.timestamp = self._raw_time_seconds + self._raw_offset_ms / 1e3

    def parse_sentence(self):
        return pynmea2.parse(self._raw_message)

    # --- Factory method ---
    def __init_subclass__(cls, nmea_prefix: str | None = None, **kwargs):
        """Register subclasses automatically if they declare a prefix."""
        super().__init_subclass__(**kwargs)
        if nmea_prefix:
            NMEAStringMessage._registry[nmea_prefix] = cls

    @classmethod
    def from_bytes(cls, data: bytes) -> "NMEAStringMessage":
        fixed_fields = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        message_str = bytes(data[cls.STRUCT.size:]).decode('ascii', errors='replace').rstrip('\x00')

        # Base object first, to be safe
        base_obj = NMEAStringMessage(*fixed_fields, _raw_message=message_str)

        # Dispatch to subclass if prefix matches
        for prefix, subclass in cls._registry.items():
            if message_str.startswith(prefix):
                return subclass(*fixed_fields, _raw_message=message_str)

        return base_obj

@dataclass
class NMEADBTMessage(NMEAStringMessage, nmea_prefix="$WHDBT"):

    depth_feet: float       = field(init=False)
    depth_meters: float     = field(init=False)
    depth_fathoms: float    = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        parsed = self.parse_sentence()
        self.depth_feet     = float(parsed.depth_feet)
        self.depth_meters   = float(parsed.depth_meters)
        self.depth_fathoms  = float(parsed.depth_fathoms)

@dataclass
class NMEARMCMessage(NMEAStringMessage, nmea_prefix="$GPRMC"):
    latitude: float     = field(init=False)
    longitude: float    = field(init=False)
    speed_knots: float  = field(init=False)
    course_deg: float   = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        parsed = self.parse_sentence()
        self.latitude       = parsed.latitude
        self.longitude      = parsed.longitude
        self.speed_knots    = parsed.spd_over_grnd
        self.course_deg     = parsed.true_course

@dataclass
class NMEADPTMessage(NMEAStringMessage, nmea_prefix="$WHDPT"):
    depth_meters: float = field(init=False)
    offset_meters: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        parsed = self.parse_sentence()
        # pynmea2's DPT object stores depth and offset
        self.depth_meters = float(parsed.depth)
        self.offset_meters = float(parsed.offset) if parsed.offset is not None else 0.0

@dataclass
class PitchRollDataMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 2020
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiBBBBhhhhhhhhhHhHihh')

    # Message fields
    _raw_time_seconds:  int = field(repr=False) # seconds since 1970
    _raw_offset_ms:     int = field(repr=False) # milliseconds in current second

    _reserved01: int = field(repr=False)
    _reserved02: int = field(repr=False)
    _reserved03: int = field(repr=False)
    _reserved04: int = field(repr=False)

    _raw_accel_x:   int = field(repr=False) # TODO: convert to m/s2
    _raw_accel_y:   int = field(repr=False)
    _raw_accel_z:   int = field(repr=False)
    _raw_rate_x:    int = field(repr=False) # TODO: convert to m/s
    _raw_rate_y:    int = field(repr=False)
    _raw_rate_z:    int = field(repr=False)
    _raw_pitch:     int = field(repr=False) # TODO: convert to deg
    _raw_roll:      int = field(repr=False)

    _raw_temperature:   int = field(repr=False) # TODO: convert to Cel
    _raw_diagnostics:   int = field(repr=False)   # device specific

    _raw_heave:     int = field(repr=False) # TODO: convert to meter
    _raw_heading:   int = field(repr=False) # TODO: convert to deg
    _data_flags:    int = field(repr=False) # TODO: break out
    _raw_yaw:       int = field(repr=False) # ???: different from heading?

    _reserved05: int = field(repr=False)

    # Derived fields
    timestamp:  float = field(init=False)   # seconds since 1970

    pitch:  Optional[float] = field(init=False, default=None) # deg positive bow up
    roll:   Optional[float] = field(init=False, default=None) # deg positive port up

    heave:      Optional[float] = field(init=False, default=None) # meters positive down 
    heading:    Optional[float] = field(init=False, default=None) # deg (0-360)
    yaw:        Optional[float] = field(init=False, default=None) # deg (0-360)

    temperature:    Optional[float] = field(init=False, default=None) # deg C

    accel_x:  Optional[float] = field(init=False, default=None)  # Gs
    accel_y:  Optional[float] = field(init=False, default=None)  # Gs
    accel_z:  Optional[float] = field(init=False, default=None)  # Gs

    rate_x:   Optional[float] = field(init=False, default=None)  # deg/sec
    rate_y:   Optional[float] = field(init=False, default=None)  # deg/sec
    rate_z:   Optional[float] = field(init=False, default=None)  # deg/sec

    @staticmethod
    def parse_validity_flags(flags: int) -> dict[str,bool]:
        logging.debug(f"Validity Flags: {bin(flags)}")
        return {
            "accel_x":      bool(flags & (1 << 0)),
            "accel_y":      bool(flags & (1 << 1)),
            "accel_z":      bool(flags & (1 << 2)),
            "rate_x":       bool(flags & (1 << 3)),
            "rate_y":       bool(flags & (1 << 4)),
            "rate_z":       bool(flags & (1 << 5)),
            "pitch":        bool(flags & (1 << 6)),
            "roll":         bool(flags & (1 << 7)),
            "heave":        bool(flags & (1 << 8)),
            "heading":      bool(flags & (1 << 9)),
            "temperature":  bool(flags & (1 << 10)),
            "device_info":  bool(flags & (1 << 11)),
            "yaw":          bool(flags & (1 << 12)),
        }

    def __post_init__(self) -> None:
        # TODO: add necessary values as needed
        self.timestamp = self._raw_time_seconds + self._raw_offset_ms/1e3
        validity_flags = self.parse_validity_flags(self._data_flags)

        if validity_flags['accel_x']:
            self.accel_x = self._raw_accel_x * (20*1.5)/32768
        if validity_flags['accel_y']:
            self.accel_y = self._raw_accel_y * (20*1.5)/32768
        if validity_flags['accel_z']:
            self.accel_z = self._raw_accel_z * (20*1.5)/32768
        if validity_flags['rate_x']:
            self.rate_x = self._raw_rate_x * (500*1.5)/(32768)
        if validity_flags['rate_y']:
            self.rate_y = self._raw_rate_y * (500*1.5)/(32768)
        if validity_flags['rate_z']:
            self.rate_z = self._raw_rate_z * (500*1.5)/(32768)
        if validity_flags['pitch']:
            self.pitch = self._raw_pitch * (180.0/32768.0)
        if validity_flags['roll']:
            self.roll = self._raw_roll * (180.0/32768.0)
        if validity_flags['heave']:
            self.heave = self._raw_heave / 1e3  # mm -> meters
        if validity_flags['heading']:
            self.heading = self._raw_heading * 0.01     # -> deg
        if validity_flags['temperature']:
            self.temperature = self._raw_temperature * 10   # -> deg C
        if validity_flags['device_info']:
            pass    # TODO: define device info
        if validity_flags['yaw']:
            self.yaw = self._raw_yaw * 0.01 # -> deg

    @classmethod
    def from_bytes(cls, data: bytes) -> "PitchRollDataMessage":
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
@dataclass
class PressureSensorReadingMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 2060
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiBBBB'+'iiiiiiiiiiiiiiii')

    _raw_time_seconds:  int = field(repr=False)
    _raw_offset_ms:     int = field(repr=False)

    _reserved01: int = field(repr=False)
    _reserved02: int = field(repr=False)
    _reserved03: int = field(repr=False)
    _reserved04: int = field(repr=False)

    _raw_pressure:      int = field(repr=False)      # 1/1000th PSI
    _raw_temperature:   int = field(repr=False)      # 1/1000th °C
    _raw_salinity:      int = field(repr=False)      # PPM
    _data_flags:        int = field(repr=False)      # TODO: break out flags

    _raw_conductivity:  int = field(repr=False)      # micro-Siemens/cm
    _raw_sound_vel:     int = field(repr=False)      # mm/s
    _raw_depth:         int = field(repr=False)      # meters

    _reserved05: int = field(repr=False)
    _reserved06: int = field(repr=False)
    _reserved07: int = field(repr=False)
    _reserved08: int = field(repr=False)
    _reserved09: int = field(repr=False)
    _reserved10: int = field(repr=False)
    _reserved11: int = field(repr=False)
    _reserved12: int = field(repr=False)
    _reserved13: int = field(repr=False)

    # Derived fields
    timestamp:      float = field(init=False)
    pressure:       float = field(init=False)        # PSI
    temperature:    float = field(init=False)        # °C
    salinity:       float = field(init=False)        # PPM
    conductivity:   float = field(init=False)        # micro-Siemens/cm
    sound_speed:    float = field(init=False)        # m/s
    depth:          float = field(init=False)        # meters

    def __post_init__(self) -> None:
        self.timestamp = self._raw_time_seconds + self._raw_offset_ms / 1e3
        self.pressure = self._raw_pressure / 1e3
        self.temperature = self._raw_temperature / 1e3
        self.salinity = self._raw_salinity
        self.conductivity = self._raw_conductivity
        self.sound_speed = self._raw_sound_vel / 1e3      # mm/s -> m/s
        self.depth = self._raw_depth

    @classmethod
    def from_bytes(cls, data: bytes) -> "PressureSensorReadingMessage":
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    

def calculate_dvl_altitude_plane_fit(beam_distances, beam_angle=20.0):  # FIXME: maybe move this
    """
    Calculate altitude, pitch, and roll from DVL 4-beam distances using plane fitting.
    
    This method fits a plane to the 4 beam endpoints and extracts:
    - Altitude (vertical distance to the fitted plane)
    - Pitch (rotation about cross-track axis)
    - Roll (rotation about along-track axis)
    
    DVL beams are typically arranged in a Janus configuration:
    - Beam 1: Forward-starboard (+x, +y)
    - Beam 2: Aft-starboard (-x, +y)
    - Beam 3: Aft-port (-x, -y)
    - Beam 4: Forward-port (+x, -y)
    
    Parameters:
    -----------
    beam_distances : array-like
        Bottom distances for 4 beams, shape (n_samples, 4) or (4,)
    beam_angle : float
        Beam angle from vertical in degrees (default: 20.0)
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'altitude': Vertical distance to bottom plane (m)
        - 'pitch': Pitch angle from plane fit (degrees, positive = bow down)
        - 'roll': Roll angle from plane fit (degrees, positive = starboard down)
        - 'altitude_std': RMS residual of plane fit (m)
        - 'altitude_per_beam': Vertical distance for each beam (m)
        - 'bottom_points': 3D coordinates of beam endpoints (m)
        - 'plane_normal': Unit normal vector of fitted plane
        - 'quality_flag': 0=good, 1=suspect, 2=bad (insufficient beams)
    """
    
    beam_distances = np.asarray(beam_distances)
    beam_angle_rad = np.radians(beam_angle)
    
    # Handle both single measurement and time series
    single_measurement = beam_distances.ndim == 1
    if single_measurement:
        beam_distances = beam_distances.reshape(1, -1)
    
    n_samples = beam_distances.shape[0]
    
    # Define beam unit vectors in DVL coordinate frame
    # DVL frame: x=forward, y=starboard, z=down
    sin_a = np.sin(beam_angle_rad)
    cos_a = np.cos(beam_angle_rad)
    
    # Beam unit vectors (normalized)
    beam_vectors = np.array([
        [sin_a, sin_a, cos_a],    # Beam 1: forward-starboard
        [-sin_a, sin_a, cos_a],   # Beam 2: aft-starboard
        [-sin_a, -sin_a, cos_a],  # Beam 3: aft-port
        [sin_a, -sin_a, cos_a]    # Beam 4: forward-port
    ])
    
    # Initialize output arrays
    altitude = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    roll = np.zeros(n_samples)
    altitude_std = np.zeros(n_samples)
    quality_flag = np.zeros(n_samples, dtype=int)
    plane_normals = np.zeros((n_samples, 3))
    all_bottom_points = np.zeros((n_samples, 4, 3))
    altitude_per_beam = np.zeros((n_samples, 4))
    
    for i in range(n_samples):
        # Get valid beams
        valid = ~np.isnan(beam_distances[i]) & (beam_distances[i] > 0)
        n_valid = np.sum(valid)
        
        if n_valid < 3:
            # Need at least 3 points to fit a plane
            quality_flag[i] = 2
            altitude[i] = np.nan
            pitch[i] = np.nan
            roll[i] = np.nan
            continue
        
        # Calculate 3D positions of beam endpoints
        # bottom_point = beam_distance * beam_unit_vector
        bottom_points = beam_distances[i, valid, np.newaxis] * beam_vectors[valid]
        all_bottom_points[i, valid] = bottom_points
        
        # Fit plane to bottom points using least squares
        # Plane equation: ax + by + cz + d = 0
        # Normalized form: nx*x + ny*y + nz*z = d (where [nx,ny,nz] is unit normal)
        
        if n_valid == 3:
            # Exactly 3 points - compute plane directly
            p1, p2, p3 = bottom_points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
        else:
            # 4 points - use least squares plane fitting
            # Fit plane by finding normal vector that minimizes squared distances
            centroid = np.mean(bottom_points, axis=0)
            centered = bottom_points - centroid
            
            # SVD to find best-fit plane
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]  # Last row = normal to best-fit plane
        
        # Ensure normal points upward (negative z in DVL frame)
        if normal[2] > 0:
            normal = -normal
        
        plane_normals[i] = normal
        
        # Calculate altitude as distance from origin to plane along z-axis
        # Distance from origin to plane: d = normal · point_on_plane
        d = np.dot(normal, bottom_points[0])
        altitude[i] = d / abs(normal[2])  # Project onto vertical
        
        # Calculate pitch and roll from plane normal
        # Normal vector: [nx, ny, nz] where nz ≈ -1 for level bottom
        # Pitch: rotation about y-axis (cross-track)
        # Roll: rotation about x-axis (along-track)
        
        nx, ny, nz = normal
        
        # pitch = arctan(nx / -nz)  # positive when bow down
        # roll = arctan(ny / -nz)   # positive when starboard down
        pitch[i] = np.degrees(np.arctan2(nx, -nz))
        roll[i] = np.degrees(np.arctan2(ny, -nz))
        
        # Calculate residuals (how well points fit the plane)
        distances_to_plane = np.abs(np.dot(bottom_points, normal) - d)
        altitude_std[i] = np.sqrt(np.mean(distances_to_plane**2))
        
        # Altitude per beam (vertical component only)
        altitude_per_beam[i, valid] = bottom_points[:, 2]
        
        # Quality flag
        if altitude_std[i] > 0.5:
            quality_flag[i] = 1  # Suspect - high residuals
        elif n_valid < 4:
            quality_flag[i] = 1  # Suspect - missing beam
    
    # Prepare results
    results = {
        'altitude': altitude,
        'pitch': pitch,
        'roll': roll,
        'altitude_std': altitude_std,
        'altitude_per_beam': altitude_per_beam,
        'bottom_points': all_bottom_points,
        'plane_normal': plane_normals,
        'quality_flag': quality_flag,
        'n_valid_beams': np.sum(~np.isnan(beam_distances) & (beam_distances > 0), axis=1)
    }
    
    # If single measurement, return scalars
    if single_measurement:
        for key in ['altitude', 'pitch', 'roll', 'altitude_std', 'n_valid_beams']:
            results[key] = results[key][0]
        results['quality_flag'] = results['quality_flag'][0]
        results['altitude_per_beam'] = results['altitude_per_beam'][0]
        results['bottom_points'] = results['bottom_points'][0]
        results['plane_normal'] = results['plane_normal'][0]
    
    return results

@dataclass
class DopplerVelocityLogDataMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 2080
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiBBBB'+'I'+'iiii'+'hhhhhhHhhHHhhhhhhhhh')

    _raw_time_seconds:  int = field(repr=False)
    _raw_offset_ms:     int = field(repr=False)

    _reserved01: int = field(repr=False)
    _reserved02: int = field(repr=False)
    _reserved03: int = field(repr=False)
    _reserved04: int = field(repr=False)

    _data_validity_flags:        int = field(repr=False)

    _raw_bottom_dist1:  int = field(repr=False)
    _raw_bottom_dist2:  int = field(repr=False)
    _raw_bottom_dist3:  int = field(repr=False)
    _raw_bottom_dist4:  int = field(repr=False)
    _raw_bottom_vel_x:  int = field(repr=False)
    _raw_bottom_vel_y:  int = field(repr=False)
    _raw_bottom_vel_z:  int = field(repr=False)
    _raw_water_vel_x:   int = field(repr=False)
    _raw_water_vel_y:   int = field(repr=False)
    _raw_water_vel_z:   int = field(repr=False)

    _raw_depth_from_depth_sensor: int = field(repr=False)

    _raw_pitch:         int = field(repr=False) # 0.01 degs (-180 to +180) positive bow up
    _raw_roll:          int = field(repr=False) # 0.01 degs (-180 to +180) positive port up
    _raw_heading:       int = field(repr=False) # 0.01 degs (0 to +360)
    _raw_salinity:      int = field(repr=False) # parts per thousand
    _raw_temperature:   int = field(repr=False) # 1/100 deg Celsius
    _raw_sound_vel:     int = field(repr=False) # m/s

    _reserved05: int = field(repr=False)
    _reserved06: int = field(repr=False)
    _reserved07: int = field(repr=False)
    _reserved09: int = field(repr=False)
    _reserved10: int = field(repr=False)
    _reserved11: int = field(repr=False)
    _reserved12: int = field(repr=False)

    # Derived fields
    timestamp:      float = field(init=False)

    bottom_dist:    Optional[float] = field(init=False, default=None) # meters
    
    bottom_vel_x:   Optional[float] = field(init=False, default=None) # m/s
    bottom_vel_y:   Optional[float] = field(init=False, default=None) # m/s
    bottom_vel_z:   Optional[float] = field(init=False, default=None) # m/s

    water_vel_x:    Optional[float] = field(init=False, default=None) # m/s
    water_vel_y:    Optional[float] = field(init=False, default=None) # m/s
    water_vel_z:    Optional[float] = field(init=False, default=None) # m/s

    sensor_depth:   Optional[float] = field(init=False, default=None) # m

    pitch:          Optional[float] = field(init=False, default=None) # deg (positive bow up)
    roll:           Optional[float] = field(init=False, default=None) # deg (positive port up)
    heading:        Optional[float] = field(init=False, default=None) # deg (0-360)

    salinity:       Optional[float] = field(init=False, default=None) # parts per thousand
    temperature:    Optional[float] = field(init=False, default=None) # deg C
    sound_speed:    Optional[float] = field(init=False, default=None) # m/s

    @staticmethod
    def parse_validity_flags(flags: int) -> dict:
        logging.debug(f"Validity Flags: {bin(flags)}")
        return {
            "xy_velocity":        bool(flags & (1 << 0)),
            "earth_coord":        bool(flags & (1 << 1)),
            "z_velocity":         bool(flags & (1 << 2)),
            "xy_water_velocity":  bool(flags & (1 << 3)),
            "z_water_velocity":   bool(flags & (1 << 4)),
            "bottom_dist":        bool(flags & (1 << 5)),
            "heading":            bool(flags & (1 << 6)),
            "pitch":              bool(flags & (1 << 7)),
            "roll":               bool(flags & (1 << 8)),
            "temp":               bool(flags & (1 << 9)),
            "depth":              bool(flags & (1 << 10)),
            "salinity":           bool(flags & (1 << 11)),
            "sound_velocity":     bool(flags & (1 << 12)),
            "error_detected":     bool(flags & (1 << 13)),
            # Rest unused, presently 9
        }

    def __post_init__(self) -> None:

        self.timestamp = self._raw_time_seconds + self._raw_offset_ms/1e3

        validity_flags = self.parse_validity_flags(self._data_validity_flags)
        if validity_flags['error_detected']:    # FIXME: not sure what to do with this
            logging.warning(f"DVL Error detected")
            return
        

        if validity_flags['earth_coord']:   # FIXME: what should I do with these
            logging.debug("DVL uses earth coordinates")
        else:
            logging.debug("DVL uses ship coordinates")

        if validity_flags['bottom_dist']:
            self.bottom_dist = np.array([
                self._raw_bottom_dist1,
                self._raw_bottom_dist2,
                self._raw_bottom_dist3,
                self._raw_bottom_dist4,
            ]).mean() / 1e2 # cm -> m # ???: not sure if this is right
            # FIXME: this should be based fitting a plane

        logging.debug((f'Botom Dist = {np.array([
                self._raw_bottom_dist1,
                self._raw_bottom_dist2,
                self._raw_bottom_dist3,
                self._raw_bottom_dist4,                
            ]) / 1e2}'))
        logging.debug(f'Bottom Dist (mean) = {self.bottom_dist}')
        
        # FIXME: this is in utils but not used
        # beam_results = calculate_dvl_altitude_plane_fit(
        #     np.array([
        #         self._raw_bottom_dist1,
        #         self._raw_bottom_dist2,
        #         self._raw_bottom_dist3,
        #         self._raw_bottom_dist4,                
        #     ]) / 1e2 # cm -> m
        # )


        # if validity_flags['bottom_dist']:
        #     self.bottom_dist = beam_results['altitude']

        logging.debug(f'Bottom Dist (fit) = {self.bottom_dist}')

        # NOTE: I think that the DVL is outputing the correct bottom distance
        # because all 4 beams are always the same and the pitch/roll is always
        # 0, so it must not be working in a traditional way.


        if validity_flags['xy_velocity']:
            self.bottom_vel_x = self._raw_bottom_vel_x / 1e3 # mm/s -> m/s
            self.bottom_vel_y = self._raw_bottom_vel_y / 1e3 # mm/s -> m/s

        if validity_flags['z_velocity']:
            self.bottom_vel_z = self._raw_bottom_vel_z / 1e3 # mm/s -> m/s

        if validity_flags['xy_water_velocity']:
            self.water_vel_x = self._raw_water_vel_x / 1e3 # mm/s -> m/s
            self.water_vel_y = self._raw_water_vel_y / 1e3 # mm/s -> m/s

        if validity_flags['z_water_velocity']:
            self.water_vel_z = self._raw_water_vel_z / 1e3 # mm/s -> m/s

        if validity_flags['depth']:
            self.sensor_depth = self._raw_depth_from_depth_sensor / 10 # dm -> m

        if validity_flags['pitch']:
            self.pitch = self._raw_pitch * 0.01 # -> deg (positive bow up)
        if validity_flags['roll']:
            self.roll = self._raw_pitch * 0.01  # -> deg (positive port up)

        logging.debug(f"Pitch: {(self.pitch, beam_results['pitch'])}")
        logging.debug(f"Roll: {(self.roll, beam_results['roll'])}")

        if validity_flags['heading']:
            self.heading = self._raw_heading * 0.01 # -> deg (0-360)

        if validity_flags['salinity']:
            self.salinity = self._raw_salinity  # parts per thousand

        if validity_flags['temp']:
            self.temperature = self._raw_temperature / 100 # deg C

        if validity_flags['sound_velocity']:
            self.sound_speed = self._raw_sound_vel*1.0 # m/s

    @classmethod
    def from_bytes(cls, data: bytes) -> "DopplerVelocityLogDataMessage":
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
@dataclass
class SituationMessage(JSFMessageBody): # TODO
    MESSAGE_TYPE: ClassVar[int] = 2090
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<B')

    @classmethod
    def from_bytes(cls, data: bytes) -> "SituationMessage":
        raise NotImplementedError 
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
@dataclass
class SituationComprehensiveMessage(JSFMessageBody):    # TODO
    MESSAGE_TYPE: ClassVar[int] = 2091
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<B')

    @classmethod
    def from_bytes(cls, data: bytes) -> "SituationComprehensiveMessage":
        raise NotImplementedError
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
@dataclass
class CableCounterDataMessage(JSFMessageBody):  # TODO
    MESSAGE_TYPE: ClassVar[int] = 2100
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<B')

    @classmethod
    def from_bytes(cls, data: bytes) -> "CableCounterDataMessage":
        raise NotImplementedError
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
@dataclass
class KilometerOfPipeDataMessage(JSFMessageBody):   # TODO
    MESSAGE_TYPE: ClassVar[int] = 2101
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<B')

    @classmethod
    def from_bytes(cls, data: bytes) -> "KilometerOfPipeDataMessage":
        raise NotImplementedError
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))

@dataclass
class ContainerTimeStampMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 2111
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiBBBB')

    _raw_time_seconds:  int = field(repr=False) # seconds since 1970
    _raw_offset_ms:     int = field(repr=False) # milliseconds within second

    _reserved01:    int = field(repr=False)
    _reserved02:    int = field(repr=False)
    _reserved03:    int = field(repr=False)
    _reserved04:    int = field(repr=False)

    # Converted fields
    timestamp:    float = field(init=False) # seconds since 1970

    def __post_init__(self) -> None:
        self.timestamp = self._raw_time_seconds + self._raw_offset_ms/1e3

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContainerTimeStampMessage":
        values = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*values)
    

@dataclass
class UnknownMessage2040(JSFMessageBody):
    # HACK: I cannot find any references to this message type in any
    # documentation. I do not know what it references, but I am assuming it is
    # not critical. This placeholder assure the does not pop an error.
    MESSAGE_TYPE: ClassVar[int] = 2040
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<iiiiiiiii')

    raw_0: int = field(repr=False)
    raw_1: int = field(repr=False)
    raw_2: int = field(repr=False)
    raw_3: int = field(repr=False)
    raw_4: int = field(repr=False)
    raw_5: int = field(repr=False)
    raw_6: int = field(repr=False)
    raw_7: int = field(repr=False)
    raw_8: int = field(repr=False)

    @classmethod
    def from_bytes(cls, data: bytes) -> "UnknownMessage2040":
        logging.debug("Unknown message type 2040 ignored!")
        values = cls.STRUCT.unpack(data[:cls.STRUCT.size])
        return cls(*values)
    
    def __repr__(self) -> str:
        """Print raw values in hex and decimal for manual inspection"""
        values = [self.raw_0, self.raw_1, self.raw_2, self.raw_3, self.raw_4, 
                  self.raw_5, self.raw_6, self.raw_7, self.raw_8]
        hex_str = ' '.join(f'0x{v:08x}' for v in values)
        dec_str = ' '.join(f'{v:12d}' for v in values)
        return f"UnknownMessage2040(\n  HEX: {hex_str}\n  DEC: {dec_str}\n)"
    

@dataclass
class SonarMessageStatusMessage(JSFMessageBody):
    MESSAGE_TYPE: ClassVar[int] = 40
    STRUCT: ClassVar[struct.Struct] = struct.Struct('<II'+'hhhhhhhhhh'+'II'+'BBBBBB'+'H'+'iii'+'hhhhBBBBBBBB'+'iiiiii')

    overflow_count: int                 # UINT32
    error_count: int                    # UINT32
    
    last_error_0: int = field(repr=False)
    last_error_1: int = field(repr=False)
    last_error_2: int = field(repr=False)
    last_error_3: int = field(repr=False)
    last_error_4: int = field(repr=False)
    last_error_5: int = field(repr=False)
    last_error_6: int = field(repr=False)
    last_error_7: int = field(repr=False)
    last_error_8: int = field(repr=False)
    last_error_9: int = field(repr=False)
    
    free_disk_space: int                    # UINT32, kilobytes
    data_acquisition_indicator: int         # UINT32
    
    service_needed: int                     # UINT8, bitmap
    reserved_37: int = field(repr=False)    # UINT8
    reserved_38: int = field(repr=False)    # UINT8
    storage_flags: int                      # UINT8, bitmap
    temperature_status: int                 # UINT8, bitmap
    time_status: int                        # UINT8, bitmap

    reserved_42_43: int = field(repr=False) # UINT16
    
    bottle_temperature: int             # INT32, Degrees C * 1000
    ambient_temperature: int            # INT32, Degrees C * 1000
    power_48_volts: int                 # INT32, millivolts
    
    reserved_56_59: int = field(repr=False)   
    low_rate_io_0: int = field(repr=False)
    low_rate_io_1: int = field(repr=False)
    low_rate_io_2: int = field(repr=False)
    low_rate_io_3: int = field(repr=False)
    
    serial_port_state_0: int = field(repr=False)
    serial_port_state_1: int = field(repr=False)
    serial_port_state_2: int = field(repr=False)
    serial_port_state_3: int = field(repr=False)
    serial_port_state_4: int = field(repr=False)
    serial_port_state_5: int = field(repr=False)
    serial_port_state_6: int = field(repr=False)
    serial_port_state_7: int = field(repr=False)
    
    runtime_alerts: int                 # INT32, bitmap
    reserved_80: int = field(repr=False)
    reserved_84: int = field(repr=False)
    reserved_88: int = field(repr=False)
    reserved_92: int = field(repr=False)
    # reserved_96: int = field(repr=False)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SonarMessageStatusMessage":
        return cls(*cls.STRUCT.unpack(data[:cls.STRUCT.size]))
    
    if __name__ == '__main__':
        logging.basicConfig(level=logging.DEBUG)
        
        # Simple message instantiation tests
        print("Testing JSF Message Classes...")
        
        # Test SonarDataMessage
        test_data = b'\x00' * 100  # Placeholder data
        try:
            msg = SonarDataMessage.from_bytes(test_data)
            print(f"✓ SonarDataMessage: {type(msg)}")
        except Exception as e:
            print(f"✗ SonarDataMessage: {e}")
        
        # Test SystemInformationMessage
        test_sys_data = struct.pack('<iiiiii', 1, 0, 100, 2, 1, 12345)
        try:
            msg = SystemInformationMessage.from_bytes(test_sys_data)
            print(f"✓ SystemInformationMessage: system_type={msg.system_type}")
        except Exception as e:
            print(f"✗ SystemInformationMessage: {e}")
        
        # TODO: test other messages?
        
        # ???: is this test correct without real data?

        print("Message class verification complete!")