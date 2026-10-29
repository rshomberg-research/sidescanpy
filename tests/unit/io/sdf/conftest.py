"""
Shared fixtures and helpers for SDF unit tests.

The synthetic SDF header bytes builder constructs 512 bytes (the typed
header, excluding the 4-byte marker) with all 100+ named fields plus
the 26 reserved uint32s. Tests use this to construct headers with
known values and verify parsing.

2026 rshomberg@uri.edu
"""

import struct

# ============================================================
# CONSTANTS — Klein SDF page layout (per spec)
# ============================================================

MARKER = 0xFFFFFFFF
MARKER_BYTES = 4
PAGE_HEADER_BYTES = 512                         # excluding marker
HEADER_TOTAL_BYTES = MARKER_BYTES + PAGE_HEADER_BYTES   # 516

# Header is 128 uint32-equivalent slots (some are float64 = 2 slots)
HEADER_SLOT_COUNT = 128


# ============================================================
# HEADER BYTES BUILDER (512 bytes, no marker)
# ============================================================

def make_sdf_header_bytes(
    *,
    # Fields 1-26 (uint32)
    page_size: int = 27904,
    page_version: int = 3502,
    configuration: int = 0,
    ping_number: int = 1000,
    num_samples: int = 2500,
    beams_to_display: int = 0,
    error_flags: int = 0,
    range_m: int = 60,
    speed_cms: int = 200,
    sound_speed_cms: int = 0,
    res_mode: int = 0,
    tx_waveform: int = 0,
    responder_divisor: int = 0,
    responder_frequency: int = 0,
    manual_speed_switch: int = 0,
    despeckle_switch: int = 0,
    speed_filter_switch: int = 0,
    year: int = 2026,
    month: int = 5,
    day: int = 13,
    hour: int = 14,
    minute: int = 50,
    second: int = 5,
    hsecond: int = 0,
    fix_time_hour: int = 0,
    fix_time_minute: int = 0,

    # Field 27 (float32)
    fix_time_second: float = 0.0,

    # Fields 28-33 (float32)
    heading_deg: float = 90.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    depth_volts: float = 0.0,
    altitude_m: float = 0.0,
    temperature_c: float = 0.0,

    # Fields 34-36 (float32)
    ship_speed_ms: float = 0.0,
    ship_heading_deg: float = 90.0,
    ship_magnetic_variation: float = 0.0,

    # Fields 37-40 (float64, radians)
    ship_latitude_rad: float = 0.770159,    # ~44.13°N
    ship_longitude_rad: float = -1.202396,  # ~-68.89°W
    fish_latitude_rad: float = 0.0,
    fish_longitude_rad: float = 0.0,

    # Fields 41-45 (uint32)
    tvg_page: int = 0,
    header_size: int = 0,
    fix_time_year: int = 0,
    fix_time_month: int = 0,
    fix_time_day: int = 0,

    # Fields 46-49 (float32)
    aux_pitch: float = 0.0,
    aux_roll: float = 0.0,
    aux_depth: float = 0.0,
    aux_altitude: float = 0.0,

    # Fields 50-51 (float32)
    cable_out_m: float = 0.0,
    fseconds: float = 0.5,

    # Fields 52-55 (uint32)
    altimeter_on: int = 0,
    sample_freq_hz: int = 0,
    depressor_type: int = 0,
    cable_type: int = 0,

    # Fields 56-59 (float32, meters)
    shieve_x_offset_m: float = 0.0,
    shieve_y_offset_m: float = 0.0,
    shieve_z_offset_m: float = 0.0,
    gps_height_m: float = 0.0,

    # Fields 60-67 (uint32)
    raw_data_config: int = 0,
    header_3_extension_size: int = 0,
    sbp_tx_waveform: int = 0,
    sbp_preamp_gain: int = 0,
    sbp_data_raw: int = 0,
    sbp_samples: int = 0,
    sbp_sample_frequency: int = 0,
    sbp_tx_waveform_version: int = 0,

    # Field 68 (float32)
    wing_angle_deg: float = 0.0,

    # Fields 69-70 (uint32)
    emergency_switch_state: int = 0,
    layback_method: int = 0,

    # Fields 71-72 (float64, radians)
    layback_fish_latitude_rad: float = 0.0,
    layback_fish_longitude_rad: float = 0.0,

    # Fields 73-74 (float32)
    fish_heading_offset_deg: float = 0.0,
    pressure_sensor_offset_psi: float = 0.0,

    # Fields 75-79 (uint32)
    tpu_sw_version: int = 0,
    capability_mask: int = 0,
    tx_version: int = 0,
    num_samples_extra: int = 0,
    post_processing_version: int = 0,

    # Fields 80-83 (uint32 holding uint16 values per spec)
    motion_sensor_type: int = 0,
    ping_time_ref_count: int = 0,
    ext_trig_time_ref_count: int = 0,
    one_pps_time_ref_count: int = 0,

    # Field 84 (uint32)
    time_ref_count_weight: int = 0,

    # Fields 85-86 (float32)
    altitude_bathy_m: float = 0.0,
    ping_interval_s: float = 0.0,

    # Field 87 (uint32)
    sdf_extension_size: int = 0,

    # Field 88 (float64)
    seconds_of_week: float = 0.0,

    # Field 89 (uint32)
    sound_speed_source: int = 0,

    # Fields 90-92 (float32)
    pressure_sensor_max: float = 0.0,
    pressure_sensor_voltage_min: float = 0.0,
    pressure_sensor_voltage_max: float = 0.0,

    # Field 93 (uint32)
    # processed_ping_number: int = 0,

    # Fields 94-95 (float32)
    # processed_ping_spacing_m: float = 0.0,
    temperature_ambient_c: float = 0.0,

    # Field 99 (int32 — SIGNED)
    saturation_detect_threshold: int = 0,

    # Field 100 (uint32)
    sonar_freq_khz: int = 455,

    # Reserved fields 101-124 (24 uint32)
    reserved: tuple[int, ...] | None = None,
) -> bytes:
    """Build 512 bytes of SDF header for testing.

    Defaults reflect plausible Klein 4900 survey values.
    Override individual fields for targeted tests.

    The reserved tuple defaults to all zeros if not specified.
    """
    if reserved is None:
        reserved = tuple(0 for _ in range(26))
    if len(reserved) != 26:
        raise ValueError(
            f"reserved must have exactly 26 uint32 values, got {len(reserved)}"
        )

    # Match the STRUCT format exactly:
    # 26I, f, 6f, f, f, f, 4d, 5I, 4f, 2f, 4I, 4f, 8I, f, 2I, 2d, 2f, 5I,
    # 4I, I, 2f, I, d, I, 3f, I, 2f, i, I, 24I

    return struct.pack(
        '<'
        '26I'
        'f'
        '6f'
        'f' 'f' 'f'
        '4d'
        '5I'
        '4f'
        '2f'
        '4I'
        '4f'
        '8I'
        'f'
        '2I'
        '2d'
        '2f'
        '5I'
        '4I'
        'I'
        '2f'
        'I'
        'd'
        'I'
        '3f'
        'f'
        'i'
        'I'
        '26I'
        ,

        # 26I: fields 1-26
        page_size, page_version, configuration, ping_number, num_samples,
        beams_to_display, error_flags, range_m, speed_cms, sound_speed_cms,
        res_mode, tx_waveform, responder_divisor, responder_frequency,
        manual_speed_switch, despeckle_switch, speed_filter_switch,
        year, month, day, hour, minute, second, hsecond,
        fix_time_hour, fix_time_minute,

        # f: field 27
        fix_time_second,

        # 6f: fields 28-33
        heading_deg, pitch_deg, roll_deg, depth_volts, altitude_m, temperature_c,

        # f, f, f: fields 34, 35, 36
        ship_speed_ms, ship_heading_deg, ship_magnetic_variation,

        # 4d: fields 37-40
        ship_latitude_rad, ship_longitude_rad,
        fish_latitude_rad, fish_longitude_rad,

        # 5I: fields 41-45
        tvg_page, header_size, fix_time_year, fix_time_month, fix_time_day,

        # 4f: fields 46-49
        aux_pitch, aux_roll, aux_depth, aux_altitude,

        # 2f: fields 50-51
        cable_out_m, fseconds,

        # 4I: fields 52-55
        altimeter_on, sample_freq_hz, depressor_type, cable_type,

        # 4f: fields 56-59
        shieve_x_offset_m, shieve_y_offset_m, shieve_z_offset_m, gps_height_m,

        # 8I: fields 60-67
        raw_data_config, header_3_extension_size,
        sbp_tx_waveform, sbp_preamp_gain, sbp_data_raw,
        sbp_samples, sbp_sample_frequency, sbp_tx_waveform_version,

        # f: field 68
        wing_angle_deg,

        # 2I: fields 69-70
        emergency_switch_state, layback_method,

        # 2d: fields 71-72
        layback_fish_latitude_rad, layback_fish_longitude_rad,

        # 2f: fields 73-74
        fish_heading_offset_deg, pressure_sensor_offset_psi,

        # 5I: fields 75-79
        tpu_sw_version, capability_mask, tx_version,
        num_samples_extra, post_processing_version,

        # 4I: fields 80-83
        motion_sensor_type, ping_time_ref_count,
        ext_trig_time_ref_count, one_pps_time_ref_count,

        # I: field 84
        time_ref_count_weight,

        # 2f: fields 85-86
        altitude_bathy_m, ping_interval_s,

        # I: field 87
        sdf_extension_size,

        # d: field 88
        seconds_of_week,

        # I: field 89
        sound_speed_source,

        # 3f: fields 90-92
        pressure_sensor_max, pressure_sensor_voltage_min,
        pressure_sensor_voltage_max,

        # I: field 93
        # processed_ping_number,

        # 2f: fields 94-95
        # processed_ping_spacing_m,
        temperature_ambient_c,

        # i: field 99 (SIGNED)
        saturation_detect_threshold,

        # I: field 100
        sonar_freq_khz,

        # 24I: reserved
        *reserved,
    )

import numpy as np  # noqa: E402


def make_sdf_data_section(
    port_data: np.ndarray,
    stbd_data: np.ndarray,
) -> bytes:
    """Build the data section bytes for an SDF sonar page.

    Layout: [n_port][port_data][n_stbd][stbd_data]
    """
    parts = [
        struct.pack("<I", len(port_data)),
        port_data.astype(np.uint32).tobytes(),
        struct.pack("<I", len(stbd_data)),
        stbd_data.astype(np.uint32).tobytes(),
    ]
    return b"".join(parts)


def make_sdf_page_bytes(
    *,
    n_port: int = 1000,
    n_stbd: int = 1000,
    page_version: int = 3502,
    extension_bytes: bytes = b"",
    **header_kwargs,
) -> bytes:
    """Build a complete SDF page (marker + header + body) for testing.

    The page_size is computed automatically based on the data section
    size, accounting for any extension bytes.

    Parameters
    ----------
    n_port : int, default 1000
        Number of port channel samples.
    n_stbd : int, default 1000
        Number of starboard channel samples.
    page_version : int, default 3502
        Page version (3501 = low-freq, 3502 = high-freq).
    extension_bytes : bytes, default b""
        Optional bytes appended after stbd data (data page extension).
    **header_kwargs
        Forwarded to make_sdf_header_bytes.

    Returns
    -------
    bytes
        Complete page (4 marker + 512 header + body), ready to write to a file.
    """
    port_data = np.arange(n_port, dtype=np.uint32)
    stbd_data = np.arange(n_stbd, dtype=np.uint32) + 10000

    data_section = make_sdf_data_section(port_data, stbd_data) + extension_bytes
    body_size = len(data_section)

    # page_size_field is bytes after the marker (header + body)
    page_size = PAGE_HEADER_BYTES + body_size

    # Build the 512-byte header
    header = make_sdf_header_bytes(
        page_version=page_version,
        page_size=page_size,
        num_samples=n_port,
        **header_kwargs,
    )

    # Prepend the marker
    marker = struct.pack("<I", MARKER)

    return marker + header + data_section

import math  # noqa: E402
from datetime import UTC  # noqa: E402


def make_sdf_sonar_page(
    *,
    page_version: int = 3502,
    ping_number: int = 1,
    num_samples: int = 100,
    n_port_samples: int | None = None,
    n_stbd_samples: int | None = None,
    range_m: int = 60,
    speed_cms: int = 200,
    timestamp: float = 1746136205.5,
    heading_deg: float = 90.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    latitude_deg: float = 44.13,
    longitude_deg: float = -68.89,
    altitude_m: float = 5.0,
    fish_latitude_deg: float = 0.0,
    fish_longitude_deg: float = 0.0,
    center_freq_hz: int = 455_000,
    cable_out_m: float = 0.0,
    layback_method: int = 0,
):
    """Build a synthetic SDFSonarPage for testing.

    Constructs the header by:
    1. Calling make_sdf_header_bytes to produce 512 raw bytes
    2. Calling SDFPageHeader.from_bytes to parse them
    3. Wrapping the result in SDFSonarPage with synthetic port/stbd arrays

    This dogfoods the bytes builder + parser, providing free coverage.

    Parameters
    ----------
    n_port_samples, n_stbd_samples : int, optional
        Defaults to num_samples for each. Set differently to test
        ragged data.
    timestamp : float
        UTC epoch seconds.
    latitude_deg, longitude_deg : float
        Ship lat/lon (will be converted to radians for storage).
    fish_latitude_deg, fish_longitude_deg : float
        Towfish lat/lon (used by the .latitude property if non-zero).
    center_freq_hz : int
        Center frequency (will be stored as kHz in the header).
    """
    from datetime import datetime

    from sidescanpy.io.sdf.pages import SDFPageHeader, SDFSonarPage

    if n_port_samples is None:
        n_port_samples = num_samples
    if n_stbd_samples is None:
        n_stbd_samples = num_samples

    # Convert timestamp to date components
    dt = datetime.fromtimestamp(timestamp, tz=UTC)
    fseconds = (timestamp - int(timestamp))

    range_m = int(range_m)
    speed_cms = int(speed_cms)
    num_samples = int(num_samples)

    # Build header bytes
    raw = make_sdf_header_bytes(
        page_version=page_version,
        ping_number=ping_number,
        num_samples=num_samples,
        range_m=range_m,
        speed_cms=speed_cms,
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        fseconds=fseconds,
        heading_deg=heading_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        ship_latitude_rad=math.radians(latitude_deg),
        ship_longitude_rad=math.radians(longitude_deg),
        fish_latitude_rad=math.radians(fish_latitude_deg),
        fish_longitude_rad=math.radians(fish_longitude_deg),
        altitude_m=altitude_m,
        sonar_freq_khz=center_freq_hz // 1000,
        cable_out_m=cable_out_m,
        layback_method=layback_method,
    )

    # Parse it back into a header
    header = SDFPageHeader.from_bytes(raw)

    # Synthetic sample data
    port_data = np.arange(n_port_samples, dtype=np.uint32)
    stbd_data = np.arange(n_stbd_samples, dtype=np.uint32) + 10000

    return SDFSonarPage(
        header=header,
        port_data=port_data,
        stbd_data=stbd_data,
    )
