"""
XTF binary structures as numpy structured dtypes.

Field names, order and sizes follow the eXtended Triton Format
specification, revision 42 (Exail), and agree field for field with the
ctypes structures in pyxtf 1.5.0 (checked by the unit tests). All
structures are little-endian and packed (no alignment padding); each
dtype's ``itemsize`` is asserted at import.

Structures
----------
- ``CHAN_INFO_DTYPE`` (128 bytes): one channel description in the file
  header (``CHANINFO``).
- ``FILE_HEADER_DTYPE`` (1024 bytes): the file header (``XTFFILEHEADER``)
  including its first six ``ChanInfo`` blocks.
- ``PACKET_START_DTYPE`` (14 bytes): the fields every packet starts with
  (magic ``0xFACE``, header type, sub-channel, channels to follow,
  reserved, total packet byte count).
- ``PING_HEADER_DTYPE`` (256 bytes): the sonar ping header
  (``XTFPINGHEADER``); its first 14 bytes are the packet start fields.
- ``PING_CHAN_HEADER_DTYPE`` (64 bytes): the per-channel header
  (``XTFPINGCHANHEADER``) that precedes each channel's samples.

Units as given by the specification (the reader does not convert):
times in seconds, ranges in metres, ``SensorDepth`` /
``SensorPrimaryAltitude`` in metres, angles in degrees,
``ShipAltitude`` / ``ShipDepth`` in decimetres, coordinates in the units
named by the file header's ``NavUnits`` (0 = projected metres,
3 = degrees with X = longitude, Y = latitude). Sound-velocity fields may
hold half the true value (some recorders store the one-way factor
``c / 2``).

2026 rshomberg@uri.edu
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Magic number at the start of every packet.
PACKET_MAGIC = 0xFACE
#: Value of ``FileFormat`` (first byte of the file) for XTF files.
FILE_FORMAT_XTF = 0x7B

#: Size of one file-header block; files with more than six channels
#: append further blocks holding the remaining ``ChanInfo`` entries.
FILE_HEADER_BLOCK_BYTES = 1024
#: Number of ``ChanInfo`` entries inside the first file-header block.
CHAN_INFO_IN_FIRST_BLOCK = 6

# Header types (``HeaderType`` in the packet start), from the spec.
HEADER_TYPE_SONAR = 0
HEADER_TYPE_NOTES = 1

#: Names of the packet header types defined by the specification.
HEADER_TYPE_NAMES = {
    0: "sonar",
    1: "notes",
    2: "bathy",
    3: "attitude",
    4: "forward",
    5: "elac",
    6: "raw_serial",
    7: "embed_head",
    8: "hidden_sonar",
    9: "seaview_processed_bathy",
    10: "seaview_depths",
    11: "rsvd_highspeed_sensor",
    12: "echostrength",
    13: "georec",
    14: "klein_raw_bathy",
    15: "highspeed_sensor2",
    16: "elac_xse",
    17: "bathy_xyza",
    18: "k5000_bathy_iq",
    19: "bathy_snippet",
    20: "gps",
    21: "gps_statistics",
    22: "single_beam",
    23: "gyro",
    24: "trackpoint",
    25: "multibeam",
    26: "q_singlebeam",
    27: "q_multitx",
    28: "q_multibeam",
    42: "navigation",
    50: "time",
    60: "benthos_caati_sara",
    61: "reson_7125",
    62: "reson_7125_snippet",
    65: "qinsy_r2sonic_bathy",
    66: "qinsy_r2sonic_fts",
    67: "r2sonic_truepix",
    68: "r2sonic_bathy",
    69: "r2sonic_fts",
    70: "coda_echoscope_data",
    71: "coda_echoscope_config",
    72: "coda_echoscope_image",
    73: "edgetech_4600",
    74: "multibeam_raw_beam_angle",
    78: "reson_7018_watercolumn",
    84: "sourcetime_gyro",
    100: "reson_position",
    102: "bathy_proc",
    103: "attitude_proc",
    104: "singlebeam_proc",
    105: "aux_proc",
    106: "klein3000_data_page",
    107: "pos_raw_navigation",
    108: "kleinv4_data_page",
    199: "custom_vendor_data",
    200: "user_defined",
}

# ``TypeOfChannel`` in ChanInfo.
CHANNEL_TYPE_SUBBOTTOM = 0
CHANNEL_TYPE_PORT = 1
CHANNEL_TYPE_STBD = 2
CHANNEL_TYPE_BATHY = 3

# ``NavUnits`` in the file header.
NAV_UNITS_METERS = 0
NAV_UNITS_LATLON = 3

#: ``SampleFormat`` (ChanInfo, added in X41) to sample dtype. Format 0
#: ("legacy") means "use ``BytesPerSample``", see ``LEGACY_SAMPLE_DTYPES``.
#: Formats 1 (IBM float), 4, 6 and 7 are not supported.
SAMPLE_FORMAT_DTYPES = {
    2: np.dtype("<u4"),
    3: np.dtype("<u2"),
    5: np.dtype("<f4"),
    8: np.dtype("u1"),
}
#: ``SampleFormat`` values the XTF specification defines (0-8). Any other
#: value is treated as unset, i.e. as 0 (legacy, ``BytesPerSample``): some
#: KleinXlt 3.0 files have 0xFF in the ChanInfo reserved bytes, which
#: include ``SampleFormat``.
KNOWN_SAMPLE_FORMATS = frozenset(range(9))
#: ``BytesPerSample`` to sample dtype for legacy (``SampleFormat`` 0) data.
#: Samples are read as unsigned integers, as pyxtf does.
LEGACY_SAMPLE_DTYPES = {
    1: np.dtype("u1"),
    2: np.dtype("<u2"),
    4: np.dtype("<u4"),
}


def _packed(fields: list[tuple], itemsize: int) -> np.dtype:
    """Build a packed little-endian dtype and check its size."""
    dt = np.dtype(fields)
    if dt.itemsize != itemsize:  # pragma: no cover - guarded at import
        raise AssertionError(f"dtype size {dt.itemsize} != {itemsize}")
    return dt


CHAN_INFO_DTYPE = _packed(
    [
        ("TypeOfChannel", "u1"),
        ("SubChannelNumber", "u1"),
        ("CorrectionFlags", "<u2"),
        ("UniPolar", "<u2"),
        ("BytesPerSample", "<u2"),
        ("Reserved", "<u4"),  # formerly SamplesPerChannel
        ("ChannelName", "S16"),
        ("VoltScale", "<f4"),
        ("Frequency", "<f4"),
        ("HorizBeamAngle", "<f4"),
        ("TiltAngle", "<f4"),
        ("BeamWidth", "<f4"),
        ("OffsetX", "<f4"),
        ("OffsetY", "<f4"),
        ("OffsetZ", "<f4"),
        ("OffsetYaw", "<f4"),
        ("OffsetPitch", "<f4"),
        ("OffsetRoll", "<f4"),
        ("BeamsPerArray", "<u2"),
        ("SampleFormat", "u1"),
        ("ReservedArea2", "u1", (53,)),
    ],
    128,
)

FILE_HEADER_DTYPE = _packed(
    [
        ("FileFormat", "u1"),
        ("SystemType", "u1"),
        ("RecordingProgramName", "S8"),
        ("RecordingProgramVersion", "S8"),
        ("SonarName", "S16"),
        ("SonarType", "<u2"),
        ("NoteString", "S64"),
        ("ThisFileName", "S64"),
        ("NavUnits", "<u2"),
        ("NumberOfSonarChannels", "<u2"),
        ("NumberOfBathymetryChannels", "<u2"),
        ("NumberOfSnippetChannels", "u1"),
        ("NumberOfForwardLookArrays", "u1"),
        ("NumberOfEchoStrengthChannels", "<u2"),
        ("NumberOfInterferometryChannels", "u1"),
        ("Reserved1", "u1"),
        ("Reserved2", "<u2"),
        ("ReferencePointHeight", "<f4"),
        ("ProjectionType", "u1", (12,)),
        ("SpheriodType", "u1", (10,)),  # sic, spelled so in the spec
        ("NavigationLatency", "<i4"),
        ("OriginY", "<f4"),
        ("OriginX", "<f4"),
        ("NavOffsetY", "<f4"),
        ("NavOffsetX", "<f4"),
        ("NavOffsetZ", "<f4"),
        ("NavOffsetYaw", "<f4"),
        ("MRUOffsetY", "<f4"),
        ("MRUOffsetX", "<f4"),
        ("MRUOffsetZ", "<f4"),
        ("MRUOffsetYaw", "<f4"),
        ("MRUOffsetPitch", "<f4"),
        ("MRUOffsetRoll", "<f4"),
        ("ChanInfo", CHAN_INFO_DTYPE, (CHAN_INFO_IN_FIRST_BLOCK,)),
    ],
    FILE_HEADER_BLOCK_BYTES,
)

_PACKET_START_FIELDS = [
    ("MagicNumber", "<u2"),
    ("HeaderType", "u1"),
    ("SubChannelNumber", "u1"),
    ("NumChansToFollow", "<u2"),
    ("Reserved1", "<u2", (2,)),
    ("NumBytesThisRecord", "<u4"),
]

PACKET_START_DTYPE = _packed(_PACKET_START_FIELDS, 14)

PING_HEADER_DTYPE = _packed(
    _PACKET_START_FIELDS
    + [
        ("Year", "<u2"),
        ("Month", "u1"),
        ("Day", "u1"),
        ("Hour", "u1"),
        ("Minute", "u1"),
        ("Second", "u1"),
        ("HSeconds", "u1"),  # hundredths of a second
        ("JulianDay", "<u2"),
        ("EventNumber", "<u4"),
        ("PingNumber", "<u4"),
        ("SoundVelocity", "<f4"),  # may be halved
        ("OceanTide", "<f4"),
        ("Reserved2", "<u4"),
        ("ConductivityFreq", "<f4"),
        ("TemperatureFreq", "<f4"),
        ("PressureFreq", "<f4"),
        ("PressureTemp", "<f4"),
        ("Conductivity", "<f4"),
        ("WaterTemperature", "<f4"),
        ("Pressure", "<f4"),
        ("ComputedSoundVelocity", "<f4"),
        ("MagX", "<f4"),
        ("MagY", "<f4"),
        ("MagZ", "<f4"),
        ("AuxVal1", "<f4"),
        ("AuxVal2", "<f4"),
        ("AuxVal3", "<f4"),
        ("AuxVal4", "<f4"),
        ("AuxVal5", "<f4"),
        ("AuxVal6", "<f4"),
        ("SpeedLog", "<f4"),
        ("Turbidity", "<f4"),
        ("ShipSpeed", "<f4"),
        ("ShipGyro", "<f4"),
        ("ShipYcoordinate", "<f8"),
        ("ShipXcoordinate", "<f8"),
        ("ShipAltitude", "<u2"),  # decimetres
        ("ShipDepth", "<u2"),  # decimetres
        ("FixTimeHour", "u1"),
        ("FixTimeMinute", "u1"),
        ("FixTimeSecond", "u1"),
        ("FixTimeHsecond", "u1"),
        ("SensorSpeed", "<f4"),
        ("KP", "<f4"),
        ("SensorYcoordinate", "<f8"),
        ("SensorXcoordinate", "<f8"),
        ("SonarStatus", "<u2"),
        ("RangeToFish", "<u2"),
        ("BearingToFish", "<u2"),
        ("CableOut", "<u2"),
        ("Layback", "<f4"),
        ("CableTension", "<f4"),
        ("SensorDepth", "<f4"),
        ("SensorPrimaryAltitude", "<f4"),
        ("SensorAuxAltitude", "<f4"),
        ("SensorPitch", "<f4"),
        ("SensorRoll", "<f4"),
        ("SensorHeading", "<f4"),
        ("Heave", "<f4"),
        ("Yaw", "<f4"),
        ("AttitudeTimeTag", "<u4"),
        ("DOT", "<f4"),
        ("NavFixMilliseconds", "<u4"),
        ("ComputerClockHour", "u1"),
        ("ComputerClockMinute", "u1"),
        ("ComputerClockSecond", "u1"),
        ("ComputerClockHsec", "u1"),
        ("FishPositionDeltaX", "<i2"),
        ("FishPositionDeltaY", "<i2"),
        ("FishPositionErrorCode", "u1"),
        ("OptionalOffset", "<u4"),
        ("CableOutHundredths", "u1"),
        ("ReservedSpace2", "u1", (6,)),
    ],
    256,
)

PING_CHAN_HEADER_DTYPE = _packed(
    [
        ("ChannelNumber", "<u2"),
        ("DownsampleMethod", "<u2"),
        ("SlantRange", "<f4"),  # metres
        ("GroundRange", "<f4"),  # metres, optional
        ("TimeDelay", "<f4"),  # seconds, transmit to start of recording
        ("TimeDuration", "<f4"),  # seconds of recorded data
        ("SecondsPerPing", "<f4"),
        ("ProcessingFlags", "<u2"),
        ("Frequency", "<u2"),
        ("InitialGainCode", "<u2"),
        ("GainCode", "<u2"),
        ("BandWidth", "<u2"),
        ("ContactNumber", "<u4"),
        ("ContactClassification", "<u2"),
        ("ContactSubNumber", "u1"),
        ("ContactType", "u1"),
        ("NumSamples", "<u4"),
        ("MillivoltScale", "<u2"),
        ("ContactTimeOffTrack", "<f4"),
        ("ContactCloseNumber", "u1"),
        ("Reserved2", "u1"),
        ("FixedVSOP", "<f4"),
        ("Weight", "<i2"),
        ("ReservedSpace", "u1", (4,)),
    ],
    64,
)


def sample_dtype(chan_info: np.void) -> np.dtype:
    """Return the sample dtype described by one ``ChanInfo`` record.

    ``SampleFormat`` (X41 and later) takes precedence; format 0 (legacy)
    falls back to ``BytesPerSample``. A format that disagrees with
    ``BytesPerSample`` is used as given. A value outside
    :data:`KNOWN_SAMPLE_FORMATS` (such as 255) is treated as unset, like 0;
    this logs at DEBUG only, and :func:`~sidescanpy.io.xtf.read_xtf_pings`
    warns once per file.

    Parameters
    ----------
    chan_info : numpy.void
        One record of ``CHAN_INFO_DTYPE``.

    Returns
    -------
    numpy.dtype

    Raises
    ------
    ValueError
        If the sample format or byte width is not supported (IBM float,
        formats 4, 6 and 7, or a legacy width other than 1, 2 or 4 bytes).
    """
    fmt = int(chan_info["SampleFormat"])
    bps = int(chan_info["BytesPerSample"])
    if fmt not in KNOWN_SAMPLE_FORMATS:
        logger.debug("XTF ChanInfo SampleFormat %d is not defined; using BytesPerSample %d", fmt, bps)
        fmt = 0
    if fmt in SAMPLE_FORMAT_DTYPES:
        dt = SAMPLE_FORMAT_DTYPES[fmt]
        if bps and bps != dt.itemsize:
            logger.warning(
                "XTF ChanInfo SampleFormat %d (%d bytes) disagrees with "
                "BytesPerSample %d; using SampleFormat",
                fmt, dt.itemsize, bps,
            )
        return dt
    if fmt == 0:
        if bps in LEGACY_SAMPLE_DTYPES:
            return LEGACY_SAMPLE_DTYPES[bps]
        raise ValueError(f"Unsupported XTF BytesPerSample {bps} (legacy SampleFormat)")
    raise ValueError(f"Unsupported XTF SampleFormat {fmt}")
