"""
Synthetic XTF file builders for the XTF reader tests.

Files are assembled from the reader's own numpy dtypes (``structs.py``);
``test_structs.py`` checks those dtypes field for field against pyxtf,
and a few reader tests read the same synthetic files back with pyxtf.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidescanpy.io.xtf.structs import (
    CHAN_INFO_DTYPE,
    CHAN_INFO_IN_FIRST_BLOCK,
    FILE_HEADER_DTYPE,
    PACKET_MAGIC,
    PACKET_START_DTYPE,
    PING_CHAN_HEADER_DTYPE,
    PING_HEADER_DTYPE,
)

PORT, STBD = 1, 2


def chan_info(
    type_of_channel: int,
    frequency: float,
    bytes_per_sample: int = 2,
    sample_format: int = 0,
    name: str = "",
    reserved: int = 1024,
) -> np.ndarray:
    """One ChanInfo record."""
    ci = np.zeros((), dtype=CHAN_INFO_DTYPE)
    ci["TypeOfChannel"] = type_of_channel
    ci["BytesPerSample"] = bytes_per_sample
    ci["SampleFormat"] = sample_format
    ci["Frequency"] = frequency
    ci["ChannelName"] = name.encode()
    ci["Reserved"] = reserved
    return ci


def dual_frequency_infos(bps: int = 2, fmt: int = 0) -> list[np.ndarray]:
    """Port/stbd at 100 and port/stbd at 400 (kHz), channels 0..3."""
    return [
        chan_info(PORT, 100.0, bps, fmt, "PORT_LOW"),
        chan_info(STBD, 100.0, bps, fmt, "STBD_LOW"),
        chan_info(PORT, 400.0, bps, fmt, "PORT_HI"),
        chan_info(STBD, 400.0, bps, fmt, "STBD_HI"),
    ]


def file_header(infos: list[np.ndarray], nav_units: int = 3, **fields) -> bytes:
    """File header bytes; channels beyond six spill into extra 1024-byte blocks."""
    fh = np.zeros((), dtype=FILE_HEADER_DTYPE)
    fh["FileFormat"] = 0x7B
    fh["SystemType"] = 1
    fh["RecordingProgramName"] = b"synth"
    fh["SonarName"] = b"synthetic"
    fh["SonarType"] = 38
    fh["NavUnits"] = nav_units
    fh["NumberOfSonarChannels"] = len(infos)
    for k, v in fields.items():
        fh[k] = v
    for i, ci in enumerate(infos[:CHAN_INFO_IN_FIRST_BLOCK]):
        fh["ChanInfo"][i] = ci
    out = fh.tobytes()
    extra = infos[CHAN_INFO_IN_FIRST_BLOCK:]
    if extra:
        blob = b"".join(np.asarray(ci).tobytes() for ci in extra)
        pad = -len(blob) % 1024
        out += blob + b"\x00" * pad
    return out


def packet(header_type: int, body: bytes = b"", nbytes: int | None = None) -> bytes:
    """Generic packet: packet start + body."""
    start = np.zeros((), dtype=PACKET_START_DTYPE)
    start["MagicNumber"] = PACKET_MAGIC
    start["HeaderType"] = header_type
    start["NumBytesThisRecord"] = (
        PACKET_START_DTYPE.itemsize + len(body) if nbytes is None else nbytes
    )
    return start.tobytes() + body


def ping_header(ping_number: int = 0, **fields) -> np.ndarray:
    """A ping header record with a valid date and position."""
    ph = np.zeros((), dtype=PING_HEADER_DTYPE)
    ph["MagicNumber"] = PACKET_MAGIC
    ph["HeaderType"] = 0
    ph["Year"], ph["Month"], ph["Day"] = 2005, 3, 14
    ph["Hour"], ph["Minute"] = 5, 54
    ph["Second"] = ping_number % 60
    ph["HSeconds"] = (ping_number * 7) % 100
    ph["PingNumber"] = ping_number
    ph["SoundVelocity"] = 750.0
    ph["SensorXcoordinate"] = -73.7 + ping_number * 1e-5
    ph["SensorYcoordinate"] = -46.3 - ping_number * 1e-5
    ph["SensorHeading"] = 123.5
    ph["SensorDepth"] = 17.25
    ph["SensorPrimaryAltitude"] = 8.5
    for k, v in fields.items():
        ph[k] = v
    return ph


def sonar_packet(
    channels: list[tuple[int, np.ndarray]],
    ping_number: int = 0,
    num_chans_to_follow: int | None = None,
    pad: int = 0,
    nbytes: int | None = None,
    chan_fields: dict | None = None,
    num_samples: dict[int, int] | None = None,
    **ping_fields,
) -> bytes:
    """Sonar packet with one (channel_number, samples) block per channel."""
    body = b""
    for number, samples in channels:
        ch = np.zeros((), dtype=PING_CHAN_HEADER_DTYPE)
        ch["ChannelNumber"] = number
        ch["SlantRange"] = 100.0
        ch["TimeDuration"] = 100.0 / 750.0
        ch["NumSamples"] = (num_samples or {}).get(number, len(samples))
        for k, v in (chan_fields or {}).items():
            ch[k] = v
        body += ch.tobytes() + np.ascontiguousarray(samples).tobytes()
    body += b"\x00" * pad
    ph = ping_header(ping_number, **ping_fields)
    ph["NumChansToFollow"] = len(channels) if num_chans_to_follow is None else num_chans_to_follow
    ph["NumBytesThisRecord"] = PING_HEADER_DTYPE.itemsize + len(body) if nbytes is None else nbytes
    return ph.tobytes() + body


def notes_packet(text: bytes = b"synthetic note") -> bytes:
    """Notes packet (type 1), 256 bytes like the spec's XTFNOTESHEADER."""
    body = bytearray(256 - PACKET_START_DTYPE.itemsize)
    body[42 : 42 + len(text)] = text
    return packet(1, bytes(body))


def ramp(n: int, dtype, offset: int = 0) -> np.ndarray:
    """Deterministic distinguishable samples."""
    return ((np.arange(n) * 7 + offset * 131) % 251 + 1).astype(dtype)


def write_xtf(path: Path, header: bytes, packets: list[bytes]) -> Path:
    path.write_bytes(header + b"".join(packets))
    return path
