"""
Synthetic Humminbird recordings, byte for byte.

Builds the 64-byte (Helix) ``.DAT`` header and ``.SON`` ping frames with
the tagged header layout described in
:mod:`sidescanpy.io.humminbird.reader`: magic ``C0 DE AB 21``, then
1-byte tags with 1-byte (tags < 0x80) or 4-byte big-endian values, the
sample count (tag 160), the end byte ``0x21`` and 8-bit samples.

2026 rshomberg@uri.edu
"""

import struct
from pathlib import Path

import numpy as np

from sidescanpy.io.humminbird import lonlat_to_humminbird

START_UNIX = 1_657_213_366  # 2022-07-07T17:02:46Z
LON, LAT = -147.9, 64.8

# The 72-byte header's tags, in file order (the 67-byte one lacks 134).
TAGS_72 = (128, 129, 130, 131, 132, 133, 134, 135, 80, 81, 146, 83, 84, 149, 86, 87, 160)
TAGS_67 = tuple(t for t in TAGS_72 if t != 134)


def dat_bytes(*, unix_time=START_UNIX, water_code=0, name="R00001.SON", n_records=0, ms=0) -> bytes:
    e, n = lonlat_to_humminbird(LON, LAT)
    b = bytearray(64)
    b[0], b[1], b[2], b[3] = 0xC1, water_code, 0x70, 0x01
    struct.pack_into(">I", b, 4, 670)
    struct.pack_into(">I", b, 20, unix_time)
    struct.pack_into(">i", b, 24, int(round(float(e))))
    struct.pack_into(">i", b, 28, int(round(float(n))))
    b[32:42] = name.encode()[:10].ljust(10, b"\x00")
    struct.pack_into(">I", b, 44, n_records)
    struct.pack_into(">I", b, 48, ms)
    return bytes(b)


def frame(*, beam, record, time_ms, samples, freq=455_000, depth_dm=30, heading_ddeg=900,
          speed_dms=15, lon=LON, lat=LAT, tags=TAGS_72) -> bytes:
    e, n = lonlat_to_humminbird(lon, lat)
    values = {
        128: record, 129: time_ms, 130: int(round(float(e))), 131: int(round(float(n))),
        132: (1 << 16) | heading_ddeg, 133: (1 << 16) | speed_dms, 134: 0, 135: depth_dm,
        80: beam, 81: 31, 146: freq, 83: 8, 84: 1, 149: 26, 86: 20, 87: 40,
        160: len(samples),
    }
    out = bytearray(b"\xc0\xde\xab\x21")
    for t in tags:
        out.append(t)
        if t & 0x80:
            out += struct.pack(">i" if t in (130, 131) else ">I", values[t])
        else:
            out.append(values[t])
    out.append(0x21)
    out += np.asarray(samples, dtype=np.uint8).tobytes()
    return bytes(out)


def ramp(n, start=0) -> np.ndarray:
    """``n`` samples rising from ``start`` (mod 256): sample 0 is the nearest."""
    return (np.arange(n) + start) % 256


def write_recording(
    directory: Path,
    stem: str = "R00001",
    *,
    n_pings: int = 5,
    n_samples=100,
    beams=(2, 3),
    stbd_time_offset_ms=0,
    tags=TAGS_72,
    freq=455_000,
    extra_frames: dict | None = None,
    son_tail: dict | None = None,
) -> Path:
    """Write ``<stem>.DAT`` and ``<stem>/B00n.SON`` and return the DAT path.

    Ping ``i`` is at ``i * 50`` ms; port samples start at ``i``,
    starboard at ``100 + i``, down beams at ``200 + i``. ``n_samples``
    may be a list (one count per ping).
    """
    directory = Path(directory)
    son_dir = directory / stem
    son_dir.mkdir(parents=True, exist_ok=True)
    counts = [n_samples] * n_pings if np.isscalar(n_samples) else list(n_samples)
    record = 0
    frames = {b: bytearray() for b in beams}
    for i in range(n_pings):
        for b in beams:
            start = {2: i, 3: 100 + i}.get(b, 200 + i)
            t = i * 50 + (stbd_time_offset_ms if b == 3 else 0)
            frames[b] += frame(beam=b, record=record, time_ms=t, samples=ramp(counts[i], start),
                               freq=freq if b in (2, 3) else 200_000, depth_dm=30 + i,
                               tags=tags)
            record += 1
    for b, extra in (extra_frames or {}).items():
        frames[b] += extra
    for b, tail in (son_tail or {}).items():
        frames[b] += tail
    for b, data in frames.items():
        (son_dir / f"B{b:03d}.SON").write_bytes(bytes(data))
    dat = directory / f"{stem}.DAT"
    dat.write_bytes(dat_bytes(n_records=record, ms=(n_pings - 1) * 50))
    return dat
