"""
Read Humminbird side imaging recordings (``.DAT`` + ``.SON``).

A recording is a small ``RNNNNN.DAT`` (or ``RecNNNNN.DAT``) file and a
directory of the same name holding one ``B00n.SON`` file per beam (and a
``B00n.IDX`` index, which is not needed here). Humminbird publishes no
specification; the layout below follows the open readers PyHum and
PINGVerter (PING-Mapper) and was checked on Helix recordings.

``.DAT`` (:func:`read_dat`)
    One header for the recording: water type, start time (Unix seconds),
    first position, the recording name, ping count and duration. Three
    layouts are known, told apart by file size: 64 bytes, big-endian
    (Helix and older units, the one tested); 96 and 100 bytes,
    little-endian (Solix, laid out as in PINGVerter, untested). Onix
    units write an XML ``.DAT``, which is not read.

``.SON`` (:func:`read_son`)
    A sequence of ping frames. Each frame starts with the 4 bytes
    ``C0 DE AB 21``, then a run of tagged big-endian fields: a 1-byte tag
    followed by a 1-byte value when the tag's top bit is clear (tags
    80-87) or a 4-byte value when it is set (tags 128-160). Tag 160 is
    the number of samples; the byte ``0x21`` ends the header; that many
    8-bit samples follow. Headers are 67 bytes on older Helix files and
    72 or 152 bytes on newer ones: the set of tags differs, the encoding
    does not, so the tags are decoded generically. Known tags are named
    in :data:`TAG_NAMES`; others are kept as ``tag_<n>``.

Positions (tags 130 and 131) are easting and northing in metres on a
spherical Mercator projection with the International 1924 radius
(6 378 388 m), the convention PyHum and PINGVerter use;
:func:`humminbird_to_lonlat` converts them.

The reader returns the fields as recorded (tenths of a degree,
decimetres); :mod:`sidescanpy.io.humminbird.converter` converts units
and pairs the port and starboard beams.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Bytes that start every SON ping frame.
FRAME_MAGIC = b"\xc0\xde\xab\x21"
#: Byte that ends a SON ping header.
HEADER_END = 0x21
#: Tag of the sample count, the last field of a SON ping header.
TAG_N_SAMPLES = 160

#: Beam numbers (tag 80) of the side imaging channels.
BEAM_PORT = 2
BEAM_STARBOARD = 3
#: What each beam number holds (names as in PINGVerter).
BEAM_NAMES = {
    0: "down_low_frequency",
    1: "down_high_frequency",
    2: "side_port",
    3: "side_starboard",
    4: "down_imaging",
}

#: Radius (m) of the sphere Humminbird's Mercator positions are on
#: (International 1924 semi-major axis).
MERCATOR_RADIUS_M = 6_378_388.0

#: Names of the SON header fields that are understood. A 4-byte field
#: holding two 2-byte values is split in two (see :data:`SPLIT_TAGS`).
TAG_NAMES: dict[int, str] = {
    80: "beam",
    81: "volt_scale",
    86: "easting_error",
    87: "northing_error",
    128: "record_number",
    129: "time_ms",
    130: "easting",
    131: "northing",
    135: "depth_dm",
    146: "frequency_hz",
    153: "frequency_min_hz",
    154: "frequency_max_hz",
    160: "n_samples",
}
#: Tags whose 4 bytes are two big-endian 16-bit values (high, low).
SPLIT_TAGS: dict[int, tuple[str, str]] = {
    132: ("gps_flag_1", "heading_ddeg"),
    133: ("gps_flag_2", "speed_dms"),
}
#: Tags whose 4-byte value is signed.
SIGNED_TAGS = frozenset({130, 131})

# DAT layouts: name -> (byte order, {field: (offset, struct code)}). The
# recording name is read separately (offset 32, 10 or 12 bytes).
_DAT_FIELDS = {
    "water_code": (1, "B"),
    "sonar_id": (4, "I"),
    "unix_time": (20, "I"),
    "easting": (24, "i"),
    "northing": (28, "i"),
    "n_records": (44, "I"),
    "recording_ms": (48, "I"),
    "line_size": (52, "I"),
}
_DAT_LAYOUTS = {64: (">", 10), 96: ("<", 12), 100: ("<", 12)}
# Water type from the DAT water code, per layout family.
_WATER_TYPES = {
    64: {0: "fresh", 1: "deep salt", 2: "shallow salt"},
    96: {1: "fresh", 2: "shallow salt", 3: "deep salt"},
}


@dataclass(frozen=True)
class HumDat:
    """The ``.DAT`` header of one recording, as recorded."""

    path: Path
    size: int
    water_code: int
    water_type: str
    sonar_id: int
    unix_time: int
    easting: int
    northing: int
    name: str
    n_records: int
    recording_ms: int
    line_size: int


@dataclass
class HumBeam:
    """The ping frames of one ``.SON`` file.

    ``fields`` maps each header field name to a per-ping array (int64);
    a field missing from a ping is -1. ``frame_offset`` is where each
    ping's frame starts in the file; ``sample_offset`` and ``n_samples``
    locate its samples, read with :meth:`samples`.
    """

    path: Path
    beam: int
    fields: dict[str, np.ndarray]
    frame_offset: np.ndarray
    sample_offset: np.ndarray
    n_samples: np.ndarray
    header_sizes: dict[int, int] = field(default_factory=dict)
    n_truncated: int = 0
    n_resyncs: int = 0
    _data: bytes = field(default=b"", repr=False)

    @property
    def n_pings(self) -> int:
        return int(self.n_samples.size)

    def samples(self, index: np.ndarray | None = None) -> np.ndarray:
        """Samples of the pings ``index`` (all by default) as float32
        ``(ping, sample)``, NaN beyond each ping's sample count."""
        idx = np.arange(self.n_pings) if index is None else np.asarray(index)
        width = int(self.n_samples[idx].max()) if idx.size else 0
        out = np.full((idx.size, width), np.nan, dtype=np.float32)
        raw = np.frombuffer(self._data, dtype=np.uint8)
        for row, i in enumerate(idx):
            start, n = int(self.sample_offset[i]), int(self.n_samples[i])
            out[row, :n] = raw[start:start + n]
        return out


@dataclass
class HumRecording:
    """One recording: its ``.DAT`` header and each beam's frames by beam number."""

    dat: HumDat
    beams: dict[int, HumBeam]

    @property
    def path(self) -> Path:
        return self.dat.path


# ============================================================
# PUBLIC API
# ============================================================

def son_directory(dat_path: str | Path) -> Path:
    """The directory of SON files belonging to ``dat_path``.

    It has the DAT file's stem (``R00013.DAT`` -> ``R00013/``), matched
    case-insensitively.

    Raises
    ------
    FileNotFoundError
        If there is no such directory.
    """
    dat_path = Path(dat_path)
    exact = dat_path.with_suffix("")
    if exact.is_dir():
        return exact
    stem = dat_path.stem.lower()
    for p in dat_path.parent.iterdir():
        if p.is_dir() and p.name.lower() == stem:
            return p
    raise FileNotFoundError(
        f"{dat_path}: no SON directory {exact.name}/ next to it; a Humminbird recording "
        "is the .DAT file and a directory of the same name holding B00n.SON files"
    )


def read_dat(path: str | Path) -> HumDat:
    """Read a Humminbird ``.DAT`` header.

    Raises
    ------
    NotImplementedError
        If the file is not one of the known binary layouts (64, 96 or
        100 bytes), such as the XML header Onix units write.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) not in _DAT_LAYOUTS:
        kind = "an XML (Onix) header" if raw.lstrip()[:1] == b"<" or b"<" in raw[:200] else "an unknown layout"
        raise NotImplementedError(
            f"{path}: {len(raw)}-byte .DAT looks like {kind}; only the 64-byte (Helix) and "
            "96/100-byte (Solix) binary headers are read"
        )
    order, name_len = _DAT_LAYOUTS[len(raw)]
    values = {k: struct.unpack_from(order + code, raw, off)[0] for k, (off, code) in _DAT_FIELDS.items()}
    name = raw[32:32 + name_len].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    water_types = _WATER_TYPES[64 if len(raw) == 64 else 96]
    if len(raw) != 64:
        logger.debug("%s: %d-byte (Solix) DAT layout, read as in PINGVerter; not tested on real files",
                     path, len(raw))
    return HumDat(
        path=path,
        size=len(raw),
        water_type=water_types.get(values["water_code"], "unknown"),
        name=name,
        **values,
    )


def read_son(path: str | Path) -> HumBeam:
    """Read every ping frame of one ``.SON`` file.

    A frame cut short by the end of the file is dropped and counted in
    ``n_truncated``. Bytes that do not start a frame where one is due
    are skipped to the next frame start, counted in ``n_resyncs`` and
    logged at WARNING.

    Raises
    ------
    ValueError
        If the file holds no ping frame, or its frames disagree on the
        beam number.
    """
    path = Path(path)
    data = path.read_bytes()
    rows: list[dict[str, int]] = []
    offsets: list[int] = []
    frames: list[int] = []
    header_sizes: dict[int, int] = {}
    n_truncated = n_resyncs = 0
    pos = 0
    size = len(data)
    while pos + len(FRAME_MAGIC) <= size:
        if data[pos:pos + 4] != FRAME_MAGIC:
            nxt = data.find(FRAME_MAGIC, pos + 1)
            n_resyncs += 1
            logger.warning("%s: no ping frame at byte %d; skipping %s bytes to the next frame",
                           path.name, pos, "all remaining" if nxt < 0 else nxt - pos)
            if nxt < 0:
                break
            pos = nxt
            continue
        parsed = _parse_header(data, pos)
        if parsed is None:
            n_truncated += 1
            break
        row, sample_start = parsed
        end = sample_start + row["n_samples"]
        if end > size:
            n_truncated += 1
            break
        header_sizes[sample_start - pos] = header_sizes.get(sample_start - pos, 0) + 1
        rows.append(row)
        frames.append(pos)
        offsets.append(sample_start)
        pos = end
    if n_truncated:
        logger.debug("%s: dropped a truncated final ping frame", path.name)
    if not rows:
        raise ValueError(f"{path}: no Humminbird ping frames (expected frames starting C0 DE AB 21)")

    names = sorted({k for r in rows for k in r})
    fields = {k: np.array([r.get(k, -1) for r in rows], dtype=np.int64) for k in names}
    beams = np.unique(fields.get("beam", np.array([-1])))
    if beams.size != 1:
        raise ValueError(f"{path}: ping frames from several beams {beams.tolist()} in one SON file")
    return HumBeam(
        path=path,
        beam=int(beams[0]),
        fields=fields,
        frame_offset=np.array(frames, dtype=np.int64),
        sample_offset=np.array(offsets, dtype=np.int64),
        n_samples=fields["n_samples"].copy(),
        header_sizes=header_sizes,
        n_truncated=n_truncated,
        n_resyncs=n_resyncs,
        _data=data,
    )


def read_humminbird(dat_path: str | Path, beams: tuple[int, ...] | None = None) -> HumRecording:
    """Read a Humminbird recording from its ``.DAT`` file.

    Parameters
    ----------
    dat_path : str or Path
        The recording's ``.DAT`` file; its SON files are found with
        :func:`son_directory`.
    beams : tuple of int, optional
        Beam numbers to read (``B00n.SON``). Defaults to every SON file
        in the directory.

    Returns
    -------
    HumRecording
    """
    dat_path = Path(dat_path)
    dat = read_dat(dat_path)
    directory = son_directory(dat_path)
    sons = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".son")
    out: dict[int, HumBeam] = {}
    for son in sons:
        if beams is not None and _beam_from_name(son) not in beams:
            continue
        beam = read_son(son)
        if beam.beam in out:
            raise ValueError(f"{directory}: two SON files hold beam {beam.beam}")
        out[beam.beam] = beam
        logger.debug("%s: beam %d (%s), %d pings, header sizes %s", son.name, beam.beam,
                     BEAM_NAMES.get(beam.beam, "unknown"), beam.n_pings, beam.header_sizes)
    if not out:
        raise ValueError(f"{directory}: no SON files" + (f" for beams {beams}" if beams else ""))
    return HumRecording(dat=dat, beams=out)


def humminbird_to_lonlat(easting, northing) -> tuple[np.ndarray, np.ndarray]:
    """Humminbird Mercator easting / northing (m) to WGS84 longitude / latitude (degrees).

    The positions are on a sphere of radius :data:`MERCATOR_RADIUS_M`;
    the latitude formula (from PyHum and PINGVerter) inverts the
    Mercator northing to geocentric latitude and converts that to
    geodetic latitude with the factor 1.0067642927.
    """
    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)
    lon = np.degrees(e / MERCATOR_RADIUS_M)
    lat = np.degrees(np.arctan(np.tan(2.0 * np.arctan(np.exp(n / MERCATOR_RADIUS_M)) - np.pi / 2) * 1.0067642927))
    return lon, lat


def lonlat_to_humminbird(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`humminbird_to_lonlat` (for tests and fixtures)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    e = np.radians(lon) * MERCATOR_RADIUS_M
    geocentric = np.arctan(np.tan(np.radians(lat)) / 1.0067642927)
    n = MERCATOR_RADIUS_M * np.log(np.tan((geocentric + np.pi / 2) / 2.0))
    return e, n


# ============================================================
# INTERNALS
# ============================================================

def _beam_from_name(path: Path) -> int | None:
    stem = path.stem.upper()
    if stem.startswith("B") and stem[1:].isdigit():
        return int(stem[1:])
    return None


def _parse_header(data: bytes, pos: int) -> tuple[dict[str, int], int] | None:
    """Decode the tagged header of the frame at ``pos``.

    Returns the fields and the offset of the first sample, or None if
    the file ends inside the header.
    """
    size = len(data)
    p = pos + len(FRAME_MAGIC)
    row: dict[str, int] = {}
    while True:
        if p >= size:
            return None
        tag = data[p]
        p += 1
        if tag & 0x80:
            if p + 4 > size:
                return None
            raw = data[p:p + 4]
            p += 4
            if tag in SPLIT_TAGS:
                hi, lo = struct.unpack(">HH", raw)
                row[SPLIT_TAGS[tag][0]], row[SPLIT_TAGS[tag][1]] = hi, lo
            else:
                fmt = ">i" if tag in SIGNED_TAGS else ">I"
                row[TAG_NAMES.get(tag, f"tag_{tag}")] = struct.unpack(fmt, raw)[0]
            if tag == TAG_N_SAMPLES:
                if p >= size:
                    return None
                if data[p] != HEADER_END:
                    raise ValueError(
                        f"ping frame at byte {pos}: sample count not followed by the header end byte 0x21"
                    )
                return row, p + 1
        else:
            if p + 1 > size:
                return None
            row[TAG_NAMES.get(tag, f"tag_{tag}")] = data[p]
            p += 1
