"""
Low-level XTF (eXtended Triton Format) reader.

Reads the file header, frames packets, and parses sonar packets into
per-ping header records and per-channel sample arrays. The converter to
the standard survey model is a separate step; ``read_xtf`` is still a
placeholder.

File layout
-----------
A file starts with a 1024-byte header holding up to six ``ChanInfo``
channel descriptions; with more than six channels the remaining
``ChanInfo`` entries follow contiguously in further 1024-byte blocks.
Packets follow. Each starts with the magic ``0xFACE``, a header type and
``NumBytesThisRecord``, the packet's total size including any padding.
That byte count is authoritative: the reader steps from packet to packet
with it, never with sizes derived from the packet contents. A packet with
a bad magic triggers a forward scan for the next ``0xFACE`` (counted, and
logged at WARNING); a truncated final packet is logged at WARNING and
ends the read.

A sonar packet (header type 0) is a 256-byte ping header followed, for
each channel, by a 64-byte channel header and ``NumSamples`` samples.
The sample type comes from the file header's ``ChanInfo`` entry for that
channel (``SampleFormat``, else ``BytesPerSample``).

Channel blocks are walked using the packet byte count
------------------------------------------------------
``NumChansToFollow`` in the ping header is treated as an upper bound.
Channel blocks are parsed while at least a full 64-byte channel header
remains inside the packet; a channel whose samples would run past the
packet end is clipped to the samples that fit. Each mismatch kind is
logged at WARNING once per file with a count.

Evidence for this rule (EdgeTech 4200 files recorded by Isis, four
channels declared in the file header: port/starboard at 120 kHz as
channels 0/1 and at 410 kHz as channels 2/3, ``BytesPerSample`` 2):

- In two such files the first sonar packets (2976 of 25539 in one file,
  1680 of 20475 in the other, a single contiguous run at the start) are
  4480 bytes long while their ping headers say ``NumChansToFollow = 4``
  and the first two channel headers say ``NumSamples = 1024``. Normal
  packets are 8704 = 256 + 4 * (64 + 1024 * 2) bytes.
- 4480 = 256 + 2 * (64 + 1024 * 2) exactly: the packet holds two
  complete channel blocks (channel numbers 0 and 1, the low-frequency
  pair) and nothing else. The "third channel header" a reader trusting
  ``NumChansToFollow`` would parse starts with ``0xFACE``: it is the next
  packet.
- The samples are 16-bit, 1024 per channel as declared. Read as uint16,
  adjacent samples correlate (lag-1 r = 0.96 and 0.79 over the two
  runs, 0.86 and 0.80 in the same files' normal pings), values stay
  within the normal pings' 0..32767 range, and medians are comparable
  (1148 vs 1505, 285 vs 731). Read as 8-bit, adjacent samples do not
  correlate (r = -0.01 and -0.46). The first pings of each run are
  faint (high byte zero in 88-99.99 % of the first 200 pings), rising to
  the normal level. Ping numbers are continuous across the switch to
  four-channel packets, navigation is continuous, and the low-only
  pings repeat faster (0.08 s vs 0.14 s at 100 m range, 0.21 s vs
  0.42 s at 300 m).

So those pings carry only the low-frequency pair; the high-frequency
channels are absent, not empty. pyxtf 1.5.0 trusts ``NumChansToFollow``
and fails on such packets.

Sample arrays
-------------
Samples keep their recorded dtype (uint8/uint16/uint32/float32), so
values are bit-identical to the file. A channel's array is
``(n_ping, max_samples)`` over all sonar pings of the file; pings where
the channel is absent or shorter are zero-padded on the right, and the
true length is in ``num_samples`` (0 where absent, with ``present``
False). The converter decides whether padding becomes NaN.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

import numpy as np

from sidescanpy.io.xtf.structs import (
    CHAN_INFO_DTYPE,
    CHAN_INFO_IN_FIRST_BLOCK,
    CHANNEL_TYPE_PORT,
    CHANNEL_TYPE_STBD,
    FILE_FORMAT_XTF,
    FILE_HEADER_BLOCK_BYTES,
    FILE_HEADER_DTYPE,
    HEADER_TYPE_NAMES,
    HEADER_TYPE_SONAR,
    PACKET_MAGIC,
    PACKET_START_DTYPE,
    PING_CHAN_HEADER_DTYPE,
    PING_HEADER_DTYPE,
    sample_dtype,
)

logger = logging.getLogger(__name__)

_MAGIC_BYTES = PACKET_MAGIC.to_bytes(2, "little")
_START_SIZE = PACKET_START_DTYPE.itemsize
_PING_SIZE = PING_HEADER_DTYPE.itemsize
_CHAN_SIZE = PING_CHAN_HEADER_DTYPE.itemsize
_SCAN_CHUNK_BYTES = 1 << 20

_CHANNEL_COUNT_FIELDS = (
    "NumberOfSonarChannels",
    "NumberOfBathymetryChannels",
    "NumberOfSnippetChannels",
    "NumberOfForwardLookArrays",
    "NumberOfEchoStrengthChannels",
    "NumberOfInterferometryChannels",
)


# ============================================================
# FILE HEADER
# ============================================================

@dataclass(frozen=True)
class XtfFileHeader:
    """Parsed XTF file header.

    Attributes
    ----------
    record : numpy.void
        The first 1024-byte header block as a ``FILE_HEADER_DTYPE``
        record (its ``ChanInfo`` field holds the first six slots).
    chan_info : numpy.ndarray
        ``CHAN_INFO_DTYPE`` array with one entry per declared channel
        (the sum of the header's channel counts), including entries
        from extra header blocks.
    size : int
        Header size in bytes, i.e. the offset of the first packet.
    """

    record: np.void
    chan_info: np.ndarray
    size: int

    def __getitem__(self, name: str):
        return self.record[name]

    @property
    def n_channels(self) -> int:
        """Number of channels declared in the header."""
        return len(self.chan_info)


def _declared_channel_count(record: np.void) -> int:
    return int(sum(int(record[f]) for f in _CHANNEL_COUNT_FIELDS))


def header_size_for_channels(n_channels: int) -> int:
    """Return the file-header size in bytes for ``n_channels`` channels.

    The first 1024-byte block holds six ``ChanInfo`` entries; further
    entries follow contiguously from byte 1024, padded to whole
    1024-byte blocks.

    Parameters
    ----------
    n_channels : int
        Declared channel count.

    Returns
    -------
    int
    """
    extra = max(0, n_channels - CHAN_INFO_IN_FIRST_BLOCK)
    extra_bytes = extra * CHAN_INFO_DTYPE.itemsize
    blocks = -(-extra_bytes // FILE_HEADER_BLOCK_BYTES)
    return FILE_HEADER_BLOCK_BYTES * (1 + blocks)


def _read_header(f: BinaryIO, path: Path) -> XtfFileHeader:
    block = f.read(FILE_HEADER_BLOCK_BYTES)
    if len(block) < FILE_HEADER_BLOCK_BYTES:
        raise ValueError(f"{path}: too short for an XTF file header ({len(block)} bytes)")
    record = np.frombuffer(block, dtype=FILE_HEADER_DTYPE)[0].copy()
    if int(record["FileFormat"]) != FILE_FORMAT_XTF:
        raise ValueError(
            f"{path}: not an XTF file (FileFormat 0x{int(record['FileFormat']):02X}, "
            f"expected 0x{FILE_FORMAT_XTF:02X})"
        )
    n = _declared_channel_count(record)
    first = np.array(record["ChanInfo"], dtype=CHAN_INFO_DTYPE)
    size = header_size_for_channels(n)
    if n > CHAN_INFO_IN_FIRST_BLOCK:
        rest = f.read(size - FILE_HEADER_BLOCK_BYTES)
        n_extra = n - CHAN_INFO_IN_FIRST_BLOCK
        need = n_extra * CHAN_INFO_DTYPE.itemsize
        if len(rest) < need:
            raise ValueError(f"{path}: file header truncated in extra ChanInfo blocks")
        extra = np.frombuffer(rest, dtype=CHAN_INFO_DTYPE, count=n_extra)
        chan_info = np.concatenate([first, extra])
        logger.debug("%s: %d channels, header spans %d bytes", path, n, size)
    else:
        chan_info = first[:n].copy()
    return XtfFileHeader(record=record, chan_info=chan_info, size=size)


def read_xtf_header(path: str | Path) -> XtfFileHeader:
    """Read the file header of an XTF file.

    Parameters
    ----------
    path : str or Path
        XTF file.

    Returns
    -------
    XtfFileHeader

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is too short or its first byte is not the XTF
        ``FileFormat`` value 0x7B.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return _read_header(f, path)


# ============================================================
# PACKET FRAMING
# ============================================================

@dataclass(frozen=True)
class XtfPacket:
    """One framed packet.

    Attributes
    ----------
    header_type : int
        ``HeaderType`` from the packet start (see
        ``structs.HEADER_TYPE_NAMES``).
    offset : int
        Byte offset of the packet in the file.
    data : bytes
        The whole packet, starting with its 14 packet-start bytes, of
        length ``NumBytesThisRecord``.
    """

    header_type: int
    offset: int
    data: bytes

    @property
    def start(self) -> np.void:
        """The packet start fields as a ``PACKET_START_DTYPE`` record."""
        return np.frombuffer(self.data, dtype=PACKET_START_DTYPE, count=1)[0]


@dataclass
class PacketStats:
    """Counts gathered while framing packets.

    Attributes
    ----------
    counts : collections.Counter
        Packets framed, by header type (yielded or skipped).
    skipped : collections.Counter
        Packets not yielded because of the ``header_types`` filter.
    unknown : collections.Counter
        Packets whose header type the specification does not define.
    resyncs : int
        Bad-magic events that needed a forward scan.
    resync_bytes : int
        Bytes skipped by those scans.
    truncated : bool
        True if the file ended inside a packet.
    """

    counts: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    unknown: Counter = field(default_factory=Counter)
    resyncs: int = 0
    resync_bytes: int = 0
    truncated: bool = False


def _scan_for_packet(f: BinaryIO, start: int, file_size: int) -> int | None:
    """Return the offset of the next plausible packet start at or after ``start``."""
    pos = start
    while pos < file_size - 1:
        f.seek(pos)
        chunk = f.read(_SCAN_CHUNK_BYTES + 1)
        if len(chunk) < 2:
            return None
        i = chunk.find(_MAGIC_BYTES)
        while i != -1:
            cand = pos + i
            f.seek(cand)
            head = f.read(_START_SIZE)
            if len(head) < _START_SIZE:
                return None
            nbytes = int(np.frombuffer(head, PACKET_START_DTYPE)[0]["NumBytesThisRecord"])
            if nbytes >= _START_SIZE:
                return cand
            i = chunk.find(_MAGIC_BYTES, i + 1)
        pos += len(chunk) - 1
    return None


def iter_packets(
    path: str | Path,
    header_types: Iterable[int] | None = None,
    stats: PacketStats | None = None,
) -> Iterator[XtfPacket]:
    """Stream the packets of an XTF file.

    Parameters
    ----------
    path : str or Path
        XTF file.
    header_types : iterable of int, optional
        Header types to yield. Others are skipped by their byte count
        without reading their body. Default: yield every packet.
    stats : PacketStats, optional
        Filled in place with counts as packets are framed (useful because
        a generator has no return value to the caller).

    Yields
    ------
    XtfPacket
        Packets in file order.

    Notes
    -----
    Packets whose header type is not defined by the specification are
    counted in ``stats.unknown`` and logged at DEBUG; they are yielded
    like any other when no filter is given. A bad magic number or a byte
    count smaller than the packet start triggers a forward scan for the
    next ``0xFACE`` (logged at WARNING). A packet extending past the end
    of the file is logged at WARNING and ends iteration.
    """
    path = Path(path)
    wanted = None if header_types is None else frozenset(int(t) for t in header_types)
    if stats is None:
        stats = PacketStats()
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        header = _read_header(f, path)
        offset = header.size
        while offset < file_size:
            f.seek(offset)
            head = f.read(_START_SIZE)
            if len(head) < _START_SIZE:
                stats.truncated = True
                logger.warning(
                    "%s: %d trailing bytes at offset %d, too short for a packet; stopping",
                    path, len(head), offset,
                )
                break
            start = np.frombuffer(head, PACKET_START_DTYPE)[0]
            nbytes = int(start["NumBytesThisRecord"])
            if int(start["MagicNumber"]) != PACKET_MAGIC or nbytes < _START_SIZE:
                found = _scan_for_packet(f, offset + 1, file_size)
                stats.resyncs += 1
                if found is None:
                    stats.resync_bytes += file_size - offset
                    logger.warning(
                        "%s: bad packet start at offset %d and no further 0xFACE; stopping",
                        path, offset,
                    )
                    break
                stats.resync_bytes += found - offset
                logger.warning(
                    "%s: bad packet start at offset %d; resynced at %d (%d bytes skipped)",
                    path, offset, found, found - offset,
                )
                offset = found
                continue
            htype = int(start["HeaderType"])
            if offset + nbytes > file_size:
                stats.truncated = True
                logger.warning(
                    "%s: final packet at offset %d (type %d) truncated: %d of %d bytes; stopping",
                    path, offset, htype, file_size - offset, nbytes,
                )
                break
            stats.counts[htype] += 1
            if htype not in HEADER_TYPE_NAMES:
                stats.unknown[htype] += 1
                logger.debug("%s: unknown packet type %d at offset %d", path, htype, offset)
            if wanted is not None and htype not in wanted:
                stats.skipped[htype] += 1
            else:
                body = f.read(nbytes - _START_SIZE)
                yield XtfPacket(header_type=htype, offset=offset, data=head + body)
            offset += nbytes
    if stats.skipped:
        logger.debug("%s: skipped packets by type: %s", path, dict(stats.skipped))


def count_packets(path: str | Path) -> PacketStats:
    """Frame every packet of a file without parsing them.

    Parameters
    ----------
    path : str or Path
        XTF file.

    Returns
    -------
    PacketStats
        Counts by header type, plus resync and truncation information.
    """
    stats = PacketStats()
    for _ in iter_packets(path, header_types=(), stats=stats):
        pass
    return stats


# ============================================================
# SONAR PINGS
# ============================================================

@dataclass
class XtfChannel:
    """Samples and headers of one sonar channel over all pings.

    Attributes
    ----------
    channel_number : int
        ``ChannelNumber`` from the channel headers (index into the file
        header's ``ChanInfo``).
    chan_info : numpy.void
        The ``ChanInfo`` record for this channel.
    name : str
        ``ChanInfo.ChannelName``.
    side : str
        ``"port"``, ``"stbd"`` or ``"other"`` (from ``TypeOfChannel``).
    frequency : float
        ``ChanInfo.Frequency``, or the most common nonzero per-ping
        channel-header ``Frequency`` when that is 0, else NaN. The unit is
        whatever the recorder wrote (kHz in the files seen so far).
    samples : numpy.ndarray
        ``(n_ping, max_samples)`` in the recorded sample dtype,
        zero-padded on the right.
    num_samples : numpy.ndarray
        ``(n_ping,)`` uint32, samples stored for each ping (0 where absent).
    present : numpy.ndarray
        ``(n_ping,)`` bool, whether the ping held this channel.
    headers : numpy.ndarray
        ``(n_ping,)`` ``PING_CHAN_HEADER_DTYPE`` records; all zero where
        the channel is absent.
    """

    channel_number: int
    chan_info: np.void
    name: str
    side: str
    frequency: float
    samples: np.ndarray
    num_samples: np.ndarray
    present: np.ndarray
    headers: np.ndarray


@dataclass
class XtfFrequencyGroup:
    """A port/starboard channel pair sharing one frequency.

    Attributes
    ----------
    frequency : float
        Frequency as recorded (see ``XtfChannel.frequency``); NaN when
        unknown, in which case channels were paired in channel order.
    port, stbd : XtfChannel or None
        The two sides; one may be missing.
    """

    frequency: float
    port: XtfChannel | None = None
    stbd: XtfChannel | None = None


@dataclass
class XtfPingData:
    """Sonar pings of one XTF file, as recorded.

    Attributes
    ----------
    path : Path
        Source file.
    file_header : XtfFileHeader
        Parsed file header.
    ping_headers : numpy.ndarray
        ``(n_ping,)`` ``PING_HEADER_DTYPE`` records, in file order.
    packet_offsets : numpy.ndarray
        ``(n_ping,)`` int64 file offset of each sonar packet.
    channels : dict[int, XtfChannel]
        Channels keyed by ``ChannelNumber``.
    frequency_groups : list[XtfFrequencyGroup]
        Port/starboard channels grouped by frequency, ascending
        (unknown frequencies last).
    stats : PacketStats
        Packet counts from the read.
    """

    path: Path
    file_header: XtfFileHeader
    ping_headers: np.ndarray
    packet_offsets: np.ndarray
    channels: dict[int, XtfChannel]
    frequency_groups: list[XtfFrequencyGroup]
    stats: PacketStats

    @property
    def n_pings(self) -> int:
        """Number of sonar pings."""
        return len(self.ping_headers)

    @property
    def times(self) -> np.ndarray:
        """Ping times as ``datetime64[ns]`` (see :func:`ping_times`)."""
        return ping_times(self.ping_headers)


def ping_times(ping_headers: np.ndarray) -> np.ndarray:
    """Compute ping times from ping header records.

    Uses ``Year``, ``Month``, ``Day``, ``Hour``, ``Minute``, ``Second``
    and ``HSeconds`` (hundredths). The time base is whatever the recorder
    used (normally UTC). Records with an invalid date give ``NaT``.

    Parameters
    ----------
    ping_headers : numpy.ndarray
        ``PING_HEADER_DTYPE`` records.

    Returns
    -------
    numpy.ndarray
        ``datetime64[ns]``, one per record.
    """
    h = np.asarray(ping_headers)
    year = h["Year"].astype(np.int64)
    month = h["Month"].astype(np.int64)
    day = h["Day"].astype(np.int64)
    valid = (year > 0) & (month >= 1) & (month <= 12) & (day >= 1) & (day <= 31)
    months = np.where(valid, (year - 1970) * 12 + month - 1, 0)
    date = months.astype("datetime64[M]").astype("datetime64[D]") + np.where(
        valid, day - 1, 0
    ).astype("timedelta64[D]")
    ns = (
        h["Hour"].astype(np.int64) * 3600
        + h["Minute"].astype(np.int64) * 60
        + h["Second"].astype(np.int64)
    ) * 1_000_000_000 + h["HSeconds"].astype(np.int64) * 10_000_000
    out = date.astype("datetime64[ns]") + ns.astype("timedelta64[ns]")
    out[~valid] = np.datetime64("NaT", "ns")
    return out


@dataclass
class _Mismatches:
    fewer_channels: int = 0
    declared: Counter = field(default_factory=Counter)
    found: Counter = field(default_factory=Counter)
    clipped: int = 0
    no_chan_info: int = 0
    legacy_count: int = 0


def _parse_layout(
    data: bytes,
    chan_info: np.ndarray,
    dtypes: list[np.dtype | None],
    mm: _Mismatches,
) -> list[tuple[int, int, int, int]]:
    """Locate channel blocks in one sonar packet.

    Returns a list of ``(channel_number, chan_header_offset, data_offset,
    n_samples)`` with offsets relative to the packet start.
    """
    end = len(data)
    declared = int(np.frombuffer(data, PACKET_START_DTYPE, count=1)[0]["NumChansToFollow"])
    pos = _PING_SIZE
    out = []
    for k in range(declared):
        if end - pos < _CHAN_SIZE:
            mm.fewer_channels += 1
            mm.declared[declared] += 1
            mm.found[k] += 1
            break
        ch = np.frombuffer(data, PING_CHAN_HEADER_DTYPE, count=1, offset=pos)[0]
        number = int(ch["ChannelNumber"])
        idx = number if number < len(chan_info) else k
        if idx >= len(chan_info) or dtypes[idx] is None:
            mm.no_chan_info += 1
            break
        dt = dtypes[idx]
        n = int(ch["NumSamples"])
        if n == 0:
            # Pre-X41 files kept the sample count in ChanInfo (now Reserved).
            legacy = int(chan_info[idx]["Reserved"])
            if legacy and legacy * dt.itemsize <= end - pos - _CHAN_SIZE:
                n = legacy
                mm.legacy_count += 1
        avail = (end - pos - _CHAN_SIZE) // dt.itemsize
        if n > avail:
            mm.clipped += 1
            n = avail
        out.append((idx, pos, pos + _CHAN_SIZE, n))
        pos += _CHAN_SIZE + n * dt.itemsize
    return out


def _report_mismatches(path: Path, mm: _Mismatches, n_sonar: int) -> None:
    if mm.fewer_channels:
        logger.warning(
            "%s: %d of %d sonar packets hold fewer channel blocks than NumChansToFollow "
            "(declared %s, found %s); using the packet byte count, the missing channels "
            "are absent from those pings",
            path, mm.fewer_channels, n_sonar, dict(mm.declared), dict(mm.found),
        )
    if mm.clipped:
        logger.warning(
            "%s: %d channel blocks declare more samples than fit in their packet; "
            "clipped to the packet byte count",
            path, mm.clipped,
        )
    if mm.no_chan_info:
        logger.warning(
            "%s: %d channel blocks have no usable ChanInfo entry (unknown sample type); "
            "the rest of those packets was skipped",
            path, mm.no_chan_info,
        )
    if mm.legacy_count:
        logger.debug(
            "%s: %d channel blocks took their sample count from ChanInfo (NumSamples 0)",
            path, mm.legacy_count,
        )


def _group_frequencies(channels: dict[int, XtfChannel]) -> list[XtfFrequencyGroup]:
    known: dict[float, XtfFrequencyGroup] = {}
    unknown: list[XtfFrequencyGroup] = []
    for number in sorted(channels):
        ch = channels[number]
        if ch.side not in ("port", "stbd"):
            continue
        if np.isfinite(ch.frequency):
            group = known.setdefault(ch.frequency, XtfFrequencyGroup(frequency=ch.frequency))
        else:
            if not unknown or getattr(unknown[-1], ch.side) is not None:
                unknown.append(XtfFrequencyGroup(frequency=float("nan")))
            group = unknown[-1]
        if getattr(group, ch.side) is not None:
            logger.warning(
                "XTF channels %d and %d are both %s at frequency %s; keeping %d",
                getattr(group, ch.side).channel_number, number, ch.side,
                ch.frequency, getattr(group, ch.side).channel_number,
            )
            continue
        setattr(group, ch.side, ch)
    if unknown:
        logger.warning(
            "XTF frequency unknown for %d channel pair(s); paired in channel order",
            len(unknown),
        )
    return [known[f] for f in sorted(known)] + unknown


def read_xtf_pings(path: str | Path) -> XtfPingData:
    """Read every sonar ping of an XTF file.

    Two passes over the file: the first frames sonar packets and records
    the ping and channel headers and where each channel's samples are;
    the second reads the samples straight into preallocated arrays.
    Other packet types are counted and skipped.

    Parameters
    ----------
    path : str or Path
        XTF file.

    Returns
    -------
    XtfPingData
        Ping headers, per-channel sample arrays (recorded dtype,
        zero-padded) and port/starboard frequency groups.

    Raises
    ------
    ValueError
        If the file header is invalid or a channel uses an unsupported
        sample format.
    """
    path = Path(path)
    header = read_xtf_header(path)
    chan_info = header.chan_info
    dtypes: list[np.dtype | None] = []
    for ci in chan_info:
        try:
            dtypes.append(sample_dtype(ci))
        except ValueError:
            if int(ci["TypeOfChannel"]) in (CHANNEL_TYPE_PORT, CHANNEL_TYPE_STBD):
                raise
            dtypes.append(None)

    stats = PacketStats()
    mm = _Mismatches()
    ping_buf = bytearray()
    chan_buf = bytearray()
    offsets = array("q")
    ent_ping = array("q")
    ent_idx = array("q")
    ent_pos = array("q")
    ent_n = array("q")
    n_ping = 0
    for pkt in iter_packets(path, header_types=(HEADER_TYPE_SONAR,), stats=stats):
        if len(pkt.data) < _PING_SIZE:
            logger.warning(
                "%s: sonar packet at offset %d is %d bytes, shorter than a ping header; skipped",
                path, pkt.offset, len(pkt.data),
            )
            continue
        ping_buf += pkt.data[:_PING_SIZE]
        offsets.append(pkt.offset)
        for idx, hpos, dpos, n in _parse_layout(pkt.data, chan_info, dtypes, mm):
            chan_buf += pkt.data[hpos:hpos + _CHAN_SIZE]
            ent_ping.append(n_ping)
            ent_idx.append(idx)
            ent_pos.append(pkt.offset + dpos)
            ent_n.append(n)
        n_ping += 1
    _report_mismatches(path, mm, n_ping)

    ping_headers = np.frombuffer(bytes(ping_buf), dtype=PING_HEADER_DTYPE).copy()
    chan_headers = np.frombuffer(bytes(chan_buf), dtype=PING_CHAN_HEADER_DTYPE)
    e_ping = np.frombuffer(ent_ping, dtype=np.int64)
    e_idx = np.frombuffer(ent_idx, dtype=np.int64)
    e_pos = np.frombuffer(ent_pos, dtype=np.int64)
    e_n = np.frombuffer(ent_n, dtype=np.int64)

    channels: dict[int, XtfChannel] = {}
    for idx in np.unique(e_idx):
        idx = int(idx)
        sel = e_idx == idx
        pings = e_ping[sel]
        if len(np.unique(pings)) != len(pings):
            logger.warning(
                "%s: channel %d appears more than once in some pings; keeping the last",
                path, idx,
            )
        ci = chan_info[idx]
        dt = dtypes[idx]
        max_n = int(e_n[sel].max()) if sel.any() else 0
        num = np.zeros(n_ping, dtype=np.uint32)
        num[pings] = e_n[sel]
        present = np.zeros(n_ping, dtype=bool)
        present[pings] = True
        heads = np.zeros(n_ping, dtype=PING_CHAN_HEADER_DTYPE)
        heads[pings] = chan_headers[sel]
        ctype = int(ci["TypeOfChannel"])
        side = {CHANNEL_TYPE_PORT: "port", CHANNEL_TYPE_STBD: "stbd"}.get(ctype, "other")
        freq = float(ci["Frequency"])
        if freq <= 0:
            per_ping = heads["Frequency"][present]
            per_ping = per_ping[per_ping > 0]
            freq = float(np.bincount(per_ping).argmax()) if per_ping.size else float("nan")
        channels[idx] = XtfChannel(
            channel_number=idx,
            chan_info=ci,
            name=ci["ChannelName"].decode("ascii", "replace").rstrip("\x00 "),
            side=side,
            frequency=freq,
            samples=np.zeros((n_ping, max_n), dtype=dt),
            num_samples=num,
            present=present,
            headers=heads,
        )

    with open(path, "rb") as f:
        for p, idx, pos, n in zip(e_ping.tolist(), e_idx.tolist(), e_pos.tolist(), e_n.tolist()):
            if n == 0:
                continue
            f.seek(pos)
            row = channels[idx].samples[p, :n]
            if f.readinto(row) != row.nbytes:  # pragma: no cover - checked in pass 1
                raise ValueError(f"{path}: short read of channel {idx} at offset {pos}")

    groups = _group_frequencies(channels)
    logger.debug(
        "%s: %d pings, channels %s, frequency groups %s",
        path, n_ping, sorted(channels), [g.frequency for g in groups],
    )
    return XtfPingData(
        path=path,
        file_header=header,
        ping_headers=ping_headers,
        packet_offsets=np.frombuffer(offsets, dtype=np.int64).copy(),
        channels=channels,
        frequency_groups=groups,
        stats=stats,
    )


# ============================================================
# SURVEY ENTRY POINT (placeholder)
# ============================================================

def read_xtf(path: str | Path):
    """Read an XTF file into a survey (not yet implemented).

    The low-level readers (:func:`read_xtf_header`, :func:`iter_packets`,
    :func:`read_xtf_pings`) exist; the conversion to a survey does not.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "XTF conversion to a survey is not implemented yet; "
        "use read_xtf_pings() for the recorded pings."
    )
