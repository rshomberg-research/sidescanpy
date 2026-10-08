"""
Real-data tests for the low-level XTF reader, with pyxtf as the oracle.

Set ``SIDESCANPY_XTF_DATA`` to a directory of XTF files to run them; they
skip otherwise. Developed against EdgeTech 4200 dual-frequency files
recorded by Isis (four sidescan channels, 16-bit samples). The file
names below are those of that set; missing files are skipped.

- ``PYXTF_READABLE``: files pyxtf 1.5.0 reads. Header fields, packet type
  counts, every sample and the mapped ping-header fields must match
  pyxtf exactly.
- ``FEWER_CHANNELS``: files whose first pings hold only the low-frequency
  pair while declaring four channels (see ``sidescanpy.io.xtf.reader``);
  pyxtf fails on them, so they are checked for self-consistency.

2026 rshomberg@uri.edu
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.xtf import count_packets, read_xtf_header, read_xtf_pings
from sidescanpy.io.xtf.structs import FILE_HEADER_DTYPE, PING_HEADER_DTYPE

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]

FAST_FILES = ["NBP050501C.XTF", "NBP050504A.XTF"]
PYXTF_READABLE = FAST_FILES + [
    "NBP050501A.XTF",
    "NBP050502.XTF",
    "NBP050503.XTF",
    "NBP050505.XTF",
    "NBP050506A.XTF",
    "NBP050506B.XTF",
]
FEWER_CHANNELS = ["NBP050501B.XTF", "NBP050504B.XTF"]

PING_FIELDS = [
    "Year", "Month", "Day", "Hour", "Minute", "Second", "HSeconds", "PingNumber",
    "SensorXcoordinate", "SensorYcoordinate", "ShipXcoordinate", "ShipYcoordinate",
    "SensorHeading", "SensorDepth", "SensorPrimaryAltitude", "SensorPitch", "SensorRoll",
    "SoundVelocity", "Heave", "Layback", "SensorSpeed", "NumChansToFollow",
]
CHAN_FIELDS = ["ChannelNumber", "SlantRange", "TimeDelay", "TimeDuration", "NumSamples",
               "Frequency", "GainCode", "InitialGainCode", "Weight"]


def _data_file(name: str) -> Path:
    root = os.environ.get("SIDESCANPY_XTF_DATA")
    if not root:
        pytest.skip("SIDESCANPY_XTF_DATA not set")
    path = Path(root) / name
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _pyxtf_read(path: Path):
    pyxtf = pytest.importorskip("pyxtf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyxtf, *pyxtf.xtf_read(str(path))


def _equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _check_against_pyxtf(path: Path) -> None:
    pyxtf, fh, packets = _pyxtf_read(path)

    header = read_xtf_header(path)
    assert header.record.tobytes() == bytes(fh)
    for name in FILE_HEADER_DTYPE.names:
        if name == "ChanInfo":
            continue
        dt = FILE_HEADER_DTYPE[name].base
        theirs = getattr(fh, name)
        if not isinstance(theirs, (bytes, int, float)):
            theirs = list(theirs)
        assert _equal(np.asarray(header[name], dt), np.asarray(theirs, dt)), name
    for i, ci in enumerate(header.chan_info):
        assert ci.tobytes() == bytes(fh.ChanInfo[i]), f"ChanInfo[{i}]"

    stats = count_packets(path)
    assert dict(stats.counts) == {int(k): len(v) for k, v in packets.items()}
    assert stats.resyncs == 0 and not stats.truncated

    data = read_xtf_pings(path)
    sonar = packets[pyxtf.XTFHeaderType.sonar]
    assert data.n_pings == len(sonar)
    for name in PING_FIELDS:
        theirs = np.array([getattr(p, name) for p in sonar], dtype=PING_HEADER_DTYPE[name])
        assert _equal(data.ping_headers[name], theirs), name
    raw_headers = b"".join(bytes(p)[:256] for p in sonar)
    assert data.ping_headers.tobytes() == raw_headers
    times = np.array([p.get_time() for p in sonar], dtype="datetime64[ns]")
    assert _equal(data.times, times)
    n_chan = len(sonar[0].data)
    assert sorted(data.channels) == list(range(n_chan))
    for c in range(n_chan):
        ch = data.channels[c]
        assert ch.present.all()
        theirs = np.vstack([p.data[c] for p in sonar])
        assert ch.samples.dtype == theirs.dtype
        assert _equal(ch.samples, theirs), f"channel {c} samples"
        for name in CHAN_FIELDS:
            vals = np.array([getattr(p.ping_chan_headers[c], name) for p in sonar],
                            dtype=ch.headers.dtype[name])
            assert _equal(ch.headers[name], vals), f"channel {c} {name}"


@pytest.mark.parametrize("name", FAST_FILES)
def test_matches_pyxtf(name):
    _check_against_pyxtf(_data_file(name))


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n in PYXTF_READABLE if n not in FAST_FILES])
def test_matches_pyxtf_all_readable(name):
    _check_against_pyxtf(_data_file(name))


def test_dual_frequency_layout():
    data = read_xtf_pings(_data_file(FAST_FILES[0]))
    assert data.file_header["NavUnits"] == 3
    assert [g.frequency for g in data.frequency_groups] == [120.0, 410.0]
    for group in data.frequency_groups:
        assert group.port.side == "port" and group.stbd.side == "stbd"
        assert group.port.samples.shape == group.stbd.samples.shape
    # Sensor X/Y are longitude/latitude.
    lon = data.ping_headers["SensorXcoordinate"]
    lat = data.ping_headers["SensorYcoordinate"]
    assert np.all((lon > -180) & (lon < 180)) and np.all((lat > -90) & (lat < 90))


@pytest.mark.parametrize("name", FEWER_CHANNELS)
def test_fewer_channel_files_self_consistent(name, caplog):
    path = _data_file(name)
    stats = count_packets(path)
    assert stats.resyncs == 0 and not stats.truncated
    data = read_xtf_pings(path)
    assert data.n_pings == stats.counts[0]
    assert sum("fewer channel blocks" in r.getMessage() for r in caplog.records) == 1

    # Every packet framed, and each packet's byte count is fully explained
    # by the channel blocks found in it.
    nbytes = data.ping_headers["NumBytesThisRecord"].astype(np.int64)
    gaps = np.diff(np.append(data.packet_offsets, path.stat().st_size))
    assert np.all(gaps >= nbytes)  # non-sonar packets may sit in between
    assert data.packet_offsets[-1] + nbytes[-1] == path.stat().st_size
    used = 256 + sum(
        np.where(ch.present, 64 + ch.num_samples.astype(np.int64) * ch.samples.itemsize, 0)
        for ch in data.channels.values()
    )
    assert np.array_equal(used, nbytes)

    lo, hi = data.frequency_groups
    assert lo.port.present.all() and lo.stbd.present.all()
    absent = ~hi.port.present
    assert absent.any() and np.array_equal(absent, ~hi.stbd.present)
    # The absent pings form one run at the start of the file.
    first_present = int(np.argmax(hi.port.present))
    assert absent[:first_present].all() and not absent[first_present:].any()
    assert (data.ping_headers["NumChansToFollow"] == 4).all()
    assert (hi.port.samples[absent] == 0).all()
    assert np.all(np.diff(data.ping_headers["PingNumber"].astype(np.int64)) == 1)
    # The low-only pings read as 16-bit look like the normal pings: same
    # value ceiling, strong lag-1 correlation; read as 8-bit they do not.
    low_only = lo.port.samples[absent]
    assert low_only.max() <= lo.port.samples[~absent].max()
    assert _lag1(low_only) > 0.5
    assert _lag1(low_only.view(np.uint8)) < 0.2


def _lag1(x: np.ndarray) -> float:
    x = x.astype(float)
    return float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])
