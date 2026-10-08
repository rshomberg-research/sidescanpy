"""
Real-data tests for the low-level XTF reader, with pyxtf as the oracle.

Set ``SIDESCANPY_XTF_DATA`` to a directory of XTF files to run them; they
skip otherwise. Every ``*.xtf`` / ``*.XTF`` file in that directory is
tested. Developed against a public EdgeTech 4200 XTF archive recorded by
Isis (four sidescan channels at two frequencies, 16-bit samples).

Files are classified by content, not by name:

- Files whose sonar packets all hold every declared channel must match
  pyxtf exactly: header fields, packet type counts, every sample and the
  mapped ping-header fields.
- Files with packets holding fewer channel blocks than declared (see
  ``sidescanpy.io.xtf.reader``) are expected to make pyxtf 1.5.0 fail,
  and are checked for self-consistency instead.

The two smallest files run without ``slow``; the rest are marked slow.

2026 rshomberg@uri.edu
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.xtf import XtfPingData, count_packets, read_xtf_header, read_xtf_pings
from sidescanpy.io.xtf.structs import FILE_HEADER_DTYPE, PING_HEADER_DTYPE

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]

N_FAST = 2


def _data_files() -> list[Path]:
    """XTF files in ``SIDESCANPY_XTF_DATA``, smallest first."""
    root = os.environ.get("SIDESCANPY_XTF_DATA")
    if not root or not Path(root).is_dir():
        return []
    files = [p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() == ".xtf"]
    return sorted(files, key=lambda p: (p.stat().st_size, p.name))


DATA_FILES = _data_files()
_NO_DATA = pytest.param(
    None, marks=pytest.mark.skip(reason="SIDESCANPY_XTF_DATA not set or holds no .xtf files")
)


def _params(files: list[Path], slow: bool = False) -> list:
    if not files:
        return [_NO_DATA]
    marks = [pytest.mark.slow] if slow else []
    return [pytest.param(p, id=p.name, marks=marks) for p in files]


FAST_PARAMS = _params(DATA_FILES[:N_FAST])
ALL_PARAMS = _params(DATA_FILES[:N_FAST]) + (
    _params(DATA_FILES[N_FAST:], slow=True) if len(DATA_FILES) > N_FAST else []
)


def _read_with_anomaly(path: Path, caplog) -> tuple[XtfPingData, bool]:
    """Read a file; also return whether the reader found packets with
    fewer channel blocks than declared (it then warns exactly once)."""
    caplog.clear()
    with caplog.at_level("WARNING", logger="sidescanpy.io.xtf.reader"):
        data = read_xtf_pings(path)
    reports = [r for r in caplog.records if "fewer channel blocks" in r.getMessage()]
    fewer = data.block_stats.fewer_channels > 0
    assert len(reports) == int(fewer)
    return data, fewer


PING_FIELDS = [
    "Year", "Month", "Day", "Hour", "Minute", "Second", "HSeconds", "PingNumber",
    "SensorXcoordinate", "SensorYcoordinate", "ShipXcoordinate", "ShipYcoordinate",
    "SensorHeading", "SensorDepth", "SensorPrimaryAltitude", "SensorPitch", "SensorRoll",
    "SoundVelocity", "Heave", "Layback", "SensorSpeed", "NumChansToFollow",
]
CHAN_FIELDS = ["ChannelNumber", "SlantRange", "TimeDelay", "TimeDuration", "NumSamples",
               "Frequency", "GainCode", "InitialGainCode", "Weight"]


def _pyxtf_read(path: Path):
    pyxtf = pytest.importorskip("pyxtf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyxtf, *pyxtf.xtf_read(str(path))


def _equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _check_against_pyxtf(path: Path, data: XtfPingData) -> None:
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


def _check_fewer_channels(path: Path, data: XtfPingData) -> None:
    pyxtf = pytest.importorskip("pyxtf")
    with warnings.catch_warnings(), pytest.raises(RuntimeError, match="exceeds"):
        warnings.simplefilter("ignore")
        pyxtf.xtf_read(str(path))

    stats = count_packets(path)
    assert stats.resyncs == 0 and not stats.truncated
    assert data.n_pings == stats.counts[0]

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


@pytest.mark.parametrize("path", ALL_PARAMS)
def test_matches_pyxtf_or_self_consistent(path, caplog):
    data, fewer_channels = _read_with_anomaly(path, caplog)
    if fewer_channels:
        _check_fewer_channels(path, data)
    else:
        assert all(ch.present.all() for ch in data.channels.values())
        _check_against_pyxtf(path, data)


@pytest.mark.parametrize("path", FAST_PARAMS[:1])
def test_dual_frequency_layout(path):
    data = read_xtf_pings(path)
    assert data.file_header["NavUnits"] == 3
    assert [g.frequency for g in data.frequency_groups] == [120.0, 410.0]
    for group in data.frequency_groups:
        assert group.port.side == "port" and group.stbd.side == "stbd"
        assert group.port.samples.shape == group.stbd.samples.shape
    # Sensor X/Y are longitude/latitude.
    lon = data.ping_headers["SensorXcoordinate"]
    lat = data.ping_headers["SensorYcoordinate"]
    assert np.all((lon > -180) & (lon < 180)) and np.all((lat > -90) & (lat < 90))


def _lag1(x: np.ndarray) -> float:
    x = x.astype(float)
    return float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])
