"""
Unit tests for sidescanpy.io.xtf.reader on synthetic files.

Files are built in-test (``conftest.py``): file header with one or more
blocks, sonar and notes packets, unknown packet types, corrupt and
truncated packets, every supported sample format, sample counts that
change mid-file, and packets holding fewer channel blocks than their
ping header declares.

2026 rshomberg@uri.edu
"""

import logging
import warnings

import numpy as np
import pytest

from sidescanpy.io.xtf import (
    count_packets,
    iter_packets,
    ping_times,
    read_xtf,
    read_xtf_header,
    read_xtf_pings,
)
from sidescanpy.io.xtf.reader import header_size_for_channels
from tests.unit.io.xtf.conftest import (
    PORT,
    STBD,
    chan_info,
    dual_frequency_infos,
    file_header,
    notes_packet,
    packet,
    ramp,
    sonar_packet,
    write_xtf,
)

N_SAMP = 64


def _dual_ping(i, n=N_SAMP, dtype=np.uint16, chans=(0, 1, 2, 3), **kw):
    return sonar_packet([(c, ramp(n, dtype, i * 4 + c)) for c in chans], ping_number=i, **kw)


@pytest.fixture
def dual_file(tmp_path):
    pk = [notes_packet(), notes_packet()] + [_dual_ping(i) for i in range(5)]
    return write_xtf(tmp_path / "dual.xtf", file_header(dual_frequency_infos()), pk)


# ============================================================
# FILE HEADER
# ============================================================

class TestReadHeader:
    def test_fields_and_channels(self, dual_file):
        h = read_xtf_header(dual_file)
        assert h.size == 1024
        assert h["SonarType"] == 38
        assert h["NavUnits"] == 3
        assert h.n_channels == 4
        assert list(h.chan_info["Frequency"]) == [100.0, 100.0, 400.0, 400.0]
        assert list(h.chan_info["TypeOfChannel"]) == [PORT, STBD, PORT, STBD]

    def test_more_than_six_channels_uses_extra_block(self, tmp_path):
        infos = [chan_info(PORT if i % 2 == 0 else STBD, 100.0 + 10 * (i // 2), name=f"C{i}")
                 for i in range(8)]
        chans = [(i, ramp(N_SAMP, np.uint16, i)) for i in range(8)]
        path = write_xtf(tmp_path / "eight.xtf", file_header(infos), [sonar_packet(chans)])
        h = read_xtf_header(path)
        assert h.size == 2048
        assert h.n_channels == 8
        assert [n.decode() for n in h.chan_info["ChannelName"]] == [f"C{i}" for i in range(8)]
        assert list(h.chan_info["Frequency"][6:]) == [130.0, 130.0]
        packets = list(iter_packets(path))
        assert [p.offset for p in packets] == [2048]
        data = read_xtf_pings(path)
        assert len(data.frequency_groups) == 4
        np.testing.assert_array_equal(data.channels[7].samples[0], ramp(N_SAMP, np.uint16, 7))

    @pytest.mark.parametrize("n, size", [(0, 1024), (6, 1024), (7, 2048), (14, 2048), (15, 3072)])
    def test_header_size_for_channels(self, n, size):
        assert header_size_for_channels(n) == size

    def test_not_xtf_raises(self, tmp_path):
        path = tmp_path / "bad.xtf"
        path.write_bytes(b"\x00" * 2048)
        with pytest.raises(ValueError, match="not an XTF"):
            read_xtf_header(path)

    def test_short_file_raises(self, tmp_path):
        path = tmp_path / "short.xtf"
        path.write_bytes(b"\x7b" * 100)
        with pytest.raises(ValueError, match="too short"):
            read_xtf_header(path)


# ============================================================
# PACKET FRAMING
# ============================================================

class TestIterPackets:
    def test_types_offsets_and_bytes(self, dual_file):
        packets = list(iter_packets(dual_file))
        assert [p.header_type for p in packets] == [1, 1, 0, 0, 0, 0, 0]
        assert packets[0].offset == 1024
        assert packets[1].offset == 1024 + 256
        for p in packets:
            assert len(p.data) == int(p.start["NumBytesThisRecord"])
            assert int(p.start["MagicNumber"]) == 0xFACE

    def test_filter_skips_by_byte_count(self, dual_file):
        stats = count_packets(dual_file)
        assert stats.counts == {0: 5, 1: 2}
        sonar = list(iter_packets(dual_file, header_types=[0]))
        assert len(sonar) == 5

    def test_unknown_type_counted_and_skipped(self, tmp_path, caplog):
        pk = [_dual_ping(0), packet(250, b"\x01" * 50), _dual_ping(1)]
        path = write_xtf(tmp_path / "u.xtf", file_header(dual_frequency_infos()), pk)
        with caplog.at_level(logging.DEBUG, logger="sidescanpy.io.xtf.reader"):
            stats = count_packets(path)
        assert stats.counts == {0: 2, 250: 1}
        assert stats.unknown == {250: 1}
        assert "unknown packet type 250" in caplog.text
        data = read_xtf_pings(path)
        assert data.n_pings == 2
        assert data.stats.skipped == {250: 1}

    def test_padding_inside_byte_count_is_skipped(self, tmp_path):
        p0 = _dual_ping(0, pad=40)
        path = write_xtf(tmp_path / "pad.xtf", file_header(dual_frequency_infos()), [p0, _dual_ping(1)])
        assert [p.offset for p in iter_packets(path)] == [1024, 1024 + len(p0)]
        assert read_xtf_pings(path).n_pings == 2

    def test_bad_magic_resyncs(self, tmp_path, caplog):
        garbage = b"\x12\x34" * 37
        pk = [_dual_ping(0), garbage, _dual_ping(1), _dual_ping(2)]
        path = write_xtf(tmp_path / "bad.xtf", file_header(dual_frequency_infos()), pk)
        stats = count_packets(path)
        assert stats.counts == {0: 3}
        assert stats.resyncs == 1
        assert stats.resync_bytes == len(garbage)
        assert "resynced" in caplog.text
        data = read_xtf_pings(path)
        assert list(data.ping_headers["PingNumber"]) == [0, 1, 2]

    def test_zero_byte_count_resyncs(self, tmp_path):
        pk = [_dual_ping(0), packet(1, b"\x00" * 20, nbytes=0), _dual_ping(1)]
        path = write_xtf(tmp_path / "zero.xtf", file_header(dual_frequency_infos()), pk)
        stats = count_packets(path)
        assert stats.counts[0] == 2
        assert stats.resyncs == 1

    def test_garbage_to_eof_stops(self, tmp_path, caplog):
        pk = [_dual_ping(0), b"\x00" * 100]
        path = write_xtf(tmp_path / "tail.xtf", file_header(dual_frequency_infos()), pk)
        stats = count_packets(path)
        assert stats.counts == {0: 1}
        assert "no further 0xFACE" in caplog.text

    def test_truncated_final_packet_warns_and_stops(self, tmp_path, caplog):
        last = _dual_ping(2)
        pk = [_dual_ping(0), _dual_ping(1), last[: len(last) // 2]]
        path = write_xtf(tmp_path / "trunc.xtf", file_header(dual_frequency_infos()), pk)
        stats = count_packets(path)
        assert stats.truncated
        assert stats.counts == {0: 2}
        assert "truncated" in caplog.text
        assert read_xtf_pings(path).n_pings == 2

    def test_trailing_bytes_shorter_than_packet_start(self, tmp_path, caplog):
        pk = [_dual_ping(0), b"\xce\xfa\x00"]
        path = write_xtf(tmp_path / "t.xtf", file_header(dual_frequency_infos()), pk)
        stats = count_packets(path)
        assert stats.truncated and stats.counts == {0: 1}
        assert "trailing bytes" in caplog.text


# ============================================================
# SONAR PINGS
# ============================================================

class TestReadPings:
    def test_two_frequencies_grouped(self, dual_file):
        data = read_xtf_pings(dual_file)
        assert data.n_pings == 5
        assert [g.frequency for g in data.frequency_groups] == [100.0, 400.0]
        lo, hi = data.frequency_groups
        assert (lo.port.channel_number, lo.stbd.channel_number) == (0, 1)
        assert (hi.port.channel_number, hi.stbd.channel_number) == (2, 3)
        assert hi.port.name == "PORT_HI" and hi.port.side == "port"
        for c in range(4):
            ch = data.channels[c]
            assert ch.samples.shape == (5, N_SAMP)
            assert ch.samples.dtype == np.uint16
            for i in range(5):
                np.testing.assert_array_equal(ch.samples[i], ramp(N_SAMP, np.uint16, i * 4 + c))
            assert ch.present.all()
            assert (ch.num_samples == N_SAMP).all()
            assert (ch.headers["ChannelNumber"] == c).all()
        assert list(data.ping_headers["PingNumber"]) == list(range(5))
        np.testing.assert_array_equal(data.packet_offsets, [1024 + 512 + i * (256 + 4 * (64 + 128))
                                                            for i in range(5)])

    @pytest.mark.parametrize(
        "fmt, bps, dtype",
        [
            (0, 1, np.uint8),
            (0, 2, np.uint16),
            (0, 4, np.uint32),
            (8, 1, np.uint8),
            (3, 2, np.uint16),
            (2, 4, np.uint32),
            (5, 4, np.float32),
        ],
    )
    def test_each_sample_format_bit_identical(self, tmp_path, fmt, bps, dtype):
        infos = dual_frequency_infos(bps=bps, fmt=fmt)
        rng = np.random.default_rng(fmt * 10 + bps)
        if dtype is np.float32:
            samples = [rng.normal(size=(3, N_SAMP)).astype(np.float32) for _ in range(4)]
        else:
            samples = [rng.integers(0, np.iinfo(dtype).max, (3, N_SAMP), endpoint=True)
                       .astype(dtype) for _ in range(4)]
        pk = [sonar_packet([(c, samples[c][i]) for c in range(4)], ping_number=i) for i in range(3)]
        path = write_xtf(tmp_path / "f.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        for c in range(4):
            assert data.channels[c].samples.dtype == np.dtype(dtype)
            assert data.channels[c].samples.tobytes() == samples[c].tobytes()

    def test_unsupported_sample_format_raises(self, tmp_path):
        infos = dual_frequency_infos(bps=4, fmt=1)
        path = write_xtf(tmp_path / "ibm.xtf", file_header(infos), [])
        with pytest.raises(ValueError, match="SampleFormat 1"):
            read_xtf_pings(path)

    def test_mid_file_sample_count_change_zero_pads(self, tmp_path):
        counts = [64, 64, 96, 32, 96]
        pk = [_dual_ping(i, n=n) for i, n in enumerate(counts)]
        path = write_xtf(tmp_path / "var.xtf", file_header(dual_frequency_infos()), pk)
        data = read_xtf_pings(path)
        for c in range(4):
            ch = data.channels[c]
            assert ch.samples.shape == (5, 96)
            assert list(ch.num_samples) == counts
            for i, n in enumerate(counts):
                np.testing.assert_array_equal(ch.samples[i, :n], ramp(n, np.uint16, i * 4 + c))
                assert (ch.samples[i, n:] == 0).all()

    def test_fewer_channel_blocks_than_declared(self, tmp_path, caplog):
        # Like the low-frequency-only pings at the start of some EdgeTech
        # files: NumChansToFollow says 4, the packet holds channels 0 and 1.
        pk = [_dual_ping(i, chans=(0, 1), num_chans_to_follow=4) for i in range(3)]
        pk += [_dual_ping(i) for i in range(3, 6)]
        path = write_xtf(tmp_path / "x1.xtf", file_header(dual_frequency_infos()), pk)
        with caplog.at_level(logging.WARNING):
            data = read_xtf_pings(path)
        warns = [r for r in caplog.records if "fewer channel blocks" in r.getMessage()]
        assert len(warns) == 1
        assert "3 of 6" in warns[0].getMessage()
        assert data.n_pings == 6
        for c in (0, 1):
            assert data.channels[c].present.all()
        for c in (2, 3):
            ch = data.channels[c]
            assert list(ch.present) == [False] * 3 + [True] * 3
            assert list(ch.num_samples) == [0] * 3 + [N_SAMP] * 3
            assert (ch.samples[:3] == 0).all()
            assert (ch.headers["ChannelNumber"][:3] == 0).all()
            np.testing.assert_array_equal(ch.samples[4], ramp(N_SAMP, np.uint16, 4 * 4 + c))

    def test_channel_overrunning_packet_is_clipped(self, tmp_path, caplog):
        pk = [sonar_packet([(0, ramp(N_SAMP, np.uint16)), (1, ramp(N_SAMP, np.uint16, 1))],
                           num_samples={1: N_SAMP + 10})]
        infos = dual_frequency_infos()[:2]
        path = write_xtf(tmp_path / "clip.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        assert "clipped" in caplog.text
        assert data.channels[1].num_samples[0] == N_SAMP
        np.testing.assert_array_equal(data.channels[1].samples[0], ramp(N_SAMP, np.uint16, 1))

    def test_zero_num_samples_uses_chan_info_reserved(self, tmp_path):
        infos = [chan_info(PORT, 100.0, reserved=N_SAMP), chan_info(STBD, 100.0, reserved=N_SAMP)]
        pk = [sonar_packet([(0, ramp(N_SAMP, np.uint16)), (1, ramp(N_SAMP, np.uint16, 1))],
                           num_samples={0: 0, 1: 0})]
        path = write_xtf(tmp_path / "legacy.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        np.testing.assert_array_equal(data.channels[1].samples[0], ramp(N_SAMP, np.uint16, 1))

    def test_frequency_falls_back_to_ping_channel_header(self, tmp_path):
        infos = [chan_info(PORT, 0.0), chan_info(STBD, 0.0)]
        pk = [sonar_packet([(0, ramp(8, np.uint16)), (1, ramp(8, np.uint16))],
                           chan_fields={"Frequency": 455}) for _ in range(2)]
        path = write_xtf(tmp_path / "pf.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        assert [g.frequency for g in data.frequency_groups] == [455.0]

    def test_unknown_frequency_pairs_in_channel_order(self, tmp_path, caplog):
        infos = [chan_info(PORT, 0.0), chan_info(STBD, 0.0), chan_info(PORT, 0.0),
                 chan_info(STBD, 0.0)]
        pk = [_dual_ping(0)]
        path = write_xtf(tmp_path / "nf.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        groups = data.frequency_groups
        assert len(groups) == 2 and all(np.isnan(g.frequency) for g in groups)
        assert [(g.port.channel_number, g.stbd.channel_number) for g in groups] == [(0, 1), (2, 3)]
        assert "paired in channel order" in caplog.text

    def test_non_sidescan_channel_not_grouped(self, tmp_path):
        infos = dual_frequency_infos()[:2] + [chan_info(0, 4.0, name="SUBBOTTOM")]
        pk = [_dual_ping(0, chans=(0, 1, 2))]
        path = write_xtf(tmp_path / "sb.xtf", file_header(infos), pk)
        data = read_xtf_pings(path)
        assert data.channels[2].side == "other"
        assert len(data.frequency_groups) == 1

    def test_empty_file_has_no_pings(self, tmp_path):
        path = write_xtf(tmp_path / "e.xtf", file_header(dual_frequency_infos()), [notes_packet()])
        data = read_xtf_pings(path)
        assert data.n_pings == 0 and data.channels == {} and data.frequency_groups == []

    def test_ping_times(self, dual_file):
        data = read_xtf_pings(dual_file)
        expected = [np.datetime64("2005-03-14T05:54:00") + np.timedelta64(i, "s")
                    + np.timedelta64(i * 70, "ms") for i in range(5)]
        np.testing.assert_array_equal(data.times, np.array(expected, dtype="datetime64[ns]"))

    def test_ping_times_invalid_date_is_nat(self):
        from tests.unit.io.xtf.conftest import ping_header
        h = np.stack([ping_header(0), ping_header(1, Month=0)])
        t = ping_times(h)
        assert not np.isnat(t[0]) and np.isnat(t[1])


class TestPyxtfOracleOnSynthetic:
    """pyxtf reads the synthetic files the same way (guards the builders)."""

    def test_header_and_samples_match_pyxtf(self, dual_file):
        pyxtf = pytest.importorskip("pyxtf")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fh, packets = pyxtf.xtf_read(str(dual_file))
        h = read_xtf_header(dual_file)
        for name in ("SonarType", "NavUnits", "NumberOfSonarChannels", "SonarName"):
            assert h[name] == getattr(fh, name)
        assert {int(k): len(v) for k, v in packets.items()} == dict(count_packets(dual_file).counts)
        data = read_xtf_pings(dual_file)
        sonar = packets[pyxtf.XTFHeaderType.sonar]
        for c in range(4):
            np.testing.assert_array_equal(np.vstack([p.data[c] for p in sonar]),
                                          data.channels[c].samples)
        np.testing.assert_array_equal(
            np.array([p.get_time() for p in sonar], dtype="datetime64[ns]"), data.times
        )


def test_read_xtf_not_implemented(dual_file):
    with pytest.raises(NotImplementedError):
        read_xtf(dual_file)
