"""
Unit tests for sidescanpy.io.xtf.split on synthetic XTF files.

A file whose range setting changes mid-file (fixed sample count) is cut
into runs of constant sample rate; short runs are dropped and counted;
pings where a channel is absent never cause a cut; each run converts to a
survey with a single sample rate and segment provenance attributes.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pytest

from sidescanpy.io.xtf import convert, read_xtf, sample_rate_runs, split_by_sample_rate
from tests.unit.io.xtf.conftest import (
    dual_frequency_infos,
    file_header,
    ramp,
    sonar_packet,
    write_xtf,
)

N_SAMP = 64


def _ping(i, range_m=100.0, chans=(0, 1, 2, 3), n=N_SAMP):
    return sonar_packet(
        [(c, ramp(n, np.uint16, i * 4 + c)) for c in chans],
        ping_number=i,
        chan_fields={"SlantRange": range_m, "TimeDuration": range_m / 750.0},
    )


def _file(tmp_path, ranges, chans=None, name="line.xtf"):
    """One ping per entry of ``ranges``; ``chans`` optionally per ping."""
    pk = [
        _ping(i, r, chans[i] if chans is not None else (0, 1, 2, 3))
        for i, r in enumerate(ranges)
    ]
    return read_xtf(write_xtf(tmp_path / name, file_header(dual_frequency_infos()), pk))


@pytest.mark.unit
class TestRuns:
    def test_runs_cut_at_range_change(self, tmp_path):
        data = _file(tmp_path, [100.0] * 7 + [200.0] * 5)
        assert sample_rate_runs(data) == [(0, 7), (7, 12)]

    def test_single_rate_is_one_run(self, tmp_path):
        data = _file(tmp_path, [100.0] * 6)
        assert sample_rate_runs(data) == [(0, 6)]

    def test_change_within_rtol_does_not_cut(self, tmp_path):
        data = _file(tmp_path, [100.0] * 4 + [100.5] * 4)
        assert sample_rate_runs(data, rtol=0.01) == [(0, 8)]
        assert sample_rate_runs(data, rtol=0.001) == [(0, 4), (4, 8)]

    def test_absent_channel_pings_do_not_cut(self, tmp_path):
        # High-frequency pair absent from the first pings (zeroed headers):
        # its first appearance is not a range change.
        chans = [(0, 1)] * 4 + [(0, 1, 2, 3)] * 4 + [(0, 1)] * 2 + [(0, 1, 2, 3)] * 3
        data = _file(tmp_path, [100.0] * 13, chans=chans)
        assert sample_rate_runs(data) == [(0, 13)]

    def test_change_while_one_channel_absent_cuts(self, tmp_path):
        chans = [(0, 1)] * 6 + [(0, 1, 2, 3)] * 6
        data = _file(tmp_path, [100.0] * 3 + [150.0] * 9, chans=chans)
        assert sample_rate_runs(data) == [(0, 3), (3, 12)]


@pytest.mark.unit
class TestSplit:
    def test_range_change_gives_two_segments_with_their_rates(self, tmp_path):
        data = _file(tmp_path, [100.0] * 8 + [200.0] * 6)
        segs = split_by_sample_rate(data, min_pings=3)
        assert [s.n_pings for s in segs] == [8, 6]
        rates = []
        for seg in segs:
            ds = convert(seg).get_frequency("hz_100000").dataset
            rates.append(ds.attrs["sample_rate_hz"])
            assert ds.attrs["xtf_pings_other_sample_rate"] == 0
        assert rates == [pytest.approx(N_SAMP * 750.0 / 100.0), pytest.approx(N_SAMP * 750.0 / 200.0)]

    def test_segments_are_views_of_the_file(self, tmp_path):
        data = _file(tmp_path, [100.0] * 8 + [200.0] * 6)
        segs = split_by_sample_rate(data, min_pings=3)
        for seg in segs:
            for num, ch in seg.channels.items():
                assert np.shares_memory(ch.samples, data.channels[num].samples)
        np.testing.assert_array_equal(segs[1].ping_headers["PingNumber"], np.arange(8, 14))
        assert segs[1].channels[0].samples[0, 0] == data.channels[0].samples[8, 0]

    def test_short_run_dropped_and_counted(self, tmp_path, caplog):
        data = _file(tmp_path, [100.0] * 8 + [150.0] * 2 + [200.0] * 6)
        with caplog.at_level(logging.DEBUG, logger="sidescanpy.io.xtf.split"):
            segs = split_by_sample_rate(data, min_pings=5)
        assert [(s.segment.ping_start, s.segment.ping_stop) for s in segs] == [(0, 8), (10, 16)]
        info = segs[0].segment
        assert (info.index, info.count) == (1, 2)
        assert (info.n_runs_dropped, info.n_pings_dropped, info.n_pings_file) == (1, 2, 16)
        assert any("dropped (2 pings)" in r.getMessage() for r in caplog.records)

    def test_segment_provenance_attrs(self, tmp_path):
        data = _file(tmp_path, [100.0] * 8 + [150.0] * 2 + [200.0] * 6)
        seg = split_by_sample_rate(data, min_pings=5)[1]
        attrs = convert(seg).get_frequency("hz_400000").dataset.attrs
        assert attrs["xtf_segment_index"] == 2
        assert attrs["xtf_segment_count"] == 2
        assert attrs["xtf_segment_ping_start"] == 10
        assert attrs["xtf_segment_ping_stop"] == 16
        assert attrs["xtf_file_pings"] == 16
        assert attrs["xtf_runs_dropped_short"] == 1
        assert attrs["xtf_pings_dropped_short_runs"] == 2

    def test_split_segments_do_not_warn_about_mixed_rates(self, tmp_path, caplog):
        data = _file(tmp_path, [100.0] * 8 + [200.0] * 6)
        with caplog.at_level(logging.WARNING):
            for seg in split_by_sample_rate(data, min_pings=3):
                convert(seg)
        assert not any("sample rate changes" in r.getMessage() for r in caplog.records)

    def test_unsplit_file_still_warns(self, tmp_path, caplog):
        data = _file(tmp_path, [100.0] * 8 + [200.0] * 6)
        with caplog.at_level(logging.WARNING):
            ds = convert(data).get_frequency("hz_100000").dataset
        assert ds.attrs["xtf_pings_other_sample_rate"] == 6
        assert any("sample rate changes" in r.getMessage() for r in caplog.records)

    def test_single_rate_file_kept_whole_even_if_short(self, tmp_path):
        data = _file(tmp_path, [100.0] * 4)
        (seg,) = split_by_sample_rate(data, min_pings=50)
        assert seg.n_pings == 4
        assert (seg.segment.index, seg.segment.count, seg.segment.n_pings_dropped) == (1, 1, 0)

    def test_all_runs_short_gives_nothing(self, tmp_path, caplog):
        data = _file(tmp_path, [100.0] * 3 + [200.0] * 3)
        with caplog.at_level(logging.WARNING):
            assert split_by_sample_rate(data, min_pings=5) == []
        assert any("nothing kept" in r.getMessage() for r in caplog.records)

    def test_segment_without_a_frequency_leaves_it_out(self, tmp_path):
        # The first run holds only the low-frequency pair.
        chans = [(0, 1)] * 5 + [(0, 1, 2, 3)] * 6
        data = _file(tmp_path, [100.0] * 5 + [200.0] * 6, chans=chans)
        first, second = split_by_sample_rate(data, min_pings=3)
        assert [g.frequency for g in first.frequency_groups] == [100.0]
        assert sorted(first.channels) == [0, 1]
        assert convert(first).frequencies == ["hz_100000"]
        assert convert(second).frequencies == ["hz_100000", "hz_400000"]

    def test_samples_trimmed_to_segment_width(self, tmp_path):
        pk = [_ping(i, 100.0, n=64) for i in range(4)] + [_ping(i, 200.0, n=32) for i in range(4, 8)]
        data = read_xtf(write_xtf(tmp_path / "w.xtf", file_header(dual_frequency_infos()), pk))
        segs = split_by_sample_rate(data, min_pings=2)
        assert segs[1].channels[0].samples.shape == (4, 32)
        assert convert(segs[1]).get_frequency("hz_100000").n_samples == 32
