"""
Unit tests for sidescanpy.io.xtf.subset on synthetic XTF files.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.io.xtf import read_xtf_pings, write_subset
from tests.unit.io.xtf.conftest import (
    dual_frequency_infos,
    file_header,
    notes_packet,
    ramp,
    sonar_packet,
    write_xtf,
)


def _file(tmp_path, n=6):
    pings = [sonar_packet([(c, ramp(32, np.uint16, i * 4 + c)) for c in range(4)], ping_number=i) for i in range(n)]
    packets = [notes_packet()] + [p for i, ping in enumerate(pings) for p in ([ping, notes_packet()] if i == 2 else [ping])]
    return write_xtf(tmp_path / "src.xtf", file_header(dual_frequency_infos()), packets)


@pytest.mark.unit
class TestWriteSubset:
    def test_keeps_the_requested_pings_byte_for_byte(self, tmp_path):
        src = _file(tmp_path)
        assert write_subset(src, tmp_path / "out.xtf", n_pings=3, start_ping=1) == 3
        full, cut = read_xtf_pings(src), read_xtf_pings(tmp_path / "out.xtf")
        assert cut.ping_headers["PingNumber"].tolist() == [1, 2, 3]
        for ch in full.channels:
            np.testing.assert_array_equal(cut.channels[ch].samples, full.channels[ch].samples[1:4])

    def test_header_copied_and_inner_packets_kept(self, tmp_path):
        src = _file(tmp_path)
        write_subset(src, tmp_path / "out.xtf", n_pings=3, start_ping=1)
        data, out = src.read_bytes(), (tmp_path / "out.xtf").read_bytes()
        assert out[:1024] == data[:1024]
        assert out[1024:] in data  # one contiguous span, the notes packet after ping 2 included
        assert out.count(b"synthetic note") == 1

    def test_runs_to_end_of_file(self, tmp_path):
        src = _file(tmp_path)
        assert write_subset(src, tmp_path / "out.xtf", n_pings=10, start_ping=4) == 2
        assert read_xtf_pings(tmp_path / "out.xtf").ping_headers["PingNumber"].tolist() == [4, 5]

    def test_bad_arguments(self, tmp_path):
        src = _file(tmp_path)
        with pytest.raises(ValueError, match="fewer than"):
            write_subset(src, tmp_path / "out.xtf", n_pings=1, start_ping=6)
        with pytest.raises(ValueError, match="n_pings"):
            write_subset(src, tmp_path / "out.xtf", n_pings=0)
