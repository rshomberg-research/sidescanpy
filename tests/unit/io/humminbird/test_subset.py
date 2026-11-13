"""
Unit tests for sidescanpy.io.humminbird.subset on synthetic recordings.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.io.humminbird import read_humminbird, write_subset
from tests.unit.io.humminbird.conftest import write_recording


@pytest.mark.unit
class TestWriteSubset:
    def test_keeps_the_requested_pings_of_every_beam(self, tmp_path):
        src = write_recording(tmp_path / "src", n_pings=6, beams=(1, 2, 3))
        dst = tmp_path / "out" / "CUT.DAT"
        assert write_subset(src, dst, n_pings=3, start_ping=2) == 3
        full, cut = read_humminbird(src), read_humminbird(dst)
        assert dst.read_bytes() == src.read_bytes()
        for b in (1, 2, 3):
            assert cut.beams[b].fields["time_ms"].tolist() == [100, 150, 200]
            np.testing.assert_array_equal(cut.beams[b].samples(), full.beams[b].samples(np.arange(2, 5)))

    def test_frames_copied_byte_for_byte(self, tmp_path):
        src = write_recording(tmp_path / "src", n_pings=6)
        dst = tmp_path / "out" / "CUT.DAT"
        write_subset(src, dst, n_pings=2, start_ping=1)
        assert (tmp_path / "out" / "CUT" / "B002.SON").read_bytes() in (src.with_suffix("") / "B002.SON").read_bytes()

    def test_runs_to_end(self, tmp_path):
        src = write_recording(tmp_path / "src", n_pings=4)
        assert write_subset(src, tmp_path / "o" / "C.DAT", n_pings=10, start_ping=2) == 2

    def test_bad_arguments(self, tmp_path):
        src = write_recording(tmp_path / "src", n_pings=4)
        with pytest.raises(ValueError, match="fewer than"):
            write_subset(src, tmp_path / "o" / "C.DAT", n_pings=1, start_ping=4)
        with pytest.raises(ValueError, match="n_pings"):
            write_subset(src, tmp_path / "o" / "C.DAT", n_pings=0)
