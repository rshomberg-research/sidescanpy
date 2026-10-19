"""
Unit tests for sidescanpy.io.jsf.subset, on the public JSF fixture.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io import read_and_convert
from sidescanpy.io.jsf import write_subset

SRC = Path(__file__).resolve().parents[3] / "data" / "public" / "jncc_wight_barfleur_20130321174522_p100-107.jsf"


class TestWriteSubset:
    def test_keeps_every_channel_and_subsystem_of_the_pings(self, tmp_path):
        assert write_subset(SRC, tmp_path / "out.jsf", n_pings=3, start_ping=2) == 3
        full = read_and_convert([SRC], fmt="jsf")
        cut = read_and_convert([tmp_path / "out.jsf"], fmt="jsf")
        assert cut.frequencies == full.frequencies
        for key in full.frequencies:
            a, b = full.get_frequency(key).dataset, cut.get_frequency(key).dataset
            np.testing.assert_array_equal(b["ping_number"].values, a["ping_number"].values[2:5])
            np.testing.assert_array_equal(b["backscatter_intensity"].values, a["backscatter_intensity"].values[2:5])

    def test_output_is_a_contiguous_span(self, tmp_path):
        write_subset(SRC, tmp_path / "out.jsf", n_pings=2, start_ping=1)
        assert (tmp_path / "out.jsf").read_bytes() in SRC.read_bytes()

    def test_runs_to_end_of_file(self, tmp_path):
        assert write_subset(SRC, tmp_path / "out.jsf", n_pings=50, start_ping=6) == 2

    def test_bad_arguments(self, tmp_path):
        with pytest.raises(ValueError, match="fewer than"):
            write_subset(SRC, tmp_path / "out.jsf", n_pings=1, start_ping=8)
        with pytest.raises(ValueError, match="n_pings"):
            write_subset(SRC, tmp_path / "out.jsf", n_pings=0)
