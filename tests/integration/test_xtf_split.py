"""
Real-data tests for splitting XTF files at range-setting changes.

Set ``SIDESCANPY_XTF_DATA`` to a directory of XTF files to run them; they
skip otherwise. Every file is tested (the two smallest without ``slow``),
whether or not its range setting changes; nothing depends on file names.

Checks, per file: the kept segments and the dropped short runs account
for every ping, in order and without overlap; every segment converts to
a valid survey with a single sample rate per frequency, matching each
ping's ``NumSamples / TimeDuration``; segment provenance attributes are
consistent.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io.xtf import convert, read_xtf, sample_rate_runs, split_by_sample_rate
from sidescanpy.io.xtf.split import SPLIT_MIN_PINGS, SPLIT_RTOL
from tests.integration.test_xtf_reader import ALL_PARAMS

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]


@pytest.mark.parametrize("path", ALL_PARAMS)
def test_segments_have_one_sample_rate_and_account_for_all_pings(path: Path):
    data = read_xtf(path)
    runs = sample_rate_runs(data)
    segs = split_by_sample_rate(data)
    assert segs

    # Runs tile the file; kept segments are exactly the long runs.
    assert runs[0][0] == 0 and runs[-1][1] == data.n_pings
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    long_runs = runs if len(runs) == 1 else [r for r in runs if r[1] - r[0] >= SPLIT_MIN_PINGS]
    assert [(s.segment.ping_start, s.segment.ping_stop) for s in segs] == long_runs
    n_dropped = data.n_pings - sum(s.n_pings for s in segs)
    assert all(s.segment.n_pings_dropped == n_dropped for s in segs)
    assert all(s.segment.count == len(segs) for s in segs)

    for seg in segs:
        survey = convert(seg)
        for ch in survey.freq_channels.values():
            ch.dataset.sidescan.validate()
        for group in seg.frequency_groups:
            ds = survey.get_frequency(f"hz_{int(round(group.frequency * 1000))}").dataset
            assert ds.attrs["xtf_pings_other_sample_rate"] == 0
            assert ds.attrs["xtf_segment_index"] == seg.segment.index
            assert ds.attrs["xtf_pings_dropped_short_runs"] == n_dropped
            fs = ds.attrs["sample_rate_hz"]
            for side in (group.port, group.stbd):
                if side is None:
                    continue
                h = side.headers[side.present]
                rate = side.num_samples[side.present] / h["TimeDuration"].astype(np.float64)
                np.testing.assert_allclose(rate, fs, rtol=2 * SPLIT_RTOL)
