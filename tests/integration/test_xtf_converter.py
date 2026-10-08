"""
Real-data tests for the XTF converter.

Set ``SIDESCANPY_XTF_DATA`` to a directory of XTF files to run them; they
skip otherwise. Files are chosen by size and content, never by name (see
``test_xtf_reader.py``): the two smallest files, and the smallest file
whose packets hold fewer channel blocks than declared, if any.

Checks: the survey validates, positions and times equal those decoded by
pyxtf (when installed), slant ranges agree with ``TimeDuration * c / 2``,
and the ``.ss.h5`` round trip is identical.

2026 rshomberg@uri.edu
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.io.xtf import convert, read_xtf
from sidescanpy.processing.sample_axis import sample_interval_m
from tests.integration.test_xtf_reader import DATA_FILES, FAST_PARAMS, N_FAST, _params

pytestmark = [pytest.mark.integration, pytest.mark.requires_data]

OTHER_PARAMS = _params(DATA_FILES[N_FAST:], slow=True) if len(DATA_FILES) > N_FAST else []


def _validate(survey) -> None:
    for ch in survey.freq_channels.values():
        ch.dataset.sidescan.validate()
    survey.navigation.navigation_record.validate()
    survey.attitude.attitude_record.validate()


@pytest.mark.parametrize("path", FAST_PARAMS)
def test_converts_and_matches_pyxtf_positions(path: Path, tmp_path):
    data = read_xtf(path)
    survey = convert(data)
    _validate(survey)
    assert survey.source_format == "xtf"
    assert [g.frequency for g in data.frequency_groups] == [120.0, 410.0]
    assert survey.frequencies == ["hz_120000", "hz_410000"]

    pyxtf = pytest.importorskip("pyxtf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, packets = pyxtf.xtf_read(str(path))
    sonar = packets[pyxtf.XTFHeaderType.sonar]
    lat = np.array([p.SensorYcoordinate for p in sonar])
    lon = np.array([p.SensorXcoordinate for p in sonar])
    t = np.array([p.get_time() for p in sonar], dtype="datetime64[ns]")
    for ch in survey.freq_channels.values():
        ds = ch.dataset
        assert ds.sizes["ping"] == len(sonar)
        np.testing.assert_array_equal(ds["latitude"].values, lat)
        np.testing.assert_array_equal(ds["longitude"].values, lon)
        np.testing.assert_array_equal(ds["time"].values, t)
        # Port/starboard samples are pyxtf's, as float32.
        c0 = int(ds.attrs["xtf_channel_numbers"].split(",")[0])
        np.testing.assert_array_equal(
            ds["backscatter_intensity"].values[:, 0],
            np.vstack([p.data[c0] for p in sonar]).astype(np.float32),
        )
    np.testing.assert_array_equal(survey.navigation["latitude"].values, lat)
    np.testing.assert_array_equal(survey.navigation["longitude"].values, lon)


@pytest.mark.parametrize("path", FAST_PARAMS)
def test_ranges_and_round_trip(path: Path, tmp_path):
    data = read_xtf(path)
    survey = convert(data)
    for group in data.frequency_groups:
        ds = survey.get_frequency(f"hz_{int(group.frequency * 1000)}").dataset
        if ds.attrs["xtf_pings_other_sample_rate"]:
            continue
        h = group.port.headers[group.port.present]
        c = ds["sound_vel_ms"].values.astype(np.float64)
        expected = h["TimeDuration"].astype(np.float64) * c / 2
        np.testing.assert_allclose(sample_interval_m(ds) * ds.sizes["sample"], expected,
                                   rtol=1e-5)
        np.testing.assert_allclose(expected, h["SlantRange"], rtol=1e-3)
    out = tmp_path / "x.ss.h5"
    write_survey(survey, out)
    loaded = read_survey(out)
    for fk in survey.frequencies:
        xr.testing.assert_identical(loaded.get_frequency(fk).dataset,
                                    survey.get_frequency(fk).dataset)
    xr.testing.assert_identical(loaded.navigation, survey.navigation)


@pytest.mark.parametrize("path", OTHER_PARAMS or [pytest.param(None, marks=pytest.mark.skip(
    reason="fewer than three XTF files"))])
def test_larger_files_convert(path: Path):
    data = read_xtf(path)
    survey = convert(data)
    _validate(survey)
    lo, hi = (survey.get_frequency(fk).dataset for fk in survey.frequencies)
    assert lo.sizes["ping"] == data.n_pings
    if data.block_stats.fewer_channels:
        # Pings carrying only the low-frequency pair are left out of the
        # high frequency, not NaN-filled.
        absent = int((~data.frequency_groups[1].port.present).sum())
        assert absent > 0
        assert hi.sizes["ping"] == data.n_pings - absent
        assert hi.attrs["xtf_pings_absent"] == absent
        assert hi.attrs["xtf_packets_fewer_channels"] == data.block_stats.fewer_channels
        assert not np.isnan(hi["backscatter_intensity"].values).all(axis=(1, 2)).any()
    else:
        assert hi.sizes["ping"] == data.n_pings
