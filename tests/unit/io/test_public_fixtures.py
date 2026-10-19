"""
Real recordings from openly licensed public datasets.

Small byte-for-byte cuts. The files are not in the repository: point
``SIDESCANPY_PUBLIC_DATA`` at a directory holding them, or these tests
skip. Sources, licences and checksums are in ``tests/data/public/SOURCES.md``.
The JNCC Wight-Barfleur JSF and its two XTF files hold the same pings,
recorded once and written by EdgeTech Discover in both formats, so the
two readers must agree on them.

2026 rshomberg@uri.edu
"""

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest

from sidescanpy.io import read_and_convert
from sidescanpy.processing import altitude_from_bottom_track, correct_survey

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "public" / "SOURCES.md"
PUBLIC = Path(os.environ.get("SIDESCANPY_PUBLIC_DATA", "/nonexistent"))

pytestmark = pytest.mark.skipif(
    not PUBLIC.is_dir(), reason="set SIDESCANPY_PUBLIC_DATA to the public fixture directory"
)
WIGHT_JSF = PUBLIC / "jncc_wight_barfleur_20130321174522_p100-107.jsf"
WIGHT_XTF_HIGH = PUBLIC / "jncc_wight_barfleur_20130321174522H_p100-107.xtf"
WIGHT_XTF_LOW = PUBLIC / "jncc_wight_barfleur_20130321174522L_p100-107.xtf"
BENTHOS_XTF = PUBLIC / "jncc_mid_irish_sea_MIS13_2_p0-29.xtf"
KLEIN_XTF = PUBLIC / "usgs_grand_bay_15CCT03_SSS_153_150602205800_p0-59.xtf"

ALL = [WIGHT_JSF, WIGHT_XTF_HIGH, WIGHT_XTF_LOW, BENTHOS_XTF, KLEIN_XTF]


def _read(path):
    return read_and_convert([path], fmt=path.suffix[1:])


def _only(survey):
    (key,) = survey.frequencies
    return survey.get_frequency(key).dataset


@pytest.fixture(scope="module")
def wight_jsf():
    return _read(WIGHT_JSF)


class TestFixtureFiles:
    def test_every_file_is_listed_with_its_checksum(self):
        listed = dict(
            (name, digest)
            for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+)$", MANIFEST.read_text(), re.M)
        )
        on_disk = {p.name for p in PUBLIC.iterdir() if p.suffix in (".jsf", ".xtf")}
        assert on_disk == set(listed)
        for name, digest in listed.items():
            assert hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() == digest, name


class TestReadEachWriter:
    @pytest.mark.parametrize(("path", "keys", "n_pings", "n_samples"), [
        (WIGHT_JSF, ["hz_218740", "hz_521000"], 8, [11568, 23140]),
        (WIGHT_XTF_HIGH, ["hz_547500"], 8, [23140]),
        (WIGHT_XTF_LOW, ["hz_230240"], 8, [11568]),
        (BENTHOS_XTF, ["hz_100000", "hz_400000"], 30, [4096, 4096]),
        (KLEIN_XTF, ["hz_100000"], 60, [4096]),
    ])
    def test_frequencies_and_shape(self, path, keys, n_pings, n_samples):
        survey = _read(path)
        assert survey.frequencies == keys
        for key, n in zip(keys, n_samples, strict=True):
            ch = survey.get_frequency(key)
            assert (ch.n_pings, ch.n_channels, ch.n_samples) == (n_pings, 2, n)
            assert np.isfinite(ch.dataset["latitude"].values).all()

    @pytest.mark.parametrize(("path", "program", "weight_max"), [
        (WIGHT_XTF_HIGH, "Discover 7.22", 6),
        (BENTHOS_XTF, "Isis 7.0.41", 0),
        (KLEIN_XTF, "KleinXlt 4.0", 0),
    ])
    def test_xtf_writer_and_weight(self, path, program, weight_max):
        survey = _read(path)
        ds = survey.get_frequency(survey.frequencies[0]).dataset
        assert ds.attrs["xtf_recording_program"] == program
        assert ds.attrs["xtf_weight_max"] == weight_max

    @pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
    def test_corrects_with_bottom_track(self, path):
        survey = correct_survey(_read(path), steps=("interpolate_aux", "bottom_track", "tvg", "geometric", "georef"))
        for key in survey.frequencies:
            ds = survey.get_frequency(key).dataset
            assert np.isfinite(ds["latitude_per_sample"].values).any(), key


class TestJsfAndXtfAgree:
    """Same pings, written by Discover as JSF and as XTF."""

    @pytest.mark.parametrize(("jsf_key", "xtf_path"), [("hz_521000", WIGHT_XTF_HIGH), ("hz_218740", WIGHT_XTF_LOW)])
    def test_same_pings_times_navigation_and_samples(self, wight_jsf, jsf_key, xtf_path):
        a = wight_jsf.get_frequency(jsf_key).dataset
        b = _only(_read(xtf_path))
        np.testing.assert_array_equal(a["ping_number"].values, b["ping_number"].values)
        dt = np.abs(a["time"].values - b["time"].values).astype("timedelta64[ms]").astype(int)
        assert dt.max() <= 10  # XTF times have 10 ms resolution
        np.testing.assert_allclose(a["latitude"].values, b["latitude"].values, atol=1e-5)
        np.testing.assert_allclose(a["longitude"].values, b["longitude"].values, atol=1e-5)
        np.testing.assert_array_equal(a["heading_deg"].values, b["heading_deg"].values)
        np.testing.assert_array_equal(a["backscatter_intensity"].values, b["backscatter_intensity"].values)


class TestBottomTrackOnRecordedAltitude:
    def test_benthos_tracked_altitude_matches_altimeter(self):
        for key in ("hz_100000", "hz_400000"):
            ds = _read(BENTHOS_XTF).get_frequency(key).dataset
            recorded = ds["altitude_m"].values.astype(float)
            tracked = altitude_from_bottom_track(ds, replace="all")["altitude_m"].values.astype(float)
            assert np.median(np.abs(tracked - recorded)) < 0.3, key

    def test_discover_files_have_no_altitude(self, wight_jsf):
        # The reason the bottom_track step exists: 0 in the JSF, nothing in the XTF.
        assert (wight_jsf.get_frequency("hz_521000").dataset["altitude_m"].values == 0).all()
        assert np.isnan(_only(_read(WIGHT_XTF_HIGH))["altitude_m"].values).all()
