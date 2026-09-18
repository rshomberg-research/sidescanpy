"""
Unit tests for sidescan-analyze — investigate (promoted from
project-a/scripts/investigate_dataset.py) and tvg-fit (the adopted
auto-window TVG decay-fit recipe, using stage4_decay's own
r_ref_m="auto"/r_fit_min_m="auto" support).

2026 rshomberg@uri.edu
"""

import argparse
import json

import numpy as np
import pytest

from sidescanpy.cli import analyze
from sidescanpy.io.hdf5 import write_survey
from tests.conftest import make_minimal_sidescan_dataset
from sidescanpy.core.survey import SideScanSurvey


def _decay_survey(n_pings=10, n_samples=800):
    """A survey with a real range-dependent decay in backscatter_raw
    (so stage4_decay has something non-degenerate to fit) and altitude
    small enough that slant range genuinely exceeds it."""
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz_start=1_495_000)
    rng = np.random.default_rng(0)
    ranges = np.arange(n_samples) * 0.02  # sample_interval placeholder, just for a decay shape
    decay = 10.0 * np.exp(-ranges / 5.0) + 0.1
    noise = rng.normal(0, 0.01, size=(n_pings, 2, n_samples))
    data = decay[None, None, :] + noise
    ds["backscatter_raw"] = (["ping", "channel", "sample"], np.clip(data, 1e-6, None).astype(np.float32))
    ds["backscatter_intensity"] = ds["backscatter_raw"].copy()
    ds["altitude_m"] = ("ping", np.full(n_pings, 0.5, dtype=np.float32))
    survey.add_frequency("hz_1495000", ds)
    return survey


class TestRunInvestigate:
    def _args(self, tmp_path, input_path, **overrides):
        defaults = dict(
            input=input_path, frequency=None, all_frequencies=False,
            value_var="backscatter_raw", pings="0:100", nadir_n_samples=10,
            stage3_bin_width_m=2.0, stage4_bin_width_m=5.0,
            stage4_db_reference_percentile=99.0, snr_threshold=3.0,
            reject_outliers=False, outlier_mad_k=3.5, stage4_fit_value_col="mean",
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_input_returns_1(self, tmp_path):
        args = self._args(tmp_path, tmp_path / "nope.ss.h5")
        assert analyze.run_investigate(args) == 1

    def test_single_frequency_writes_summary(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        survey = _decay_survey()
        write_survey(survey, path)

        args = self._args(tmp_path, path, frequency="hz_1495000")
        assert analyze.run_investigate(args) == 0
        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert len(summary) == 1
        assert summary[0]["freq_key"] == "hz_1495000" or "freq_key" in summary[0]

    def test_multiple_frequencies_requires_all_frequencies_flag(self, tmp_path):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        for freq_key, freq_hz in (("hz_1495000", 1_495_000), ("hz_547500", 547_500)):
            ds = make_minimal_sidescan_dataset(n_pings=10, n_samples=100, freq_hz_start=freq_hz)
            survey.add_frequency(freq_key, ds)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        args = self._args(tmp_path, path)
        assert analyze.run_investigate(args) == 1


class TestRunTvgFit:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            ss_h5=ss_h5_path, frequency=1495000, pings=(0, None),
            reconstruct_raw=False, bin_width_m=1.0, r_ref_m="auto",
            r_fit_min_m="auto", range_var="slant_range_m",
            usable_range_m=None, output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_fits_decay_and_writes_png(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_tvg_fit(args) == 0
        pngs = list((tmp_path / "out").glob("*.png"))
        assert len(pngs) == 1

    def test_not_raw_without_reconstruct_flag_returns_1(self, tmp_path):
        from sidescanpy.processing import run_correction_steps

        corrected = run_correction_steps(_decay_survey(), {"tvg"})
        path = tmp_path / "a.ss.h5"
        write_survey(corrected, path)

        args = self._args(tmp_path, path)
        assert analyze.run_tvg_fit(args) == 1

    def test_reconstruct_raw_recovers_from_already_corrected_file(self, tmp_path):
        from sidescanpy.processing import run_correction_steps

        corrected = run_correction_steps(_decay_survey(), {"tvg"})
        path = tmp_path / "a.ss.h5"
        write_survey(corrected, path)

        args = self._args(tmp_path, path, reconstruct_raw=True)
        assert analyze.run_tvg_fit(args) == 0

    def test_flat_data_without_usable_range_override_returns_1(self, tmp_path):
        """A flat (no real range-dependent decay) backscatter array is
        the real Klein SDF failure mode this survives -- Stage 3's SNR
        threshold never trips against the noise floor, so it can't
        auto-determine usable_range_m."""
        flat = _decay_survey()
        ds = flat.get_frequency("hz_1495000").dataset
        ds["backscatter_raw"] = ds["backscatter_raw"] * 0 + 1.0  # constant, no decay
        ds["backscatter_intensity"] = ds["backscatter_raw"].copy()
        flat.add_frequency("hz_1495000", ds)

        path = tmp_path / "a.ss.h5"
        write_survey(flat, path)

        args = self._args(tmp_path, path)
        assert analyze.run_tvg_fit(args) == 1

    def test_usable_range_m_override_bypasses_stage3(self, tmp_path):
        flat = _decay_survey()
        ds = flat.get_frequency("hz_1495000").dataset
        ds["backscatter_raw"] = ds["backscatter_raw"] * 0 + 1.0
        ds["backscatter_intensity"] = ds["backscatter_raw"].copy()
        flat.add_frequency("hz_1495000", ds)

        path = tmp_path / "a.ss.h5"
        write_survey(flat, path)

        args = self._args(tmp_path, path, usable_range_m=10.0)
        assert analyze.run_tvg_fit(args) == 0
        pngs = list((tmp_path / "out").glob("*.png"))
        assert len(pngs) == 1

    def test_numeric_r_ref_m_override(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path, r_ref_m=6.0, r_fit_min_m=1.0)
        assert analyze.run_tvg_fit(args) == 0

    def test_literal_freq_key_fallback_for_non_hz_naming(self, tmp_path):
        """Not every consumer uses JSF/SDF's "hz_<freq>" freq_key
        convention (e.g. vendor-format's "sensor1_data1") -- --frequency must
        resolve to a literal freq_key when "hz_<value>" doesn't exist,
        not assume every survey uses that naming."""
        survey = _decay_survey()
        ds = survey.get_frequency("hz_1495000").dataset
        survey = SideScanSurvey(source_files=["a.h5"], source_format="vendor_h5", survey_name="a")
        survey.add_frequency("sensor1_data1", ds)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        args = self._args(tmp_path, path, frequency="sensor1_data1")
        assert analyze.run_tvg_fit(args) == 0
        pngs = list((tmp_path / "out").glob("*.png"))
        assert len(pngs) == 1


class TestParseHelpers:
    def test_parse_ping_range(self):
        assert analyze._parse_ping_range("0:100") == (0, 100)
        assert analyze._parse_ping_range(":100") == (0, 100)
        assert analyze._parse_ping_range("50:") == (50, None)

    def test_parse_ping_range_requires_colon(self):
        with pytest.raises(argparse.ArgumentTypeError):
            analyze._parse_ping_range("100")

    def test_parse_numeric_or_literal(self):
        assert analyze._parse_numeric_or_literal(None) is None
        assert analyze._parse_numeric_or_literal("auto") == "auto"
        assert analyze._parse_numeric_or_literal("6.0") == 6.0
