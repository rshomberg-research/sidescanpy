"""
Unit tests for sidescan-analyze — investigate (promoted from
project-a/scripts/investigate_dataset.py) and tvg-fit (the adopted
auto-window TVG decay-fit recipe, using stage4_decay's own
r_ref_m="auto"/r_fit_min_m="auto" support).

2026 rshomberg@uri.edu
"""

import argparse
import json
import warnings

import numpy as np
import pytest

from sidescanpy.cli import analyze
from sidescanpy.core import HDF5_CHUNK_PING
from sidescanpy.io.hdf5 import read_survey, write_survey
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


class TestRunSegment:
    def _args(self, tmp_path, input_path, **overrides):
        defaults = dict(
            input=input_path, frequency=None, pings=(0, None),
            gap_factor=8.0, time_gap_factor=8.0, sound_vel_rel_tol=0.05,
            garbage_value_var="backscatter_raw", garbage_min_run_pings=20, garbage_degenerate_frac=0.999,
            stability_window_pings=50, stability_threshold=1.0, stability_min_run_pings=25,
            min_segment_pings=50,
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _clean_survey(self, n_pings=200, n_samples=50):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz_start=1_495_000)
        ds["latitude"] = ("ping", np.linspace(41.0000, 41.0100, n_pings))
        ds["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
        survey.add_frequency("hz_1495000", ds)
        return survey

    def test_missing_input_returns_1(self, tmp_path):
        args = self._args(tmp_path, tmp_path / "nope.ss.h5")
        assert analyze.run_segment(args) == 1

    def test_clean_survey_writes_one_section_and_report(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(self._clean_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_segment(args) == 0

        out_dir = tmp_path / "out"
        assert (out_dir / "section1.ss.h5").exists()
        assert (out_dir / "segments.json").exists()
        assert (out_dir / "overview.png").exists()
        report = json.loads((out_dir / "segments.json").read_text())
        assert len(report["segments"]) == 1
        assert report["segments"][0]["status"] == "clean"

    def test_garbage_run_produces_excluded_segment_not_written(self, tmp_path):
        survey = self._clean_survey(n_pings=200)
        ds = survey.get_frequency("hz_1495000").dataset
        arr = ds["backscatter_raw"].values.copy()
        arr[80:130] = 0.0
        ds["backscatter_raw"] = (("ping", "channel", "sample"), arr)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        args = self._args(tmp_path, path, garbage_min_run_pings=20)
        assert analyze.run_segment(args) == 0

        out_dir = tmp_path / "out"
        report = json.loads((out_dir / "segments.json").read_text())
        statuses = {s["status"] for s in report["segments"]}
        assert "excluded" in statuses
        n_written = len(list(out_dir.glob("section*.ss.h5")))
        n_clean = sum(1 for s in report["segments"] if s["status"] == "clean")
        assert n_written == n_clean

    def test_explicit_frequency_used(self, tmp_path):
        survey = self._clean_survey()
        ds2 = make_minimal_sidescan_dataset(n_pings=200, n_samples=50, freq_hz_start=547_500)
        ds2["latitude"] = ("ping", np.linspace(41.0000, 41.0100, 200))
        ds2["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, 200))
        survey.add_frequency("hz_547500", ds2)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        args = self._args(tmp_path, path, frequency="hz_547500")
        assert analyze.run_segment(args) == 0
        assert (tmp_path / "out" / "section1.ss.h5").exists()

    def test_pings_bound_offsets_written_section_correctly(self, tmp_path):
        """Real bug caught while adding --pings: writing sliced the
        ORIGINAL full-range survey by find_segments' own window-relative
        (0-indexed-from-start) ping numbers, silently writing the wrong
        pings whenever start != 0."""
        from sidescanpy.io.hdf5 import read_survey

        survey = self._clean_survey(n_pings=400)
        expected_lat_at_100 = survey.get_frequency("hz_1495000").dataset["latitude"].values[100]
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        args = self._args(tmp_path, path, pings=(100, 300))
        assert analyze.run_segment(args) == 0

        written = read_survey(tmp_path / "out" / "section1.ss.h5")
        written_ds = written.get_frequency("hz_1495000").dataset
        assert written_ds.sizes["ping"] == 200
        assert float(written_ds["latitude"].values[0]) == pytest.approx(float(expected_lat_at_100))


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


class TestRunNadir:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            input=ss_h5_path, frequency=1495000, pings=(0, None), value_var="backscatter_raw",
            nadir_n_samples=10, strong_return_min_sample_buffer=0.0, strong_return_peak_percentile=100.0,
            min_first_bottom_return_sample=None, boundary_smoothing_window=15, verbose_diagnostics=False,
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_zoom_plot_and_metrics(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_nadir(args) == 0

        pngs = list((tmp_path / "out").glob("*_nadir_zoom.png"))
        assert len(pngs) == 1
        metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
        assert metrics["stage"] == "stage2_nadir"
        assert metrics["classification"] is not None

    def test_verbose_diagnostics_also_writes_waterfall(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path, verbose_diagnostics=True)
        assert analyze.run_nadir(args) == 0
        assert list((tmp_path / "out").glob("*_nadir_waterfall.png"))

    def test_no_verbose_diagnostics_by_default(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_nadir(args) == 0
        assert not list((tmp_path / "out").glob("*_nadir_waterfall.png"))


class TestRunBeamGeometry:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            input=ss_h5_path, frequency=1495000, pings=(0, None), value_var="backscatter_raw",
            nadir_n_samples=10, stage3_bin_width_m=2.0, snr_threshold=3.0,
            near_angle_max_std_deg=15.0, min_valid_fraction=0.2,
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_diagram_and_metrics(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_beam_geometry(args) == 0

        pngs = list((tmp_path / "out").glob("*_beam_geometry.png"))
        assert len(pngs) == 1
        metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
        assert metrics["stage"] == "stage2_5_beam_geometry"


class TestRunUsableRange:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            input=ss_h5_path, frequency=1495000, pings=(0, None), value_var="backscatter_raw",
            range_var="slant_range_m", bin_width_m=1.0, snr_threshold=3.0,
            reject_outliers=False, outlier_mad_k=3.5,
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_plot_and_metrics(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_usable_range(args) == 0

        pngs = list((tmp_path / "out").glob("*_usable_range.png"))
        assert len(pngs) == 1
        metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
        assert metrics["stage"] == "stage3_usable_range"
        assert "usable_range_m" in metrics["metrics"]


class TestRunUnits:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            input=ss_h5_path, frequency=1495000, pings=(0, None), value_var="backscatter_raw",
            output_dir=tmp_path / "out",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_histogram_and_metrics(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_units(args) == 0

        pngs = list((tmp_path / "out").glob("*_units.png"))
        assert len(pngs) == 1
        metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
        assert metrics["stage"] == "stage5_units"
        assert metrics["classification"] is not None


class TestRunRecommendConfig:
    def _args(self, tmp_path, ss_h5_path, **overrides):
        defaults = dict(
            input=ss_h5_path, frequency=1495000, pings=(0, None), value_var="backscatter_raw",
            nadir_n_samples=10, bin_width_m=1.0, snr_threshold=3.0,
            output=tmp_path / "out" / "recommended_config.yaml",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_input_returns_1(self, tmp_path):
        args = self._args(tmp_path, tmp_path / "nope.ss.h5")
        assert analyze.run_recommend_config(args) == 1

    def test_writes_yaml_with_resolution_m(self, tmp_path):
        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_recommend_config(args) == 0

        text = (tmp_path / "out" / "recommended_config.yaml").read_text()
        assert "resolution_m:" in text
        assert "sidescan-correct" in text
        assert "sidescan-export chip" in text

    def test_output_is_valid_yaml(self, tmp_path):
        import yaml

        path = tmp_path / "a.ss.h5"
        write_survey(_decay_survey(), path)

        args = self._args(tmp_path, path)
        assert analyze.run_recommend_config(args) == 0
        parsed = yaml.safe_load((tmp_path / "out" / "recommended_config.yaml").read_text())
        assert isinstance(parsed, dict)

    def test_flat_data_omits_tvg_recommendation_gracefully(self, tmp_path):
        """Same known Klein SDF failure mode TestRunTvgFit covers below --
        Stage 3 can't determine usable_range_m on flat data, so Stage 4
        never runs. recommend-config must still write a file (with the
        other sections' evidence), not crash."""
        flat = _decay_survey()
        ds = flat.get_frequency("hz_1495000").dataset
        ds["backscatter_raw"] = ds["backscatter_raw"] * 0 + 1.0
        ds["backscatter_intensity"] = ds["backscatter_raw"].copy()
        path = tmp_path / "a.ss.h5"
        write_survey(flat, path)

        args = self._args(tmp_path, path)
        assert analyze.run_recommend_config(args) == 0
        text = (tmp_path / "out" / "recommended_config.yaml").read_text()
        assert "could not be determined" in text


def _write_survey_in_parts(survey: SideScanSurvey, out_dir, n_parts: int) -> list:
    """Split `survey` into `n_parts` contiguous-ping-range `.ss.h5` files
    under `out_dir` -- simulates a mission that was converted part-by-part
    (never concatenated into one file) rather than pre-splitting a
    single already-written file."""
    from sidescanpy.processing.slicing import slice_by_pings

    out_dir.mkdir(parents=True, exist_ok=True)
    freq_keys = list(survey.frequencies)
    n_pings = survey.get_frequency(freq_keys[0]).n_pings
    bounds = np.linspace(0, n_pings, n_parts + 1).astype(int)
    paths = []
    for i in range(n_parts):
        start, stop = int(bounds[i]), int(bounds[i + 1])
        part_survey = SideScanSurvey(
            source_files=survey.source_files, source_format=survey.source_format,
            survey_name=f"{survey.survey_name}_part{i}",
        )
        for fk in freq_keys:
            sliced = slice_by_pings(survey.get_frequency(fk), start=start, stop=stop)
            part_survey.add_frequency(fk, sliced.dataset)
        path = out_dir / f"part{i:02d}.ss.h5"
        write_survey(part_survey, path)
        paths.append(path)
    return paths


class TestDirectoryInput:
    """``--input``/``--ss-h5`` accepting a directory of ``.ss.h5`` files
    (lazily dask-concatenated in filename order, same convention
    ``sidescan-export``'s own ``--ss-h5-dir`` already uses), added
    2026-09-27 -- see ``cli._common.load_channel_dataset``'s own
    docstring for the real motivation: a mission too large to
    concatenate into one file at conversion time on a memory-constrained
    container can now still be analyzed as one logical survey, without
    losing cross-file gap/instability detection (the actual concern that
    made a naive per-file-only analysis a real regression, not just an
    inconvenience)."""

    def test_nadir_directory_matches_single_file_result(self, tmp_path):
        survey = _decay_survey(n_pings=10, n_samples=200)
        single_path = tmp_path / "single.ss.h5"
        write_survey(survey, single_path)
        part_dir = tmp_path / "parts"
        _write_survey_in_parts(survey, part_dir, n_parts=2)

        single_args = TestRunNadir()._args(tmp_path, single_path, output_dir=tmp_path / "out_single")
        dir_args = TestRunNadir()._args(tmp_path, part_dir, output_dir=tmp_path / "out_dir")
        assert analyze.run_nadir(single_args) == 0
        assert analyze.run_nadir(dir_args) == 0

        single_metrics = json.loads((tmp_path / "out_single" / "metrics.json").read_text())
        dir_metrics = json.loads((tmp_path / "out_dir" / "metrics.json").read_text())
        assert single_metrics["classification"] == dir_metrics["classification"]
        assert single_metrics["metrics"]["nadir_mean"] == pytest.approx(dir_metrics["metrics"]["nadir_mean"])

    def test_tvg_fit_directory_input_succeeds(self, tmp_path):
        survey = _decay_survey(n_pings=10, n_samples=800)
        part_dir = tmp_path / "parts"
        _write_survey_in_parts(survey, part_dir, n_parts=2)

        args = TestRunTvgFit()._args(tmp_path, part_dir)
        assert analyze.run_tvg_fit(args) == 0
        assert list((tmp_path / "out").glob("*.png"))

    def test_recommend_config_directory_input_succeeds(self, tmp_path):
        survey = _decay_survey(n_pings=10, n_samples=800)
        part_dir = tmp_path / "parts"
        _write_survey_in_parts(survey, part_dir, n_parts=2)

        args = TestRunRecommendConfig()._args(tmp_path, part_dir)
        assert analyze.run_recommend_config(args) == 0
        text = (tmp_path / "out" / "recommended_config.yaml").read_text()
        assert "resolution_m:" in text

    def test_segment_directory_input_detects_gap_across_file_boundary(self, tmp_path):
        """The actual point of this whole capability: a real position
        jump right at the seam between two originally-separate files
        must still be detected and placed as a real segment boundary
        when analyzing the directory as one concatenated timeline --
        proving this isn't just "runs without crashing" but genuinely
        preserves cross-file detection. Two independent per-file calls
        (the alternative without directory support) would each just
        report their own [0:100] as "section1", with no way to know they
        were ever adjacent or where the real seam was; one continuous
        (but gap-blind) concatenation would instead wrongly merge them
        into one falsely-continuous [0:200) clean stretch. Either failure
        mode is avoided here: exactly two segments, split precisely at
        the real file boundary."""
        n_pings = 100
        near_survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        ds_near = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=50, freq_hz_start=1_495_000)
        ds_near["latitude"] = ("ping", np.linspace(41.0000, 41.0100, n_pings))
        ds_near["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
        near_survey.add_frequency("hz_1495000", ds_near)

        far_survey = SideScanSurvey(source_files=["b.jsf"], source_format="JSF", survey_name="b")
        ds_far = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=50, freq_hz_start=1_495_000)
        # A real position jump relative to near_survey's own ping spacing
        # (~0.0001 deg/ping above) -- several degrees away, unmistakably
        # a gap, not ordinary ping-to-ping noise.
        ds_far["latitude"] = ("ping", np.linspace(45.0000, 45.0100, n_pings))
        ds_far["longitude"] = ("ping", np.linspace(-71.0000, -71.0000, n_pings))
        far_survey.add_frequency("hz_1495000", ds_far)

        part_dir = tmp_path / "parts"
        part_dir.mkdir()
        write_survey(near_survey, part_dir / "part00.ss.h5")
        write_survey(far_survey, part_dir / "part01.ss.h5")

        args = TestRunSegment()._args(tmp_path, part_dir, pings=(0, None))
        assert analyze.run_segment(args) == 0

        report = json.loads((tmp_path / "out" / "segments.json").read_text())
        segments = report["segments"]
        assert len(segments) == 2
        assert all(s["status"] == "clean" for s in segments)
        boundaries = sorted(s["start"] for s in segments) + sorted(s["stop"] for s in segments)
        assert n_pings in boundaries  # the real file seam, not silently merged or lost
        total_pings = sum(s["stop"] - s["start"] for s in segments)
        assert total_pings == 2 * n_pings

    def test_missing_ss_h5_dir_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        args = TestRunNadir()._args(tmp_path, empty_dir)
        with pytest.raises(FileNotFoundError):
            analyze.run_nadir(args)


class TestPingChunksAlignment:
    """Real user report, 2026-09-28: a UserWarning ("specified chunks
    separate the stored chunks...") fired on every sidescan-analyze call
    against real data. Root cause: every lazy read here hardcoded
    chunks={"ping": 500}, not a multiple of HDF5_CHUNK_PING (the on-disk
    chunk size write_survey() actually stores, 256) -- xr.open_dataset
    had to split real HDF5 chunks to match, both the warning and real
    degraded read performance, not just noise."""

    def test_ping_chunks_constant_is_aligned(self):
        assert analyze._PING_CHUNKS["ping"] % HDF5_CHUNK_PING == 0

    def test_reading_a_real_multi_chunk_survey_raises_no_alignment_warning(self, tmp_path):
        # More pings than one on-disk chunk (256), so write_survey()
        # genuinely writes >1 HDF5 chunk -- the real scenario the
        # warning was about, not a degenerate single-chunk file too
        # small to ever trigger it either way.
        n_pings = HDF5_CHUNK_PING * 3
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=200, freq_hz_start=1_495_000)
        survey.add_frequency("hz_1495000", ds)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_survey(path, chunks=analyze._PING_CHUNKS)

        alignment_warnings = [w for w in caught if "specified chunks separate the stored chunks" in str(w.message)]
        assert not alignment_warnings


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
