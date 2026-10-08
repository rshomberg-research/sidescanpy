"""
Unit tests for sidescanpy.processing.track (build_survey_track).

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np

from sidescanpy.processing import SurveyTrack, build_survey_track


class TestBuildSurveyTrack:
    def test_none_when_no_file_has_frequency(self, sample_multi_frequency_survey):
        surveys = {Path("a.ss.h5"): sample_multi_frequency_survey}
        result = build_survey_track(surveys, "hz_999999", scanline_stride=5)
        assert result is None

    def test_returns_track_and_scanline_arrays(self, sample_multi_frequency_survey):
        surveys = {Path("a.ss.h5"): sample_multi_frequency_survey}
        n_pings = sample_multi_frequency_survey.get_frequency("hz_1495000").dataset.sizes["ping"]
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)
        assert result is not None
        assert len(result.track_lat) == n_pings
        assert len(result.track_lon) == n_pings
        expected_n_rungs = len(np.arange(0, n_pings, 3))
        assert len(result.scan_port_lat) == expected_n_rungs
        assert len(result.scan_stbd_lat) == expected_n_rungs

    def test_multi_file_concatenates_in_time_order(self, sample_multi_frequency_survey):
        from sidescanpy.core import SideScanSurvey

        ds_a = sample_multi_frequency_survey.get_frequency("hz_1495000").dataset.copy()
        ds_b = ds_a.copy()
        # Shift file b's timestamps later so file a must sort first
        ds_b = ds_b.assign_coords(time=ds_b["time"] + np.timedelta64(1, "h"))
        ds_b["latitude"] = ds_a["latitude"] + 10.0  # distinguishable values

        survey_a = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey_a.add_frequency("hz_1495000", ds_a)
        survey_b = SideScanSurvey(source_files=["b.jsf"], source_format="JSF", survey_name="b")
        survey_b.add_frequency("hz_1495000", ds_b)

        surveys = {Path("b.ss.h5"): survey_b, Path("a.ss.h5"): survey_a}
        n_pings = ds_a.sizes["ping"]
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)
        # file a (earlier timestamps) must appear first in the concatenated track
        np.testing.assert_array_equal(result.track_lat[:n_pings], ds_a["latitude"].values)
        np.testing.assert_array_equal(result.track_lat[n_pings:], ds_b["latitude"].values)

    def test_mismatched_sample_window_lengths_do_not_nan_out_rungs(self):
        """Regression test: a real survey's two files can have
        differently-sized sample axes (e.g. one file's channel was
        trimmed further by the far-range noise crop than the other's).
        The last-sample "sample" coordinate then differs by value
        across files (e.g. index 79 vs. 99); concatenating without
        accounting for that used to align on the mismatched label and
        NaN out whichever file didn't have that particular value."""
        from sidescanpy.core import SideScanSurvey
        from tests.conftest import make_minimal_sidescan_dataset

        # altitude_m must be well below the (very short, default-fixture)
        # slant range reached by the last sample, or every sample is
        # legitimately water-column-masked regardless of this bug.
        # A real .ss.h5-loaded dataset carries an explicit indexed
        # "sample" coordinate (assigned by the HDF5 backend on open,
        # not by this in-memory fixture) — assign one here too, since
        # that indexed label is exactly what the old code aligned on.
        ds_a = make_minimal_sidescan_dataset(n_pings=10, n_samples=100)
        ds_a["altitude_m"] = (["ping"], np.full(10, 0.05, dtype=np.float32))
        ds_a = ds_a.assign_coords(sample=np.arange(ds_a.sizes["sample"]))
        ds_b = make_minimal_sidescan_dataset(n_pings=10, n_samples=80)
        ds_b["altitude_m"] = (["ping"], np.full(10, 0.05, dtype=np.float32))
        ds_b = ds_b.assign_coords(sample=np.arange(ds_b.sizes["sample"]))

        survey_a = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey_a.add_frequency("hz_1495000", ds_a)
        survey_b = SideScanSurvey(source_files=["b.jsf"], source_format="JSF", survey_name="b")
        survey_b.add_frequency("hz_1495000", ds_b)

        surveys = {Path("a.ss.h5"): survey_a, Path("b.ss.h5"): survey_b}
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)

        assert not np.isnan(result.scan_port_lat).any()
        assert not np.isnan(result.scan_stbd_lat).any()

    def test_returns_survey_track(self, sample_multi_frequency_survey):
        surveys = {Path("a.ss.h5"): sample_multi_frequency_survey}
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)
        assert isinstance(result, SurveyTrack)
        assert result.n_pings == len(result.track_lat)
        assert result.n_scanlines == len(result.scan_port_lat) == len(result.scan_stbd_lon)
