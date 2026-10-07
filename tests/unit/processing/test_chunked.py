"""
Unit tests for sidescanpy.processing.chunked.correct_file_chunked —
memory-bounded, ping-chunked in-place correction.

The core claim under test: chunking must be numerically invisible --
same steps applied via correct_file_chunked() or the non-chunked
apply_correction_steps() must produce identical results, chunk
boundaries included.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing.chunked import correct_file_chunked
from sidescanpy.processing.pipeline import apply_correction_steps, step_already_done


def _survey_with_dataset(ds, freq_key="hz_1495000"):
    survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
    survey.add_frequency(freq_key, ds)
    return survey


@pytest.mark.parametrize("sample_dataset", [{"n_pings": 23, "n_samples": 40}], indirect=True)
class TestCorrectFileChunkedMatchesEager:
    """23 pings, chunked at 7 -- deliberately doesn't divide evenly, so
    the last chunk is a real partial chunk, not a coincidence of round
    numbers."""

    def test_tvg_geometric_georef_identical_to_eager(self, tmp_path, sample_dataset):
        path = tmp_path / "a.ss.h5"
        write_survey(_survey_with_dataset(sample_dataset), path)

        eager = apply_correction_steps(sample_dataset.copy(deep=True), {"tvg", "geometric", "georef"})

        correct_file_chunked(path, {"tvg", "geometric", "georef"}, ping_chunk_size=7)
        chunked_survey = read_survey(path)
        chunked = chunked_survey.get_frequency("hz_1495000").dataset

        for var in ("backscatter_intensity", "slant_range_m", "ground_range_m",
                    "across_track_m", "along_track_m", "latitude_per_sample", "longitude_per_sample"):
            np.testing.assert_allclose(chunked[var].values, eager[var].values, rtol=1e-5, atol=1e-6)
        assert chunked.attrs["processing_level"] == eager.attrs["processing_level"]

    def test_geometric_only_identical_to_eager(self, tmp_path, sample_dataset):
        path = tmp_path / "a.ss.h5"
        write_survey(_survey_with_dataset(sample_dataset), path)

        eager = apply_correction_steps(sample_dataset.copy(deep=True), {"geometric"})

        correct_file_chunked(path, {"geometric"}, ping_chunk_size=7)
        chunked = read_survey(path).get_frequency("hz_1495000").dataset

        np.testing.assert_allclose(chunked["ground_range_m"].values, eager["ground_range_m"].values, equal_nan=True)


@pytest.mark.parametrize("sample_dataset", [{"n_pings": 12, "n_samples": 20}], indirect=True)
class TestCorrectFileChunkedAlreadyDone:
    def test_returns_false_and_does_not_rewrite_when_all_already_done(self, tmp_path, sample_dataset):
        path = tmp_path / "a.ss.h5"
        already_done = apply_correction_steps(sample_dataset, {"tvg"})
        write_survey(_survey_with_dataset(already_done), path)
        mtime_before = path.stat().st_mtime_ns

        applied = correct_file_chunked(path, {"tvg"}, ping_chunk_size=5)

        assert applied is False
        assert path.stat().st_mtime_ns == mtime_before

    def test_partial_already_done_only_applies_the_rest(self, tmp_path, sample_dataset):
        """tvg already done, geometric/georef not -- chunked path should
        apply only the missing steps, not re-run tvg."""
        tvg_done = apply_correction_steps(sample_dataset, {"tvg"})
        path = tmp_path / "a.ss.h5"
        write_survey(_survey_with_dataset(tvg_done), path)

        applied = correct_file_chunked(path, {"tvg", "geometric", "georef"}, ping_chunk_size=5)

        assert applied is True
        result = read_survey(path).get_frequency("hz_1495000").dataset
        assert step_already_done(result, "tvg")
        assert step_already_done(result, "geometric")
        assert step_already_done(result, "georef")
        # tvg's own backscatter_intensity values are untouched by the
        # chunked run -- re-applying tvg on top of itself would change
        # them (TVG is not idempotent applied twice), so this is a real
        # check that tvg was actually skipped, not just that its history
        # entry looks right.
        np.testing.assert_array_equal(
            result["backscatter_intensity"].values, tvg_done["backscatter_intensity"].values,
        )


class TestCorrectFileChunkedRejectsUnchunkableSteps:
    def test_crop_farrange_rejected(self, tmp_path, sample_dataset):
        path = tmp_path / "a.ss.h5"
        write_survey(_survey_with_dataset(sample_dataset), path)
        with pytest.raises(ValueError, match="crop_farrange"):
            correct_file_chunked(path, {"crop_farrange"}, ping_chunk_size=5)

    def test_unknown_step_rejected(self, tmp_path, sample_dataset):
        path = tmp_path / "a.ss.h5"
        write_survey(_survey_with_dataset(sample_dataset), path)
        with pytest.raises(ValueError, match="not_a_real_step"):
            correct_file_chunked(path, {"not_a_real_step"}, ping_chunk_size=5)


@pytest.mark.parametrize("sample_dataset", [{"n_pings": 15, "n_samples": 20}], indirect=True)
class TestCorrectFileChunkedMultiFrequency:
    def test_processes_every_frequency_channel(self, tmp_path, sample_dataset):
        survey = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        ds2 = sample_dataset.copy(deep=True)
        ds2.attrs["freq_hz_start"] = 547_500
        ds2.attrs["freq_hz_end"] = 547_500
        survey.add_frequency("hz_1495000", sample_dataset)
        survey.add_frequency("hz_547500", ds2)
        path = tmp_path / "a.ss.h5"
        write_survey(survey, path)

        applied = correct_file_chunked(path, {"tvg"}, ping_chunk_size=6)

        assert applied is True
        result = read_survey(path)
        assert step_already_done(result.get_frequency("hz_1495000").dataset, "tvg")
        assert step_already_done(result.get_frequency("hz_547500").dataset, "tvg")
