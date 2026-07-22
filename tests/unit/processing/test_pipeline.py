"""
Unit tests for sidescanpy.processing.pipeline — the correction-step
ladder (run_pipeline) and explicit step-set primitive
(run_correction_steps).

2026 rshomberg@uri.edu
"""

import pytest

from sidescanpy.processing.pipeline import (
    VALID_PROCESS_LEVELS,
    VALID_STEPS,
    apply_correction_steps,
    run_correction_steps,
    run_pipeline,
)


def _levels(survey):
    return {
        fk: ch.dataset.attrs["processing_level"]
        for fk, ch in survey.freq_channels.items()
    }


def _has_var(survey, var):
    return all(
        var in ch.dataset.data_vars
        for ch in survey.freq_channels.values()
    )


class TestRunCorrectionSteps:
    def test_empty_steps_is_noop(self, sample_multi_frequency_survey):
        result = run_correction_steps(sample_multi_frequency_survey, frozenset())
        assert set(_levels(result).values()) == {"raw"}
        assert not _has_var(result, "ground_range_m")

    def test_tvg_only(self, sample_multi_frequency_survey):
        result = run_correction_steps(sample_multi_frequency_survey, {"tvg"})
        assert set(_levels(result).values()) == {"radiometric"}
        assert not _has_var(result, "ground_range_m")

    def test_geometric_without_tvg_keeps_raw_backscatter(self, sample_multi_frequency_survey):
        """Verified combination per workflow-decisions.md: raw backscatter
        + geometric ranges is a supported pairing, not just a ladder
        artifact."""
        result = run_correction_steps(sample_multi_frequency_survey, {"geometric"})
        assert set(_levels(result).values()) == {"raw"}
        assert _has_var(result, "ground_range_m")

    def test_tvg_and_geometric(self, sample_multi_frequency_survey):
        result = run_correction_steps(sample_multi_frequency_survey, {"tvg", "geometric"})
        assert set(_levels(result).values()) == {"radiometric"}
        assert _has_var(result, "ground_range_m")

    def test_full_steps(self, sample_multi_frequency_survey):
        result = run_correction_steps(
            sample_multi_frequency_survey, {"tvg", "geometric", "georef"}
        )
        assert set(_levels(result).values()) == {"radiometric"}
        assert _has_var(result, "ground_range_m")
        assert _has_var(result, "latitude_per_sample")

    def test_range_correction_without_georef(self, sample_multi_frequency_survey):
        """range_correction is opt-in and independent of georef — a
        supported combination the ladder can't express (range_correction
        isn't part of any ladder level). With the bundled default
        calibration table empty (no fitted curves shipped yet), this
        exercises the "no calibrated frequency match" skip path in
        apply_empirical_range_correction — must not raise, and
        processing_level/ground_range_m must be unaffected by the skip."""
        result = run_correction_steps(
            sample_multi_frequency_survey, {"tvg", "geometric", "range_correction"}
        )
        assert set(_levels(result).values()) == {"radiometric"}
        assert _has_var(result, "ground_range_m")
        assert not _has_var(result, "latitude_per_sample")

    def test_range_correction_requires_geometric(self, sample_multi_frequency_survey):
        """apply_empirical_range_correction requires ground_range_m —
        requesting it without "geometric" must raise, not silently skip."""
        with pytest.raises(ValueError, match="ground_range_m"):
            run_correction_steps(
                sample_multi_frequency_survey, {"tvg", "range_correction"}
            )

    def test_georef_without_geometric_raises(self, sample_multi_frequency_survey):
        """compute_georeferencing requires across_track_m/along_track_m
        from compute_geometric_ranges — an ordering constraint documented
        in workflow-decisions.md, not something run_correction_steps
        papers over."""
        with pytest.raises(Exception):
            run_correction_steps(sample_multi_frequency_survey, {"georef"})

    def test_unknown_step_raises(self, sample_multi_frequency_survey):
        with pytest.raises(ValueError, match="Unknown correction step"):
            run_correction_steps(sample_multi_frequency_survey, {"bogus"})

    def test_all_valid_steps_accepted(self, sample_multi_frequency_survey):
        # Sanity: VALID_STEPS as a whole is a legal call.
        result = run_correction_steps(sample_multi_frequency_survey, VALID_STEPS)
        assert _has_var(result, "latitude_per_sample")


class TestApplyCorrectionSteps:
    """apply_correction_steps operates on a bare channel dataset, not a
    SideScanSurvey — no aux interpolation involved. Used by scripts that
    interpolate aux once up front and re-apply corrections per segment."""

    def test_empty_steps_is_noop(self, sample_dataset):
        result = apply_correction_steps(sample_dataset, frozenset())
        assert result.attrs["processing_level"] == "raw"
        assert "ground_range_m" not in result.data_vars

    def test_geometric_without_tvg(self, sample_dataset):
        result = apply_correction_steps(sample_dataset, {"geometric"})
        assert result.attrs["processing_level"] == "raw"
        assert "ground_range_m" in result.data_vars

    def test_full_steps(self, sample_dataset):
        result = apply_correction_steps(sample_dataset, {"tvg", "geometric", "georef"})
        assert result.attrs["processing_level"] == "radiometric"
        assert "latitude_per_sample" in result.data_vars

    def test_does_not_mutate_input(self, sample_dataset):
        original_level = sample_dataset.attrs["processing_level"]
        apply_correction_steps(sample_dataset, {"tvg"})
        assert sample_dataset.attrs["processing_level"] == original_level

    def test_unknown_step_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Unknown correction step"):
            apply_correction_steps(sample_dataset, {"bogus"})


class TestRunPipeline:
    def test_none_is_noop(self, sample_multi_frequency_survey):
        result = run_pipeline(sample_multi_frequency_survey, "none")
        assert set(_levels(result).values()) == {"raw"}

    def test_radiometric(self, sample_multi_frequency_survey):
        result = run_pipeline(sample_multi_frequency_survey, "radiometric")
        assert set(_levels(result).values()) == {"radiometric"}
        assert not _has_var(result, "ground_range_m")

    def test_geometric_implies_tvg(self, sample_multi_frequency_survey):
        """Ladder semantics: unlike run_correction_steps({'geometric'}),
        the ladder's 'geometric' level always includes TVG first."""
        result = run_pipeline(sample_multi_frequency_survey, "geometric")
        assert set(_levels(result).values()) == {"radiometric"}
        assert _has_var(result, "ground_range_m")

    def test_georef_runs_full_ladder(self, sample_multi_frequency_survey):
        result = run_pipeline(sample_multi_frequency_survey, "georef")
        assert set(_levels(result).values()) == {"radiometric"}
        assert _has_var(result, "latitude_per_sample")

    def test_invalid_target_raises(self, sample_multi_frequency_survey):
        with pytest.raises(ValueError, match="target must be one of"):
            run_pipeline(sample_multi_frequency_survey, "bogus")

    def test_valid_process_levels_contents(self):
        assert VALID_PROCESS_LEVELS == ("none", "radiometric", "geometric", "georef")
