"""
Unit tests for sidescanpy.processing.pipeline — the correction-step
ladder (run_pipeline) and explicit step-set primitive
(run_correction_steps).

2026 rshomberg@uri.edu
"""

import json

import numpy as np
import pytest

from sidescanpy.processing.pipeline import (
    VALID_PROCESS_LEVELS,
    VALID_STEPS,
    apply_correction_steps,
    run_correction_steps,
    run_pipeline,
    step_already_done,
)
from sidescanpy.processing.radiometric import TVGParams


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
        with pytest.raises(ValueError, match="Run compute_geometric_ranges first"):
            run_correction_steps(sample_multi_frequency_survey, {"georef"})

    def test_unknown_step_raises(self, sample_multi_frequency_survey):
        with pytest.raises(ValueError, match="Unknown correction step"):
            run_correction_steps(sample_multi_frequency_survey, {"bogus"})

    def test_all_valid_steps_accepted(self, sample_multi_frequency_survey):
        # Sanity: VALID_STEPS as a whole is a legal call.
        result = run_correction_steps(sample_multi_frequency_survey, VALID_STEPS)
        assert _has_var(result, "latitude_per_sample")

    def test_step_params_reaches_every_frequency_channel(self, sample_multi_frequency_survey):
        zeroed = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0)
        result = run_correction_steps(
            sample_multi_frequency_survey, {"tvg"},
            step_params={"tvg": {"params": zeroed}},
        )
        for ch in result.freq_channels.values():
            history = json.loads(ch.dataset.attrs["processing_history"])
            tvg_entry = next(h for h in history if h["step"] == "tvg_correction")
            assert tvg_entry["params"]["spreading_loss_db"] == 0.0
            assert tvg_entry["params"]["absorption_db_per_m"] == 0.0

    def test_step_params_none_is_backward_compatible(self, sample_multi_frequency_survey):
        result = run_correction_steps(sample_multi_frequency_survey, {"tvg"}, step_params=None)
        assert set(_levels(result).values()) == {"radiometric"}


class TestApplyCorrectionSteps:
    """apply_correction_steps operates on a bare channel dataset, not a
    SideScanSurvey — no aux interpolation involved. Used by scripts that
    interpolate aux once up front and re-apply corrections per segment."""

    def test_empty_steps_is_noop(self, sample_dataset):
        result = apply_correction_steps(sample_dataset, frozenset())
        assert result.attrs["processing_level"] == "raw"


class TestStepAlreadyDone:
    """Per-dataset predicate shared by cli/correct.py's per-survey checks
    and processing/chunked.py's per-chunk checks -- the one place this
    logic lives (see the function's own docstring)."""

    def test_tvg_false_on_raw(self, sample_dataset):
        assert step_already_done(sample_dataset, "tvg") is False

    def test_tvg_true_after_applying(self, sample_dataset):
        corrected = apply_correction_steps(sample_dataset, {"tvg"})
        assert step_already_done(corrected, "tvg") is True

    def test_geometric_false_before_applying(self, sample_dataset):
        assert step_already_done(sample_dataset, "geometric") is False

    def test_geometric_true_after_applying(self, sample_dataset):
        corrected = apply_correction_steps(sample_dataset, {"geometric"})
        assert step_already_done(corrected, "geometric") is True

    def test_georef_true_after_applying(self, sample_dataset):
        corrected = apply_correction_steps(sample_dataset, {"geometric", "georef"})
        assert step_already_done(corrected, "georef") is True

    def test_range_correction_true_after_applying(self, sample_dataset):
        corrected = apply_correction_steps(sample_dataset, {"tvg", "geometric", "range_correction"})
        assert step_already_done(corrected, "range_correction") is True

    def test_unknown_step_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Unknown correction step"):
            step_already_done(sample_dataset, "not_a_real_step")

    def test_tvg_only_skips_geometric(self, sample_dataset):
        result = apply_correction_steps(sample_dataset, {"tvg"})
        assert result.attrs["processing_level"] == "radiometric"
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

    def test_apply_unknown_step_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="Unknown correction step"):
            apply_correction_steps(sample_dataset, {"bogus"})

    def test_step_params_overrides_tvg_params(self, sample_dataset):
        zeroed = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0)
        result = apply_correction_steps(
            sample_dataset, {"tvg"}, step_params={"tvg": {"params": zeroed}}
        )
        history = json.loads(result.attrs["processing_history"])
        tvg_entry = next(h for h in history if h["step"] == "tvg_correction")
        assert tvg_entry["params"]["spreading_loss_db"] == 0.0
        assert tvg_entry["params"]["absorption_db_per_m"] == 0.0

        # Both terms zeroed -> no gain added, backscatter_intensity is
        # exactly the plain linear-to-dB conversion of the raw amplitude.
        expected_db = 20.0 * np.log10(np.clip(sample_dataset["backscatter_intensity"].values, 1e-6, None))
        assert np.allclose(result["backscatter_intensity"].values, expected_db)

    def test_step_params_none_is_backward_compatible(self, sample_dataset):
        with_none = apply_correction_steps(sample_dataset, {"tvg"}, step_params=None)
        without_kwarg = apply_correction_steps(sample_dataset, {"tvg"})
        assert with_none["backscatter_intensity"].equals(without_kwarg["backscatter_intensity"])

    def test_step_params_entry_for_unrequested_step_is_harmless(self, sample_dataset):
        # "georef" isn't in the requested step set -- its entry in
        # step_params must simply be unused, not an error.
        result = apply_correction_steps(
            sample_dataset, {"tvg"}, step_params={"georef": {"bogus_kwarg": 1}}
        )
        assert result.attrs["processing_level"] == "radiometric"


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
