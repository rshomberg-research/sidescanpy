"""
Tests for calling the correction steps directly on a survey
(``apply_tvg(survey)``, ``compute_geometric_ranges(survey)``, ...), the
explicit alternative to ``correct_survey(survey, steps=[...])``.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core import SideScanSurvey
from sidescanpy.core.history import applied_steps, read_history
from sidescanpy.processing import (
    TVGParams,
    altitude_from_bottom_track,
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    correct_survey,
    crop_far_range,
    interpolate_aux_onto_pings,
)
from sidescanpy.processing.range_correction import apply_empirical_range_correction
from tests.unit.processing.test_correct import _ds, _rc_table, _survey


def _explicit(survey):
    survey = interpolate_aux_onto_pings(survey)
    survey = apply_tvg(survey)
    survey = compute_geometric_ranges(survey)
    return compute_georeferencing(survey)


class TestSurveyInput:
    def test_returns_new_survey_when_given_survey(self):
        survey = _survey()
        out = apply_tvg(survey)
        assert isinstance(out, SideScanSurvey)
        assert out is not survey
        assert out.frequencies == survey.frequencies

    def test_input_survey_unchanged_when_corrected(self):
        survey = _survey()
        before = {k: survey.get_frequency(k).dataset["backscatter_intensity"].values.copy()
                  for k in survey.frequencies}
        apply_tvg(survey)
        for k in survey.frequencies:
            ds = survey.get_frequency(k).dataset
            np.testing.assert_array_equal(ds["backscatter_intensity"].values, before[k])
            assert "tvg" not in applied_steps(ds)

    def test_every_frequency_recorded_when_given_survey(self):
        out = compute_geometric_ranges(apply_tvg(_survey()))
        for k in out.frequencies:
            assert applied_steps(out.get_frequency(k).dataset)[-2:] == ["tvg", "geometric"]

    def test_aux_records_kept_when_given_survey(self):
        survey = _survey()
        out = apply_tvg(survey)
        assert out.navigation is survey.navigation
        assert out.attitude is survey.attitude

    def test_dataset_input_returns_dataset(self):
        assert isinstance(apply_tvg(_ds()), xr.Dataset)

    @pytest.mark.parametrize("fn", [
        crop_far_range, altitude_from_bottom_track, apply_tvg,
        compute_geometric_ranges, compute_georeferencing,
    ])
    def test_step_accepts_survey(self, fn):
        survey = interpolate_aux_onto_pings(_survey())
        if fn is compute_georeferencing:
            survey = compute_geometric_ranges(survey)
        out = fn(survey)
        assert isinstance(out, SideScanSurvey)
        assert all(read_history(out.get_frequency(k).dataset) for k in out.frequencies)

    def test_range_correction_accepts_survey_and_table(self):
        survey = compute_geometric_ranges(apply_tvg(interpolate_aux_onto_pings(_survey())))
        out = apply_empirical_range_correction(survey, table=_rc_table())
        for k in out.frequencies:
            assert applied_steps(out.get_frequency(k).dataset)[-1] == "range_correction"


class TestMatchesCorrectSurvey:
    def test_explicit_chain_matches_correct_survey_when_same_steps(self):
        a = _explicit(_survey())
        b = correct_survey(_survey(), steps=["interpolate_aux", "tvg", "geometric", "georef"])
        for k in a.frequencies:
            da, db = a.get_frequency(k).dataset, b.get_frequency(k).dataset
            assert applied_steps(da) == applied_steps(db)
            for var in ("backscatter_intensity", "across_track_m", "latitude_per_sample"):
                np.testing.assert_allclose(da[var].values, db[var].values, equal_nan=True)


class TestApplyTvgKeywords:
    def test_keyword_overrides_default_when_given(self):
        out = apply_tvg(_ds(), spreading_loss_db=40.0)
        assert read_history(out)[-1]["params"]["spreading_loss_db"] == 40.0

    def test_keyword_overrides_params_field_when_both_given(self):
        params = TVGParams(spreading_loss_db=20.0, min_range_m=2.0)
        out = apply_tvg(_ds(), params, spreading_loss_db=40.0)
        record = read_history(out)[-1]["params"]
        assert record["spreading_loss_db"] == 40.0
        assert record["min_range_m"] == 2.0
        assert params.spreading_loss_db == 20.0  # caller's object untouched

    def test_keywords_match_tvgparams_when_equivalent(self):
        a = apply_tvg(_ds(), TVGParams(spreading_loss_db=40.0, absorption_db_per_m=0.05))
        b = apply_tvg(_ds(), spreading_loss_db=40.0, absorption_db_per_m=0.05)
        np.testing.assert_array_equal(a["backscatter_intensity"].values,
                                      b["backscatter_intensity"].values)

    def test_raises_when_tvg_already_applied_to_survey(self):
        with pytest.raises(ValueError, match="already been applied"):
            apply_tvg(apply_tvg(_survey()))
