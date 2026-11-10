"""
Unit tests for sidescanpy.analysis.tvg_fit (fit_tvg_models, check_tvg) and
the median-level profile it works on.
"""

import json

import numpy as np
import pytest

from sidescanpy.analysis import (
    TVGFitResult,
    check_tvg,
    fit_tvg_models,
    tvg_model_name,
)
from sidescanpy.processing import apply_tvg, compute_geometric_ranges, median_level_profile
from sidescanpy.processing.radiometric import TVGParams, absorption_for_frequency
from tests.unit.synthetic import make_range_decay_dataset

FREQ = 500_000.0
ALPHA = float(absorption_for_frequency(FREQ))


def _ds(**kwargs):
    kwargs.setdefault("freq_hz", FREQ)
    return compute_geometric_ranges(make_range_decay_dataset(**kwargs))


class TestModelName:
    @pytest.mark.parametrize(
        ("n", "absorption", "name"),
        [(0, False, "flat"), (0, True, "absorption only"), (20, False, "20 log R"), (30.0, True, "30 log R + abs")],
    )
    def test_model_name_when_formatting(self, n, absorption, name):
        assert tvg_model_name(n, absorption) == name


class TestMedianLevelProfile:
    def test_profile_regular_axis_when_binned(self):
        ds = _ds()
        prof = median_level_profile(ds, bin_width_m=1.0)
        assert np.allclose(np.diff(prof["range_m"]), 1.0)
        assert prof["range_min_m"].iloc[0] == 0.0
        assert set(prof.columns) >= {"range_m", "median", "level_db", "count"}

    def test_profile_excludes_water_column_when_ground_range_present(self):
        prof = median_level_profile(_ds(altitude_m=10.0))
        assert np.isnan(prof.loc[prof["range_max_m"] <= 10.0, "level_db"]).all()

    def test_profile_min_range_cuts_near_bins(self):
        prof = median_level_profile(_ds(), min_range_m=20.0)
        assert np.isnan(prof.loc[prof["range_max_m"] <= 20.0, "level_db"]).all()
        assert np.isfinite(prof.loc[prof["range_m"] == 30.5, "level_db"]).all()

    def test_profile_dithers_integer_samples_when_auto(self):
        assert median_level_profile(_ds(integer=True)).attrs["dithered"] is True
        assert median_level_profile(_ds(integer=False)).attrs["dithered"] is False

    def test_profile_channel_selects_one_side(self):
        ds = _ds(side_gain_db=(20.0, 0.0))
        port = median_level_profile(ds, channel=0)
        stbd = median_level_profile(ds, channel=1)
        diff = (port["level_db"] - stbd["level_db"]).to_numpy()
        sel = (port["range_m"] > 15) & (port["range_m"] < 60)
        assert np.nanmedian(diff[sel]) == pytest.approx(20.0, abs=0.5)

    def test_profile_reproducible_when_dithered(self):
        ds = _ds(integer=True)
        a = median_level_profile(ds)["level_db"].to_numpy()
        b = median_level_profile(ds)["level_db"].to_numpy()
        np.testing.assert_array_equal(a, b)


class TestFitTVGModels:
    @pytest.mark.parametrize(
        ("n", "absorption", "name", "status"),
        [
            (30.0, True, "30 log R + abs", "none"),
            (20.0, False, "20 log R", "absorption_only"),
            (40.0, True, "40 log R + abs", "none"),
            (0.0, True, "absorption only", "spreading_only"),
        ],
    )
    def test_fit_recovers_known_model(self, n, absorption, name, status):
        ds = _ds(spreading_loss_db=n, absorption_db_per_m=ALPHA if absorption else 0.0)
        fit = fit_tvg_models(ds)
        assert fit.best == name
        assert fit.tvg_status == status
        assert fit.best_model.rms_db < 0.15
        assert fit.tvg_params.spreading_loss_db == n
        assert fit.tvg_params.absorption_db_per_m == pytest.approx(ALPHA if absorption else 0.0)
        if absorption:
            assert fit.free_n == pytest.approx(n, abs=1.0)
        else:
            assert fit.free_n_no_absorption == pytest.approx(n, abs=1.0)

    def test_fit_window_runs_from_peak_margin_to_knee(self):
        fit = fit_tvg_models(_ds(signal_end_m=60.0))
        assert fit.window_source == "knee"
        assert fit.knee_found
        assert fit.peak_m == pytest.approx(10.5)
        assert fit.window_m == (pytest.approx(15.5), pytest.approx(60.5))
        assert fit.contrast_db > 60.0 and not fit.low_contrast

    def test_fit_window_ends_near_record_end_without_noise_floor(self):
        fit = fit_tvg_models(_ds(absorption_db_per_m=ALPHA))
        assert fit.knee_found
        assert 85.0 <= fit.window_m[1] <= 98.0

    def test_fit_reports_every_model_and_runner_up(self):
        fit = fit_tvg_models(_ds())
        assert len(fit.models) == 8
        assert fit.runner_up in fit.models and fit.runner_up != fit.best
        assert fit.models[fit.runner_up].rms_db >= fit.best_model.rms_db

    def test_fit_flat_data_says_flat_and_flags_low_contrast(self):
        ds = _ds(spreading_loss_db=0.0, absorption_db_per_m=0.0)
        fit = fit_tvg_models(ds)
        assert fit.best == "flat"
        assert fit.tvg_status == "full"
        assert fit.low_contrast
        assert not fit.knee_found
        assert fit.window_source == "end_of_record"
        assert any("end of the record" in note for note in fit.notes)

    def test_fit_rising_data_flags_far_range_rise(self):
        ds = _ds(spreading_loss_db=-10.0, absorption_db_per_m=0.0)
        fit = fit_tvg_models(ds)
        assert fit.rises_at_far_range
        assert fit.best == "flat"

    def test_fit_integer_samples_dithered(self):
        ds = _ds(integer=True, level_at_10m_db=50.0, absorption_db_per_m=ALPHA)
        fit = fit_tvg_models(ds)
        assert fit.dithered
        assert fit.best == "30 log R + abs"
        assert fit.free_n == pytest.approx(30.0, abs=1.0)

    def test_fit_short_window_falls_back_to_end_of_record(self):
        fit = fit_tvg_models(_ds(signal_end_m=20.0))
        assert not fit.knee_found
        assert fit.window_source == "end_of_record"
        assert fit.window_m[1] > 90.0

    def test_fit_empty_given_window_returns_no_fit(self):
        fit = fit_tvg_models(_ds(), window_m=(50.0, 50.2))
        assert fit.window_source == "given"
        assert fit.best is None
        assert fit.tvg_params is None and fit.tvg_status is None
        assert any("no fit" in note for note in fit.notes)

    def test_fit_given_window_used(self):
        fit = fit_tvg_models(_ds(), window_m=(20.0, 60.0))
        assert fit.window_m == (20.0, 60.0)
        assert fit.n_bins == 40

    def test_fit_unequal_side_levels_flagged(self, caplog):
        ds = _ds(side_gain_db=(20.0, 0.0), absorption_db_per_m=ALPHA)
        with caplog.at_level("WARNING", logger="sidescanpy.analysis.tvg_fit"):
            fit = fit_tvg_models(ds)
        assert fit.sides_disagree
        assert [s.best for s in fit.side_best] == ["30 log R + abs", "30 log R + abs"]
        assert fit.side_best[0].offset_db - fit.side_best[1].offset_db == pytest.approx(20.0, abs=1.0)
        assert "Sides disagree" in caplog.text

    def test_fit_different_side_shapes_flagged(self):
        ds = _ds(side_spreading_db=(20.0, 40.0), absorption_db_per_m=0.0)
        fit = fit_tvg_models(ds)
        assert [s.best for s in fit.side_best] == ["20 log R", "40 log R"]
        assert fit.sides_disagree

    def test_fit_equal_sides_not_flagged(self):
        fit = fit_tvg_models(_ds(absorption_db_per_m=ALPHA))
        assert not fit.sides_disagree

    def test_fit_one_channel(self):
        ds = _ds(side_spreading_db=(20.0, 40.0), absorption_db_per_m=0.0)
        assert fit_tvg_models(ds, channel="port").best == "20 log R"
        assert fit_tvg_models(ds, channel=1).best == "40 log R"

    def test_fit_peak_far_past_altitude_warns(self, caplog):
        ds = _ds(altitude_m=10.0)
        ds = ds.assign(altitude_m=ds["altitude_m"] * 0 + 2.0)  # recorded altitude too small
        ds = ds.assign(backscatter_intensity=ds["backscatter_intensity"].where(ds["slant_range_m"] > 30.0, 0.1))
        with caplog.at_level("WARNING", logger="sidescanpy.analysis.tvg_fit"):
            fit = fit_tvg_models(ds)
        assert any("Check the altitude" in note for note in fit.notes)
        assert "altitude" in caplog.text

    def test_fit_tvg_applied_raises(self):
        ds = apply_tvg(_ds())
        with pytest.raises(ValueError, match="invert"):
            fit_tvg_models(ds)

    def test_fit_missing_range_raises(self):
        ds = make_range_decay_dataset()
        with pytest.raises(ValueError, match="slant_range_m"):
            fit_tvg_models(ds)

    def test_fit_to_dict_is_json_with_params_and_status(self):
        fit = fit_tvg_models(_ds(absorption_db_per_m=ALPHA))
        d = fit.to_dict()
        json.dumps(d)
        assert d["tvg_status"] == "none"
        assert d["tvg_params"] == {"spreading_loss_db": 30.0, "absorption_db_per_m": pytest.approx(ALPHA)}
        assert d["models"]["30 log R + abs"]["rms_db"] == pytest.approx(fit.best_model.rms_db)
        assert len(d["level_db"]) == len(d["range_m"])

    def test_empty_result_to_dict(self):
        d = TVGFitResult().to_dict()
        assert d["tvg_params"] is None and d["tvg_status"] is None


class TestCheckTVG:
    def test_check_spread_matches_fit_rms(self):
        ds = _ds(absorption_db_per_m=ALPHA)
        fit = fit_tvg_models(ds)
        check = check_tvg(ds, fit.tvg_params, fit.window_m)
        assert check.spread_db == pytest.approx(fit.best_model.rms_db, abs=0.05)
        assert abs(check.trend_db_per_10m) < 0.05
        assert check.params == {"spreading_loss_db": 30.0, "absorption_db_per_m": pytest.approx(ALPHA)}

    def test_check_spread_matches_fit_rms_when_integer(self):
        ds = _ds(integer=True, level_at_10m_db=50.0, absorption_db_per_m=ALPHA)
        fit = fit_tvg_models(ds)
        check = check_tvg(ds, fit.tvg_params, fit.window_m)
        assert check.spread_db == pytest.approx(fit.best_model.rms_db, abs=0.05)

    def test_check_wrong_model_leaves_trend(self):
        ds = _ds(absorption_db_per_m=ALPHA)
        fit = fit_tvg_models(ds)
        wrong = check_tvg(ds, {"spreading_loss_db": 20.0, "absorption_db_per_m": 0.0}, fit.window_m)
        assert wrong.trend_db_per_10m < -2.0
        assert wrong.end_to_end_db < -10.0
        assert wrong.model_name == "20 log R"

    def test_check_crops_at_window_end(self):
        ds = _ds()
        check = check_tvg(ds, TVGParams(spreading_loss_db=30.0, absorption_db_per_m=0.0), (15.5, 60.0))
        assert check.crop_m == 60.0
        assert np.isnan(check.level_db[check.range_m > 61.0]).all()

    def test_check_relative_level_zero_mean_in_window(self):
        ds = _ds()
        check = check_tvg(ds, TVGParams(spreading_loss_db=30.0, absorption_db_per_m=0.0), (15.5, 70.0))
        sel = (check.range_m >= 15.5) & (check.range_m <= 70.0)
        assert np.nanmean(check.relative_db[sel]) == pytest.approx(0.0, abs=1e-6)

    def test_check_absent_absorption_uses_francois_garrison(self):
        ds = _ds()
        check = check_tvg(ds, {"spreading_loss_db": 30.0}, (15.5, 70.0))
        assert check.params["absorption_db_per_m"] == pytest.approx(ALPHA)

    def test_check_short_window_no_check(self):
        check = check_tvg(_ds(), TVGParams(), (20.0, 20.5))
        assert check.spread_db is None
        assert check.notes
