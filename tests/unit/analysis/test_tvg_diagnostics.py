"""
Unit tests for sidescanpy.analysis.tvg_diagnostics.

Stage 1 of docs/planning/v0.8.0-tvg-diagnostics.md: diagnostic tooling
for investigating the range-dependent TVG residual (CLAUDE.md Known
Issue #4) and floor-clipped samples (Known Issue #5), built on top of
apply_tvg()'s existing public API without changing it.

Uses small synthetic fixtures with hand-computed expected values, not
real survey data, per the project's TDD convention.

2026 rshomberg@uri.edu
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.analysis.tvg_diagnostics import (
    _linear_to_db,
    before_after_backscatter,
    bin_pings_by_altitude,
    fit_hardware_tvg_curve,
    mean_backscatter_vs_range,
    mean_backscatter_vs_sample_index,
)
from sidescanpy.processing.pipeline import apply_correction_steps
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db


# ============================================================
# HELPERS
# ============================================================

def _make_test_dataset(
    *,
    n_pings: int = 5,
    n_samples: int = 100,
    sample_rate_hz: float = 1000.0,
    altitude_m: float | np.ndarray = 10.0,
    sound_vel_ms: float | np.ndarray = 1500.0,
    pitch_deg: float | np.ndarray = 0.0,
    depth_m: float | np.ndarray = 50.0,
    backscatter_val: float = 1.0,
) -> xr.Dataset:
    """Minimal valid raw sidescan dataset. Mirrors test_geometric.py's helper."""
    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }

    def _per_ping(value, dtype=np.float32):
        if np.isscalar(value):
            return np.full(n_pings, value, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    data_vars = {
        "backscatter_intensity": (
            ["ping", "channel", "sample"],
            np.full((n_pings, 2, n_samples), backscatter_val, dtype=np.float32),
        ),
        "backscatter_raw": (
            ["ping", "channel", "sample"],
            np.full((n_pings, 2, n_samples), backscatter_val, dtype=np.float32),
        ),
        "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
        "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
        "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
        "pitch_deg": (["ping"], _per_ping(pitch_deg)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], _per_ping(depth_m)),
        "altitude_m": (["ping"], _per_ping(altitude_m)),
        "fathometer_reading_m": (["ping"], _per_ping(altitude_m)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], _per_ping(sound_vel_ms)),
    }

    attrs = {
        "freq_hz_start": 1_495_000,
        "freq_hz_end": 1_495_000,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }

    return xr.Dataset(data_vars, coords, attrs)


# ============================================================
# _linear_to_db
# ============================================================

class TestLinearToDb:
    """Must exactly match apply_tvg()'s own step-1 arithmetic (intentional
    duplication per the plan doc — apply_tvg has no public way to get
    dB-without-gain)."""

    def test_matches_20log10(self):
        x = np.array([1.0, 2.0, 10.0, 81.0], dtype=np.float64)
        result = _linear_to_db(x)
        np.testing.assert_allclose(result, 20.0 * np.log10(x), rtol=1e-6)

    def test_zero_clips_to_floor(self):
        result = _linear_to_db(np.array([0.0]))
        assert np.isclose(result[0], 20.0 * np.log10(1e-6), atol=1e-6)

    def test_unity_maps_to_zero_db(self):
        result = _linear_to_db(np.array([1.0]))
        assert np.isclose(result[0], 0.0, atol=1e-9)

    def test_accepts_xr_dataarray(self):
        da = xr.DataArray(np.array([1.0, 2.0]), dims=["x"])
        result = _linear_to_db(da)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, 20.0 * np.log10([1.0, 2.0]), rtol=1e-6)


# ============================================================
# before_after_backscatter
# ============================================================

class TestBeforeAfterBackscatter:
    def test_ground_range_matches_geometric_only_pass(self):
        ds_raw = _make_test_dataset()
        out = before_after_backscatter(ds_raw)
        ds_geom_only = apply_correction_steps(ds_raw, {"geometric"})
        np.testing.assert_allclose(
            out["ground_range_m"].values,
            ds_geom_only["ground_range_m"].values,
            equal_nan=True,
        )

    def test_ground_range_matches_tvg_and_geometric_pass(self):
        """Regression guard: geometric ranges must not depend on whether
        TVG has been applied first."""
        ds_raw = _make_test_dataset()
        ds_geom_only = apply_correction_steps(ds_raw, {"geometric"})
        ds_tvg_geom = apply_correction_steps(ds_raw, {"tvg", "geometric"})
        np.testing.assert_allclose(
            ds_geom_only["ground_range_m"].values,
            ds_tvg_geom["ground_range_m"].values,
            equal_nan=True,
        )

    def test_backscatter_db_raw_matches_linear_to_db(self):
        ds_raw = _make_test_dataset(backscatter_val=2.0)
        out = before_after_backscatter(ds_raw)
        expected = _linear_to_db(ds_raw["backscatter_intensity"].values)
        np.testing.assert_allclose(out["backscatter_db_raw"].values, expected, rtol=1e-5)

    def test_tvg_gain_equals_corrected_minus_raw(self):
        ds_raw = _make_test_dataset(backscatter_val=2.0)
        out = before_after_backscatter(ds_raw)
        np.testing.assert_allclose(
            out["tvg_gain_db"].values,
            (out["backscatter_db_corrected"] - out["backscatter_db_raw"]).values,
            rtol=1e-5,
        )

    def test_tvg_gain_matches_corrected_db_at_unity_backscatter(self):
        # backscatter_val=1.0 -> raw dB is exactly 0, so gain == corrected value
        ds_raw = _make_test_dataset(backscatter_val=1.0)
        out = before_after_backscatter(ds_raw)
        np.testing.assert_allclose(
            out["tvg_gain_db"].values, out["backscatter_db_corrected"].values, rtol=1e-5
        )

    def test_custom_tvg_params_changes_gain(self):
        ds_raw = _make_test_dataset(backscatter_val=1.0, n_samples=200, sample_rate_hz=1000.0)
        default_out = before_after_backscatter(ds_raw)
        custom_out = before_after_backscatter(
            ds_raw, TVGParams(absorption_db_per_m=2.0, spreading_loss_db=20.0)
        )
        # Higher absorption coefficient -> strictly larger gain at far range.
        assert (
            custom_out["tvg_gain_db"].values[..., -1].mean()
            > default_out["tvg_gain_db"].values[..., -1].mean()
        )

    def test_does_not_mutate_input(self):
        ds_raw = _make_test_dataset()
        before_after_backscatter(ds_raw)
        assert ds_raw.attrs["processing_level"] == "raw"
        assert "ground_range_m" not in ds_raw.data_vars
        assert "tvg_gain_db" not in ds_raw.data_vars


# ============================================================
# mean_backscatter_vs_range
# ============================================================

class TestMeanBackscatterVsRange:
    def test_known_bins_mean_and_count(self):
        ds = xr.Dataset(
            {
                "value": (["obs"], np.array([1.0, 3.0, 10.0, 20.0])),
                "ground_range_m": (["obs"], np.array([1.0, 4.0, 6.0, 9.0])),
            }
        )
        result = mean_backscatter_vs_range(ds, "value", bin_width_m=5.0)
        assert len(result) == 2
        assert result.iloc[0]["count"] == 2
        assert np.isclose(result.iloc[0]["mean"], 2.0)
        assert result.iloc[1]["count"] == 2
        assert np.isclose(result.iloc[1]["mean"], 15.0)

    def test_exclude_below_db_filters_floor_clipped(self):
        ds = xr.Dataset(
            {
                "value": (["obs"], np.array([-125.0, -125.0, 5.0, 6.0])),
                "ground_range_m": (["obs"], np.array([1.0, 2.0, 3.0, 4.0])),
            }
        )
        result_all = mean_backscatter_vs_range(ds, "value", bin_width_m=5.0)
        assert result_all.iloc[0]["count"] == 4

        result_filtered = mean_backscatter_vs_range(
            ds, "value", bin_width_m=5.0, exclude_below_db=-110.0
        )
        assert result_filtered.iloc[0]["count"] == 2
        assert np.isclose(result_filtered.iloc[0]["mean"], 5.5)

    def test_exclude_below_db_none_by_default(self):
        """Default must include everything — Known Issue #5's root cause
        isn't confirmed, so filtering must be opt-in, not silently assumed."""
        ds = xr.Dataset(
            {
                "value": (["obs"], np.array([-125.0, 5.0])),
                "ground_range_m": (["obs"], np.array([1.0, 2.0])),
            }
        )
        result = mean_backscatter_vs_range(ds, "value", bin_width_m=5.0)
        assert result.iloc[0]["count"] == 2

    def test_nan_values_excluded(self):
        ds = xr.Dataset(
            {
                "value": (["obs"], np.array([1.0, np.nan, 3.0])),
                "ground_range_m": (["obs"], np.array([1.0, 2.0, np.nan])),
            }
        )
        result = mean_backscatter_vs_range(ds, "value", bin_width_m=5.0)
        assert result.iloc[0]["count"] == 1
        assert np.isclose(result.iloc[0]["mean"], 1.0)

    def test_empty_after_filtering_returns_empty_dataframe(self):
        ds = xr.Dataset(
            {
                "value": (["obs"], np.array([-125.0])),
                "ground_range_m": (["obs"], np.array([1.0])),
            }
        )
        result = mean_backscatter_vs_range(ds, "value", exclude_below_db=-110.0)
        assert len(result) == 0

    def test_includes_percentile_columns(self):
        ds = xr.Dataset(
            {
                "value": (["obs"], np.arange(10.0)),
                "ground_range_m": (["obs"], np.full(10, 2.0)),
            }
        )
        result = mean_backscatter_vs_range(ds, "value", bin_width_m=5.0)
        for col in ("mean", "median", "std", "p10", "p90", "count"):
            assert col in result.columns


# ============================================================
# mean_backscatter_vs_sample_index
# ============================================================

class TestMeanBackscatterVsSampleIndex:
    def test_known_bins(self):
        ds = xr.Dataset(
            {"value": (["ping", "sample"], np.array([[1.0, 2.0, 3.0, 4.0]]))},
            coords={"sample": np.arange(4)},
        )
        result = mean_backscatter_vs_sample_index(ds, "value", bin_width=2)
        assert len(result) == 2
        assert np.isclose(result.iloc[0]["mean"], 1.5)
        assert np.isclose(result.iloc[1]["mean"], 3.5)

    def test_leading_run_detection_shape(self):
        """Sanity check matching Known Issue #5's actual shape: a fixed
        leading run of floor-clipped samples at low sample indices."""
        n_samples = 20
        values = np.full((1, n_samples), -125.0)
        values[0, 5:] = -20.0
        ds = xr.Dataset(
            {"value": (["ping", "sample"], values)},
            coords={"sample": np.arange(n_samples)},
        )
        result = mean_backscatter_vs_sample_index(ds, "value", bin_width=5)
        assert np.isclose(result.iloc[0]["mean"], -125.0)
        assert result.iloc[1]["mean"] > -125.0

    def test_exclude_below_db(self):
        ds = xr.Dataset(
            {"value": (["ping", "sample"], np.array([[-125.0, -125.0, 5.0, 6.0]]))},
            coords={"sample": np.arange(4)},
        )
        result = mean_backscatter_vs_sample_index(ds, "value", bin_width=4, exclude_below_db=-110.0)
        assert result.iloc[0]["count"] == 2
        assert np.isclose(result.iloc[0]["mean"], 5.5)


# ============================================================
# bin_pings_by_altitude
# ============================================================

class TestBinPingsByAltitude:
    def test_default_three_bins_preserve_total_ping_count(self):
        n = 9
        ds = xr.Dataset(
            {
                "altitude_m": (["ping"], np.arange(1, n + 1, dtype=np.float32)),
                "value": (["ping"], np.arange(n, dtype=np.float32)),
            }
        )
        bins = bin_pings_by_altitude(ds, n_bins=3)
        assert len(bins) == 3
        total = sum(b.sizes["ping"] for b in bins.values())
        assert total == n

    def test_bins_ordered_by_increasing_altitude(self):
        n = 12
        ds = xr.Dataset(
            {
                "altitude_m": (["ping"], np.arange(1, n + 1, dtype=np.float32)),
                "value": (["ping"], np.arange(n, dtype=np.float32)),
            }
        )
        bins = bin_pings_by_altitude(ds, n_bins=3)
        bin_datasets = list(bins.values())
        maxes = [float(b["altitude_m"].max()) for b in bin_datasets]
        mins = [float(b["altitude_m"].min()) for b in bin_datasets]
        for i in range(len(bin_datasets) - 1):
            assert maxes[i] <= mins[i + 1]

    def test_n_bins_parameter(self):
        n = 20
        ds = xr.Dataset(
            {
                "altitude_m": (["ping"], np.linspace(1.0, 20.0, n, dtype=np.float32)),
                "value": (["ping"], np.arange(n, dtype=np.float32)),
            }
        )
        bins = bin_pings_by_altitude(ds, n_bins=4)
        assert len(bins) == 4

    def test_nan_altitude_pings_excluded(self):
        altitude = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
        ds = xr.Dataset(
            {
                "altitude_m": (["ping"], altitude),
                "value": (["ping"], np.arange(6, dtype=np.float32)),
            }
        )
        bins = bin_pings_by_altitude(ds, n_bins=2)
        total = sum(b.sizes["ping"] for b in bins.values())
        assert total == 5


# ============================================================
# fit_hardware_tvg_curve
# ============================================================

def _make_binned_curve(
    true_params: TVGParams,
    *,
    r_min: float = 1.0,
    r_max: float = 80.0,
    step: float = 1.0,
    source_level_db: float = 50.0,
    count: int = 1000,
) -> pd.DataFrame:
    """A mean_backscatter_vs_range()-shaped DataFrame built from an EXACT
    known TVGParams -- reuses tvg_gain_db() (the function under test's
    own dependency) so the fixture is guaranteed self-consistent, rather
    than a hand-derived formula that could drift from it."""
    r = np.arange(r_min, r_max, step)
    mean = source_level_db - tvg_gain_db(r, true_params)
    return pd.DataFrame({
        "range_min_m": r - step / 2,
        "range_max_m": r + step / 2,
        "mean": mean,
        "median": mean,
        "std": np.zeros_like(mean),
        "p10": mean,
        "p90": mean,
        "count": np.full_like(r, count, dtype=int),
    })


_ABSORPTION = 0.15  # arbitrary, distinct from any absorption_for_frequency() default


class TestFitHardwareTvgCurve:
    def test_flat_curve_picks_flat(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0))
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        assert result.best.name == "flat"
        assert result.best.rmse_db == pytest.approx(0.0, abs=1e-9)

    def test_spreading_only_curve_picks_spreading_only(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.0))
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        assert result.best.name == "spreading_only"
        assert result.best.scale == pytest.approx(1.0, abs=1e-6)
        assert result.best.rmse_db == pytest.approx(0.0, abs=1e-6)

    def test_absorption_only_curve_picks_absorption_only(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=_ABSORPTION))
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        assert result.best.name == "absorption_only"
        assert result.best.scale == pytest.approx(1.0, abs=1e-6)
        assert result.best.rmse_db == pytest.approx(0.0, abs=1e-6)

    def test_full_physics_curve_picks_full_physics(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=20.0, absorption_db_per_m=_ABSORPTION))
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        assert result.best.name == "full_physics"
        assert result.best.scale == pytest.approx(1.0, abs=1e-6)
        assert result.best.rmse_db == pytest.approx(0.0, abs=1e-6)

    def test_all_four_candidates_present_in_fixed_order(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0))
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        assert [c.name for c in result.candidates] == [
            "flat", "spreading_only", "absorption_only", "full_physics",
        ]

    def test_r_ref_auto_selection_skips_low_count_bins(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=_ABSORPTION))
        # First bin (r~1) has too few samples; second (r~2) is fine.
        binned.loc[0, "count"] = 5
        result = fit_hardware_tvg_curve(
            binned, absorption_db_per_m=_ABSORPTION, min_count_for_r_ref=30,
        )
        assert result.best.r_ref_m == pytest.approx(binned["range_min_m"].iloc[1] + 0.5)

    def test_r_ref_auto_selection_ignores_far_edge_noise_spike(self):
        # A near-flat curve (as from a survey where hardware has already
        # compensated almost everything) with a tiny noise-level bump at
        # the far edge -- an unrestricted global-max search would anchor
        # r_ref_m there, collapsing r_fit_max_m's window to that single
        # bin and making every candidate's fit degenerate (RMSE=0 for
        # all). Reproduces a real bug found against a genuinely-flat
        # Klein survey.
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0), r_max=80.0)
        binned.loc[binned.index[-1], "mean"] += 0.05
        result = fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION)
        max_range = (binned["range_min_m"].iloc[-1] + binned["range_max_m"].iloc[-1]) / 2.0
        assert result.best.r_ref_m < max_range
        assert result.best.n_bins_used > 1

    def test_r_fit_max_auto_selection_matches_documented_formula(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=_ABSORPTION), r_max=80.0)
        result = fit_hardware_tvg_curve(
            binned, absorption_db_per_m=_ABSORPTION, fit_fraction=0.6,
        )
        r_ref = result.best.r_ref_m
        max_range = (binned["range_min_m"].iloc[-1] + binned["range_max_m"].iloc[-1]) / 2.0
        expected = r_ref + 0.6 * (max_range - r_ref)
        assert result.best.r_fit_max_m == pytest.approx(expected)

    def test_explicit_r_ref_and_r_fit_max_override_auto_selection(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=_ABSORPTION))
        result = fit_hardware_tvg_curve(
            binned, absorption_db_per_m=_ABSORPTION, r_ref_m=10.0, r_fit_max_m=40.0,
        )
        assert result.best.r_ref_m == 10.0
        assert result.best.r_fit_max_m == 40.0

    def test_empty_binned_raises(self):
        empty = pd.DataFrame(columns=["range_min_m", "range_max_m", "mean", "count"])
        with pytest.raises(ValueError, match="no rows"):
            fit_hardware_tvg_curve(empty, absorption_db_per_m=_ABSORPTION)

    def test_no_bin_meets_min_count_raises(self):
        binned = _make_binned_curve(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0), count=1)
        with pytest.raises(ValueError, match="min_count_for_r_ref"):
            fit_hardware_tvg_curve(binned, absorption_db_per_m=_ABSORPTION, min_count_for_r_ref=30)
