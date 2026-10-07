"""
Smoke tests for sidescanpy.viz.tvg_diagnostics.

Matches how viz/waterfall.py and viz/gridded.py are tested: figure/axes
returned, no exceptions, key labels present — not pixel comparisons.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.analysis.tvg import fit_tvg_curve
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db
from sidescanpy.viz.tvg_diagnostics import (
    _params_label,
    plot_altitude_bin_comparison,
    plot_candidate_fit,
    plot_mean_vs_range,
    plot_multi_frequency_comparison,
    plot_per_ping_profiles,
    plot_tvg_gain_curve,
)


@pytest.fixture
def sample_binned_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "range_min_m": [0.0, 5.0, 10.0],
            "range_max_m": [5.0, 10.0, 15.0],
            "mean": [-10.0, -15.0, -20.0],
            "median": [-10.0, -15.0, -20.0],
            "std": [1.0, 1.0, 1.0],
            "p10": [-12.0, -17.0, -22.0],
            "p90": [-8.0, -13.0, -18.0],
            "count": [100, 100, 100],
        }
    )


@pytest.fixture
def sample_ds_with_gain() -> xr.Dataset:
    n_ping, n_channel, n_sample = 3, 2, 20
    rng = np.random.default_rng(0)
    ground_range = np.broadcast_to(
        np.linspace(1.0, 50.0, n_sample), (n_ping, n_channel, n_sample)
    ).astype(np.float64)
    return xr.Dataset(
        {
            "ground_range_m": (["ping", "channel", "sample"], ground_range),
            "tvg_gain_db": (
                ["ping", "channel", "sample"],
                rng.normal(size=(n_ping, n_channel, n_sample)),
            ),
            "backscatter_db_corrected": (
                ["ping", "channel", "sample"],
                rng.normal(size=(n_ping, n_channel, n_sample)),
            ),
        }
    )


class TestPlotMeanVsRange:
    def test_returns_fig_ax(self, sample_binned_df):
        fig, ax = plot_mean_vs_range(sample_binned_df)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_reuses_provided_ax(self, sample_binned_df):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plot_mean_vs_range(sample_binned_df, ax=ax1)
        assert ax2 is ax1
        assert fig2 is fig1
        plt.close(fig1)

    def test_overlay_twice_for_before_after(self, sample_binned_df):
        fig, ax = plot_mean_vs_range(sample_binned_df, label="raw")
        plot_mean_vs_range(sample_binned_df, ax=ax, label="corrected")
        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_sample_index_columns_supported(self):
        df = pd.DataFrame(
            {
                "sample_min": [0, 50],
                "sample_max": [50, 100],
                "mean": [-5.0, -10.0],
                "p10": [-6.0, -11.0],
                "p90": [-4.0, -9.0],
            }
        )
        fig, ax = plot_mean_vs_range(df)
        assert ax.get_xlabel() == "Sample index"
        plt.close(fig)

    def test_default_linewidth_matches_matplotlib_default(self, sample_binned_df):
        """1.5 is matplotlib's own default -- must be a no-op for every
        existing caller that never passed linewidth."""
        fig, ax = plot_mean_vs_range(sample_binned_df)
        assert ax.get_lines()[0].get_linewidth() == pytest.approx(1.5)
        plt.close(fig)

    def test_linewidth_override_applies(self, sample_binned_df):
        fig, ax = plot_mean_vs_range(sample_binned_df, linewidth=3.0)
        assert ax.get_lines()[0].get_linewidth() == pytest.approx(3.0)
        plt.close(fig)

    def test_default_zorder_sits_above_a_typical_prior_layer(self, sample_binned_df):
        """3.0 default should sit above plot_per_ping_profiles' own
        lines, which don't set an explicit zorder (matplotlib default
        for the first line drawn on an axis is 2.0)."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])  # a stand-in "background" layer, default zorder
        plot_mean_vs_range(sample_binned_df, ax=ax)
        assert ax.get_lines()[-1].get_zorder() > ax.get_lines()[0].get_zorder()
        plt.close(fig)


class TestPlotTvgGainCurve:
    def test_returns_fig_ax(self, sample_ds_with_gain):
        fig, ax = plot_tvg_gain_curve(sample_ds_with_gain)
        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestPlotPerPingProfiles:
    def test_returns_fig_ax(self, sample_ds_with_gain):
        fig, ax = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2
        )
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_n_pings_larger_than_available_does_not_crash(self, sample_ds_with_gain):
        fig, ax = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=1000
        )
        plt.close(fig)

    def test_channel_idx_default_zero_matches_prior_behavior(self, sample_ds_with_gain):
        fig, ax = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1,
        )
        fig_explicit, ax_explicit = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1, channel_idx=0,
        )
        default_data = [line.get_ydata() for line in ax.get_lines()]
        explicit_data = [line.get_ydata() for line in ax_explicit.get_lines()]
        assert all(np.array_equal(a, b) for a, b in zip(default_data, explicit_data))
        plt.close(fig)
        plt.close(fig_explicit)

    def test_channel_idx_one_selects_a_different_channel(self, sample_ds_with_gain):
        fig0, ax0 = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1, channel_idx=0,
        )
        fig1, ax1 = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1, channel_idx=1,
        )
        data0 = [line.get_ydata() for line in ax0.get_lines()]
        data1 = [line.get_ydata() for line in ax1.get_lines()]
        assert not all(np.array_equal(a, b) for a, b in zip(data0, data1))
        plt.close(fig0)
        plt.close(fig1)

    def test_x_sign_default_is_a_no_op(self, sample_ds_with_gain):
        fig, ax = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1,
        )
        fig_explicit, ax_explicit = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1, x_sign=1.0,
        )
        default_x = [line.get_xdata() for line in ax.get_lines()]
        explicit_x = [line.get_xdata() for line in ax_explicit.get_lines()]
        assert all(np.array_equal(a, b) for a, b in zip(default_x, explicit_x))
        plt.close(fig)
        plt.close(fig_explicit)

    def test_x_sign_negative_mirrors_the_x_axis(self, sample_ds_with_gain):
        """x_sign=-1 must negate every plotted x-coordinate while
        keeping each x paired with its original y -- each line is drawn
        sorted by its *own* (possibly negated) x, so a plain
        element-wise zip of the two runs' raw line data isn't
        comparable; re-sort each line by |x| first to recover the
        shared physical (ping, sample) order before comparing."""
        fig, ax = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1,
        )
        fig_mirrored, ax_mirrored = plot_per_ping_profiles(
            sample_ds_with_gain, value_var="backscatter_db_corrected", n_pings=2, seed=1, x_sign=-1.0,
        )
        for line, line_mirrored in zip(ax.get_lines(), ax_mirrored.get_lines()):
            x, y = line.get_xdata(), line.get_ydata()
            xm, ym = line_mirrored.get_xdata(), line_mirrored.get_ydata()
            order = np.argsort(np.abs(x))
            order_m = np.argsort(np.abs(xm))
            assert np.allclose(-x[order], xm[order_m])
            assert np.array_equal(y[order], ym[order_m])
        plt.close(fig)
        plt.close(fig_mirrored)


class TestPlotMultiFrequencyComparison:
    def test_returns_fig_ax_with_legend(self, sample_binned_df):
        fig, ax = plot_multi_frequency_comparison(
            {"1495 kHz": sample_binned_df, "547.5 kHz": sample_binned_df}
        )
        assert len(ax.get_lines()) == 2
        assert ax.get_legend() is not None
        plt.close(fig)


class TestPlotAltitudeBinComparison:
    def test_returns_fig_ax_with_legend(self, sample_binned_df):
        fig, ax = plot_altitude_bin_comparison(
            {"altitude_3.0_5.0m": sample_binned_df, "altitude_5.0_8.0m": sample_binned_df}
        )
        assert len(ax.get_lines()) == 2
        assert ax.get_legend() is not None
        plt.close(fig)


@pytest.fixture
def sample_fit_result():
    r = np.arange(1.0, 80.0, 1.0)
    params = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)
    mean = 50.0 - tvg_gain_db(r, params)
    binned = pd.DataFrame({
        "range_min_m": r - 0.5, "range_max_m": r + 0.5,
        "mean": mean, "count": np.full_like(r, 1000, dtype=int),
    })
    return fit_tvg_curve(binned, absorption_db_per_m=0.15)


class TestPlotCandidateFit:
    def test_returns_fig_ax(self, sample_fit_result):
        fig, ax = plot_candidate_fit(
            sample_fit_result.best,
            sample_fit_result.observed_range_m,
            sample_fit_result.observed_loss_db,
        )
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_repeated_calls_show_observed_once(self, sample_fit_result):
        fig, ax = plt.subplots()
        for i, candidate in enumerate(sample_fit_result.candidates):
            plot_candidate_fit(
                candidate,
                sample_fit_result.observed_range_m,
                sample_fit_result.observed_loss_db,
                ax=ax,
                show_observed=(i == 0),
            )
        # 1 observed line + 1 zero-loss hline + 1 reference-range vline
        # + 4 candidate lines, not 4x each of the observed/hline/vline.
        assert len(ax.get_lines()) == 7
        plt.close(fig)

    def test_x_axis_is_absolute_range_not_relative_to_r_ref(self, sample_fit_result):
        """r_ref_m is not 0 for this fixture -- if the x-axis were still
        relative (range_m - r_ref_m) the observed points would be
        shifted left of their true range values."""
        fig, ax = plot_candidate_fit(
            sample_fit_result.best,
            sample_fit_result.observed_range_m,
            sample_fit_result.observed_loss_db,
        )
        observed_line = next(l for l in ax.get_lines() if l.get_label() == "observed")
        assert sample_fit_result.best.r_ref_m != 0.0
        assert np.allclose(observed_line.get_xdata(), sample_fit_result.observed_range_m)
        plt.close(fig)

    def test_legend_includes_fit_range_entry(self, sample_fit_result):
        fig, ax = plot_candidate_fit(
            sample_fit_result.best,
            sample_fit_result.observed_range_m,
            sample_fit_result.observed_loss_db,
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any(label.startswith("fit range (") for label in labels)
        plt.close(fig)

    def test_use_fixed_scale_default_draws_literal_unscaled_model(self):
        """Default (use_fixed_scale=True) must draw the model at
        scale=1.0, not the free-fit scale -- matches what
        fit_range_decay's classification is now actually based on
        (2026-08-24 change). Uses a deliberately scaled curve (true
        scale 2.4) so the two lines are visibly, checkably different,
        unlike an exact-match fixture where they'd coincide."""
        r = np.arange(1.0, 40.0, 1.0)
        true_scale = 2.4
        params = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)
        mean = 50.0 - true_scale * tvg_gain_db(r, params)
        binned = pd.DataFrame({
            "range_min_m": r - 0.5, "range_max_m": r + 0.5,
            "mean": mean, "count": np.full_like(r, 1000, dtype=int),
        })
        fit_result = fit_tvg_curve(binned, absorption_db_per_m=0.15, r_ref_m=1.5, r_fit_max_m=39.5)
        absorption_only = next(c for c in fit_result.candidates if c.name == "absorption_only")
        assert absorption_only.scale == pytest.approx(true_scale, abs=1e-3)  # sanity: fixture behaves as intended

        fig_fixed, ax_fixed = plot_candidate_fit(
            absorption_only, fit_result.observed_range_m, fit_result.observed_loss_db, use_fixed_scale=True,
        )
        fig_free, ax_free = plot_candidate_fit(
            absorption_only, fit_result.observed_range_m, fit_result.observed_loss_db, use_fixed_scale=False,
        )
        model_fixed = next(l for l in ax_fixed.get_lines() if l.get_label().startswith("absorption_only"))
        model_free = next(l for l in ax_free.get_lines() if l.get_label().startswith("absorption_only"))
        assert not np.allclose(model_fixed.get_ydata(), model_free.get_ydata())
        # at the far edge the fixed (unscaled) line should be much
        # closer to zero loss than the free-fit (2.4x scaled) line,
        # which tracks the (also 2.4x) observed curve closely
        assert abs(model_fixed.get_ydata()[-1]) < abs(model_free.get_ydata()[-1])
        plt.close(fig_fixed)
        plt.close(fig_free)

    def test_use_fixed_scale_legend_shows_fixed_scale_rmse_and_literal_params(self):
        r = np.arange(1.0, 40.0, 1.0)
        params = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)
        mean = 50.0 - 2.4 * tvg_gain_db(r, params)
        binned = pd.DataFrame({
            "range_min_m": r - 0.5, "range_max_m": r + 0.5,
            "mean": mean, "count": np.full_like(r, 1000, dtype=int),
        })
        fit_result = fit_tvg_curve(binned, absorption_db_per_m=0.15, r_ref_m=1.5, r_fit_max_m=39.5)
        absorption_only = next(c for c in fit_result.candidates if c.name == "absorption_only")

        fig, ax = plot_candidate_fit(
            absorption_only, fit_result.observed_range_m, fit_result.observed_loss_db, use_fixed_scale=True,
        )
        label = next(l.get_label() for l in ax.get_lines() if l.get_label().startswith("absorption_only"))
        assert f"RMSE={absorption_only.fixed_scale_rmse_db:.2f}dB" in label
        # the literal (uncalibrated) coefficient, not the free-fit scale
        assert "absorption=0.15dB/m" in label
        assert "free-fit scale" not in label
        plt.close(fig)

    def test_params_label_lists_only_nonzero_terms(self):
        assert _params_label(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0)) == ""
        assert _params_label(TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.0)) == "spreading=20.0dB"
        assert _params_label(TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)) == "absorption=0.15dB/m"
        assert _params_label(TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.15)) == (
            "spreading=20.0dB, absorption=0.15dB/m"
        )

    def test_flat_candidate_label_has_no_trailing_comma(self):
        r = np.arange(1.0, 40.0, 1.0)
        mean = np.full_like(r, 50.0)
        binned = pd.DataFrame({
            "range_min_m": r - 0.5, "range_max_m": r + 0.5,
            "mean": mean, "count": np.full_like(r, 1000, dtype=int),
        })
        fit_result = fit_tvg_curve(binned, absorption_db_per_m=0.15)
        flat = next(c for c in fit_result.candidates if c.name == "flat")

        fig, ax = plot_candidate_fit(flat, fit_result.observed_range_m, fit_result.observed_loss_db)
        label = next(l.get_label() for l in ax.get_lines() if l.get_label().startswith("flat"))
        assert label == f"flat (RMSE={flat.fixed_scale_rmse_db:.2f}dB)"
        plt.close(fig)

    def test_model_line_extends_back_to_zero_range(self):
        """The dashed model line must reach range=0 for reference, even
        though observed bins start well past it (mirrors a real nadir-
        gap-excluded curve, which never has data near zero range)."""
        r = np.arange(10.0, 40.0, 1.0)  # no observed data below 10m
        params = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)
        mean = 50.0 - tvg_gain_db(r, params)
        binned = pd.DataFrame({
            "range_min_m": r - 0.5, "range_max_m": r + 0.5,
            "mean": mean, "count": np.full_like(r, 1000, dtype=int),
        })
        fit_result = fit_tvg_curve(binned, absorption_db_per_m=0.15)

        fig, ax = plot_candidate_fit(fit_result.best, fit_result.observed_range_m, fit_result.observed_loss_db)
        model_line = next(l for l in ax.get_lines() if l.get_label().startswith(fit_result.best.name))
        assert fit_result.observed_range_m.min() >= 10.0  # sanity: fixture has no data below 10m
        assert model_line.get_xdata().min() == pytest.approx(0.0)
        plt.close(fig)

    def test_model_line_crosses_zero_at_r_ref(self):
        """Every candidate's model_loss is 0 exactly at r_ref by
        construction -- confirms the model line is drawn in absolute
        range terms (not relative) and that the near-side (r < r_ref)
        portion, where relative loss goes negative, is present. Uses an
        explicit mid-range r_ref_m (the sample_fit_result fixture's own
        auto-selected r_ref happens to land on the very first/nearest
        bin, which has no near side to check)."""
        r = np.arange(1.0, 80.0, 1.0)
        params = TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.15)
        mean = 50.0 - tvg_gain_db(r, params)
        binned = pd.DataFrame({
            "range_min_m": r - 0.5, "range_max_m": r + 0.5,
            "mean": mean, "count": np.full_like(r, 1000, dtype=int),
        })
        fit_result = fit_tvg_curve(binned, absorption_db_per_m=0.15, r_ref_m=40.0)

        # model_n_points=2000 over a 0-79m domain -- fine enough that the
        # nearest sampled point to r_ref_m=40.0 is well within 1e-6 of it,
        # so the check below reflects the true crossing, not
        # discretization error from a coarser default sampling.
        fig, ax = plot_candidate_fit(
            fit_result.best, fit_result.observed_range_m, fit_result.observed_loss_db, model_n_points=2000,
        )
        model_line = next(l for l in ax.get_lines() if l.get_label().startswith(fit_result.best.name))
        x, y = model_line.get_xdata(), model_line.get_ydata()
        i_ref = int(np.argmin(np.abs(x - fit_result.best.r_ref_m)))
        assert y[i_ref] == pytest.approx(0.0, abs=1e-2)
        assert np.any(x < fit_result.best.r_ref_m), "expected near-side (r < r_ref) points on the model line"
        assert np.any(y[x < fit_result.best.r_ref_m] < 0), "expected negative relative loss on the near side"
        plt.close(fig)
