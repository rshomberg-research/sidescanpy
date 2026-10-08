"""
Tests for sidescanpy.viz.tvg.

Matches how viz/waterfall.py and viz/gridded.py are tested: figure/axes
returned, no exceptions, key labels present — not pixel comparisons.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.analysis.tvg import RangeDecayResult, fit_range_decay, fit_tvg_curve
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db
from sidescanpy.viz.tvg import (
    _params_label,
    plot_altitude_bin_comparison,
    plot_candidate_fit,
    plot_mean_vs_range,
    plot_multi_frequency_comparison,
    plot_per_ping_profiles,
    plot_range_decay_fit,
    plot_range_decay_residuals,
    plot_tvg_gain_curve,
    set_fit_window_limits,
)
from tests.unit.synthetic import make_raw_dataset


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


def _make_scaled_spreading_dataset(*, scale: float, n_pings: int = 30, n_samples: int = 300) -> xr.Dataset:
    """Raw data that follows `spreading_only`'s decay shape exactly, but
    amplified by `scale` relative to the literal (scale=1.0) model --
    e.g. `scale=1.6` means the real decay is 1.6x steeper than the
    textbook spreading-loss term alone. Built so `fit_hardware_tvg_curve`'s
    free-fit scale for the `spreading_only` candidate converges close to
    `scale` (a near-perfect free fit) while its fixed-scale (1.0)
    residual stays large -- the two `plot_range_decay_residuals` modes should
    then plot visibly different data for the same candidate, which is
    exactly the distinction the 2026-08-25 fix (`use_fixed_scale`)
    exists to make correct."""
    ds = make_raw_dataset(n_pings=n_pings, n_samples=n_samples, freq_hz=1_495_000.0)
    ds = compute_geometric_ranges(ds)
    params = TVGParams(spreading_loss_db=40.0, absorption_db_per_m=0.0)
    ground_range = ds["ground_range_m"].values
    loss_db = scale * tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
    ref = np.nanmin(loss_db[np.isfinite(loss_db)])
    amplitude = 10.0 ** (-(loss_db - ref) / 20.0)
    amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
    ds["backscatter_intensity"] = (["ping", "channel", "sample"], amplitude)
    return ds


class TestPlotRangeDecayResiduals:
    """Regression coverage for the real bug a user caught 2026-08-25:
    this plot kept using the free-fit scale/RMSE after `plot_candidate_fit`
    (decay_fit.png) was fixed to fixed-scale ranking the day before --
    residuals.png and decay_fit.png silently stopped telling the same
    story. This function had zero test coverage before this fix, which
    is presumably how the drift went unnoticed."""

    def test_no_fit_result_shows_placeholder_title_not_crash(self):
        decay = RangeDecayResult()
        fig, ax = plot_range_decay_residuals(decay)
        try:
            assert ax.get_title() == "No fit available"
        finally:
            plt.close(fig)

    def test_fixed_scale_default_uses_fixed_scale_rmse_in_label(self):
        """A candidate whose free-fit scale is meaningfully off from 1.0
        (by construction here) must show `fixed_scale_rmse_db` in its
        legend label under the default `use_fixed_scale=True`, not the
        free-fit `rmse_db` -- the two are picked to differ substantially
        by this fixture's own construction, so this isn't a coincidental
        pass."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)  # matches _make_scaled_spreading_dataset's own ground_range_m construction
        spreading = next(c for c in decay.fit.candidates if c.name == "spreading_only")
        assert abs(spreading.scale - 1.6) < 0.3  # sanity: free fit actually found the injected scale
        assert spreading.fixed_scale_rmse_db > spreading.rmse_db + 1.0  # and the two RMSEs meaningfully differ

        fig, ax = plot_range_decay_residuals(decay)
        try:
            labels = [line.get_label() for line in ax.get_lines() if "spreading_only" in line.get_label()]
            assert len(labels) == 1
            assert f"{spreading.fixed_scale_rmse_db:.2f}dB" in labels[0]
            assert f"{spreading.rmse_db:.2f}dB" not in labels[0]
        finally:
            plt.close(fig)

    def test_fixed_vs_free_fit_scale_plot_different_residual_values(self):
        """Not just the label -- the actual plotted y-data must change
        too, confirming `model_loss` is really computed at scale=1.0 by
        default rather than `candidate.scale`."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)  # matches _make_scaled_spreading_dataset's own ground_range_m construction

        fig_fixed, ax_fixed = plot_range_decay_residuals(decay, use_fixed_scale=True)
        fig_free, ax_free = plot_range_decay_residuals(decay, use_fixed_scale=False)
        try:
            def _spreading_ydata(ax):
                line = next(l for l in ax.get_lines() if "spreading_only" in l.get_label())
                return np.asarray(line.get_ydata())

            fixed_y = _spreading_ydata(ax_fixed)
            free_y = _spreading_ydata(ax_free)
            assert not np.allclose(fixed_y, free_y)
            # Free fit (scale close to the injected 1.6) should track the
            # observed curve much more closely than the fixed-scale (1.0)
            # residual, which is deliberately under-scaled by this fixture.
            assert np.sqrt(np.mean(fixed_y**2)) > np.sqrt(np.mean(free_y**2))
        finally:
            plt.close(fig_fixed)
            plt.close(fig_free)

    def test_relative_x_false_shifts_xdata_by_r_ref(self):
        """relative_x=False must produce absolute-range x-data, matching
        plot_candidate_fit's current convention -- needed for a genuinely
        aligned shared x-axis when the two are stacked in one figure."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)  # matches _make_scaled_spreading_dataset's own ground_range_m construction
        spreading = next(c for c in decay.fit.candidates if c.name == "spreading_only")

        fig_rel, ax_rel = plot_range_decay_residuals(decay, relative_x=True)
        fig_abs, ax_abs = plot_range_decay_residuals(decay, relative_x=False)
        try:
            def _spreading_xdata(ax):
                line = next(l for l in ax.get_lines() if "spreading_only" in l.get_label())
                return np.asarray(line.get_xdata())

            rel_x = _spreading_xdata(ax_rel)
            abs_x = _spreading_xdata(ax_abs)
            assert np.allclose(abs_x - rel_x, spreading.r_ref_m)
        finally:
            plt.close(fig_rel)
            plt.close(fig_abs)


class TestSetFitWindowLimits:
    """Callers that reuse plot_candidate_fit/
    plot_range_decay_residuals directly but not this limit logic, leaving
    matplotlib's default full-data autoscale in place -- a near-boundary
    outlier bin could then compress the real fit curves into an
    unreadable sliver, and a misconfigured r_ref_m looked like "missing
    observations" rather than the real problem."""

    def test_no_fit_result_leaves_limits_untouched(self):
        decay = RangeDecayResult()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()
        try:
            set_fit_window_limits(ax, decay)
            assert ax.get_xlim() == original_xlim
            assert ax.get_ylim() == original_ylim
        finally:
            plt.close(fig)

    def test_shows_full_range_from_zero_not_cropped_to_fit_window(self):
        """x-limits deliberately always start at 0.0 -- an earlier
        version cropped to the fit window itself, which (combined with
        a misconfigured r_ref_m) read as "not showing the full plot"
        rather than a visibly wrong reference. See this function's own
        docstring for the real report that changed this."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(
            ds, range_var="ground_range_m", bin_width_m=2.0, usable_range_m=40.0,
            r_ref_m="mid_swath", r_fit_min_m="auto",
        )
        r_fit_max_m = decay.r_fit_max_m
        fig, ax = plt.subplots()
        try:
            set_fit_window_limits(ax, decay)
            x_lo, x_hi = ax.get_xlim()
            assert x_lo == pytest.approx(0.0)
            assert x_hi > r_fit_max_m  # padded past the far edge, not cropped to it
        finally:
            plt.close(fig)

    def test_degenerate_zero_width_window_does_not_warn(self, recwarn):
        """A single-bin fit window (r_fit_min_m == r_fit_max_m, a real
        possibility on a very short usable_range_m) would otherwise pad
        by ~0 and hand matplotlib set_xlim(x, x) -- accepted, but with a
        UserWarning ('identical low and high xlims'). x_pad_min_m must
        keep the zoom well-defined instead of relying on that fallback."""
        decay = RangeDecayResult(
            r_fit_min_m=20.0, r_fit_max_m=20.0,
            fit=type("_F", (), {
                "observed_range_m": np.array([20.0]), "observed_loss_db": np.array([1.0]),
            })(),
        )
        fig, ax = plt.subplots()
        try:
            set_fit_window_limits(ax, decay)
            x_lo, x_hi = ax.get_xlim()
            assert x_hi > x_lo
            assert not any("identical low and high xlims" in str(w.message) for w in recwarn.list)
        finally:
            plt.close(fig)

    def test_y_limits_ignore_out_of_window_outlier(self):
        """The whole reason this exists: a single bin far outside the
        fit window (the first-bottom-return artifact) must not stretch
        the y-axis and compress the real fit curves."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(
            ds, range_var="ground_range_m", bin_width_m=2.0, usable_range_m=40.0,
            r_ref_m="mid_swath", r_fit_min_m="auto",
        )
        fit_result = decay.fit
        # Inject an extreme out-of-window observed value directly into
        # the already-computed fit result -- isolates this function's
        # own windowing logic from fit_range_decay's, which is tested
        # separately.
        fit_result.observed_loss_db[0] = 500.0
        fig, ax = plt.subplots()
        try:
            set_fit_window_limits(ax, decay)
            y_lo, y_hi = ax.get_ylim()
            assert y_hi < 100.0  # nowhere near the injected 500dB outlier
        finally:
            plt.close(fig)

    def test_ax_param_draws_into_given_axes_not_a_new_figure(self):
        """For combining with plot_candidate_fit in one stacked figure
        (fit on top, residuals below, sharing an x-axis)."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)  # matches _make_scaled_spreading_dataset's own ground_range_m construction

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        try:
            returned_fig, returned_ax = plot_range_decay_residuals(decay, ax=ax_bottom)
            assert returned_fig is fig
            assert returned_ax is ax_bottom
            assert len(ax_bottom.get_lines()) > 0
            assert len(ax_top.get_lines()) == 0  # untouched
        finally:
            plt.close(fig)


class TestPlotRangeDecayFit:
    def test_no_fit_shows_placeholder(self):
        fig, ax = plot_range_decay_fit(RangeDecayResult())
        try:
            assert ax.get_title() == "No fit available"
        finally:
            plt.close(fig)

    def test_one_line_per_candidate_into_given_ax(self):
        ds = _make_scaled_spreading_dataset(scale=1.6)
        decay = fit_range_decay(ds, range_var="ground_range_m", bin_width_m=2.0)
        fig, ax = plt.subplots()
        try:
            returned_fig, returned_ax = plot_range_decay_fit(decay, ax=ax)
            assert returned_fig is fig and returned_ax is ax
            labels = [line.get_label() for line in ax.get_lines()]
            for candidate in decay.fit.candidates:
                assert any(label.startswith(candidate.name) for label in labels)
            assert "Stage" not in ax.get_title()
        finally:
            plt.close(fig)
