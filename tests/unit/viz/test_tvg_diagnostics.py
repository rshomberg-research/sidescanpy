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

from sidescanpy.analysis.tvg_diagnostics import fit_hardware_tvg_curve
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db
from sidescanpy.viz.tvg_diagnostics import (
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
    return fit_hardware_tvg_curve(binned, absorption_db_per_m=0.15)


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
        # 1 observed line + 1 reference-range vline + 4 candidate lines,
        # not 4 observed + 4 vlines + 4 candidate.
        assert len(ax.get_lines()) == 6
        plt.close(fig)
