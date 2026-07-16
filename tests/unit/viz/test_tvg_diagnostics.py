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

from sidescanpy.viz.tvg_diagnostics import (
    plot_altitude_bin_comparison,
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
