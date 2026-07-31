"""
Unit tests for sidescanpy.viz.waterfall — focused on the ping_offset
parameter added to support rendering an already-extracted segment of a
larger (e.g. multi-file concatenated) survey with correct global
y-axis labels.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import PowerNorm

from sidescanpy.viz.waterfall import _compute_y_full, _mask_floor_clip, plot_waterfall
from tests.conftest import make_minimal_sidescan_dataset


class TestComputeYFullPingOffset:
    def test_default_offset_is_local_arange(self, sample_dataset):
        y = _compute_y_full(sample_dataset, "ping")
        np.testing.assert_array_equal(y, np.arange(sample_dataset.sizes["ping"]))

    def test_nonzero_offset_shifts_ping_values(self, sample_dataset):
        n = sample_dataset.sizes["ping"]
        y = _compute_y_full(sample_dataset, "ping", ping_offset=500)
        np.testing.assert_array_equal(y, np.arange(500, 500 + n))

    def test_offset_ignored_for_time_axis(self, sample_dataset):
        y_default = _compute_y_full(sample_dataset, "time")
        y_offset = _compute_y_full(sample_dataset, "time", ping_offset=500)
        np.testing.assert_array_equal(y_default, y_offset)

    def test_offset_ignored_for_along_track_axis(self, sample_dataset):
        y_default = _compute_y_full(sample_dataset, "along_track")
        y_offset = _compute_y_full(sample_dataset, "along_track", ping_offset=500)
        np.testing.assert_array_equal(y_default, y_offset)


class TestPlotWaterfallPingOffset:
    def test_default_title_shows_local_ping_range(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, y_axis="ping", ax=ax)
            n = sample_dataset.sizes["ping"]
            assert f"pings 0–{n}" in ax.get_title()
        finally:
            plt.close(fig)

    def test_ping_offset_shows_global_ping_range_in_title(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            n = sample_dataset.sizes["ping"]
            plot_waterfall(sample_dataset, y_axis="ping", ping_offset=1000, ax=ax)
            assert f"pings 1000–{1000 + n}" in ax.get_title()
        finally:
            plt.close(fig)

    def test_ping_offset_does_not_error_for_other_y_axes(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, y_axis="along_track", ping_offset=1000, ax=ax)
        finally:
            plt.close(fig)


class TestMaskFloorClip:
    """Floor-clip artifact values (sidescanpy/CLAUDE.md Known Issue #5) are
    not real backscatter returns and should be masked to NaN, not treated
    as legitimate low-end data."""

    def test_zero_and_neg120_masked_to_nan(self):
        bs = np.array([0.0, -120.0, 5.3, -40.0, 0.1])
        result = _mask_floor_clip(bs)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[[2, 3, 4]], bs[[2, 3, 4]])

    def test_no_floor_values_returns_equivalent_array(self):
        bs = np.array([5.3, -40.0, 0.1])
        result = _mask_floor_clip(bs)
        np.testing.assert_array_equal(result, bs)

    def test_all_nan_input_stays_nan(self):
        bs = np.full(4, np.nan)
        result = _mask_floor_clip(bs)
        assert np.all(np.isnan(result))


class TestPlotWaterfallExcludesFloorClipFromColorScale:
    def test_vmin_not_pulled_down_by_floor_clip_zeros(self):
        n_pings, n_samples = 20, 50
        ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples)
        # Real values uniform in [10, 20); floor-clip artifact (exact 0.0)
        # injected into half the samples, mimicking the documented leading
        # per-ping run of exact-zero raw amplitude.
        real_values = np.random.uniform(10.0, 20.0, size=(n_pings, 2, n_samples)).astype(np.float32)
        real_values[:, :, : n_samples // 2] = 0.0
        ds["backscatter_intensity"].values[...] = real_values

        fig, ax = plt.subplots()
        try:
            plot_waterfall(ds, ax=ax)
            vmin, vmax = ax.collections[0].get_clim()
            # If the floor-clip zeros leaked into the percentile calc,
            # vmin would be pulled down near 0 instead of staying within
            # the real [10, 20) population.
            assert vmin >= 9.0
        finally:
            plt.close(fig)


class TestPlotWaterfallGamma:
    """gamma is a display-only PowerNorm applied to the color mapping —
    the plotted array and colorbar stay real dB, never renormalized to
    [0, 1]. gamma=1.0 must remain byte-for-byte identical to pre-gamma
    behavior (plain vmin/vmax, no norm object)."""

    def test_default_gamma_is_linear_vmin_vmax_not_powernorm(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, ax=ax)
            mesh = ax.collections[-1]
            assert not isinstance(mesh.norm, PowerNorm)
        finally:
            plt.close(fig)

    def test_gamma_applies_powernorm_with_matching_bounds(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, gamma=2.0, ax=ax)
            mesh = ax.collections[-1]
            assert isinstance(mesh.norm, PowerNorm)
            assert mesh.norm.gamma == 2.0
            vmin, vmax = mesh.get_clim()
            assert vmin < vmax
        finally:
            plt.close(fig)

    def test_gamma_does_not_change_underlying_plotted_values(self, sample_dataset):
        # The array handed to pcolormesh must stay real dB regardless of
        # gamma — gamma only warps the color mapping (norm), never the data.
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        try:
            plot_waterfall(sample_dataset, ax=ax1)
            plot_waterfall(sample_dataset, gamma=3.0, ax=ax2)
            np.testing.assert_array_equal(
                ax1.collections[-1].get_array(), ax2.collections[-1].get_array()
            )
        finally:
            plt.close(fig1)
            plt.close(fig2)


class TestPlotWaterfallColorbar:
    def test_no_colorbar_by_default(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, ax=ax)
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_show_colorbar_adds_axes_labeled_backscatter_db(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, show_colorbar=True, ax=ax)
            assert len(fig.axes) == 2
            cbar_label = fig.axes[-1].get_ylabel()
            assert cbar_label == "Backscatter (dB)"
        finally:
            plt.close(fig)

    def test_colorbar_label_stays_db_even_with_gamma(self, sample_dataset):
        fig, ax = plt.subplots()
        try:
            plot_waterfall(sample_dataset, gamma=2.5, show_colorbar=True, ax=ax)
            cbar_label = fig.axes[-1].get_ylabel()
            assert cbar_label == "Backscatter (dB)"
        finally:
            plt.close(fig)
