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

from sidescanpy.viz.waterfall import (
    _compute_y_full,
    _mask_floor_clip,
    plot_survey_overview,
    plot_waterfall,
    resolve_waterfall_axes,
)
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


class TestResolveWaterfallAxes:
    def test_geometric_step_forces_distance_axes_regardless_of_cli(self):
        assert resolve_waterfall_axes(frozenset({"geometric"}), "sample", "ping") == (
            "across_track", "along_track",
        )

    def test_geometric_plus_other_steps_still_forces_distance_axes(self):
        assert resolve_waterfall_axes(
            frozenset({"tvg", "geometric", "georef"}), "sample", "ping"
        ) == ("across_track", "along_track")

    def test_no_geometric_step_passes_cli_axes_through(self):
        assert resolve_waterfall_axes(frozenset({"tvg"}), "ground_range", "time") == (
            "ground_range", "time",
        )

    def test_empty_steps_passes_cli_axes_through(self):
        assert resolve_waterfall_axes(frozenset(), "sample", "ping") == ("sample", "ping")


class TestPlotSurveyOverview:
    def test_rejects_nonpositive_stride(self, sample_dataset):
        with pytest.raises(ValueError, match="ping_stride"):
            plot_survey_overview(sample_dataset, ping_stride=0)

    def test_strides_ping_axis(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, ping_stride=20, ax=ax)
            # mesh built from strided data: ceil(1000/20) = 50 rows
            mesh = ax.collections[-1]
            n_rows = mesh.get_array().shape[0]
            assert n_rows == 50
        finally:
            plt.close(fig)

    def test_default_stride_is_twenty(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, ax=ax)
            mesh = ax.collections[-1]
            assert mesh.get_array().shape[0] == 50
        finally:
            plt.close(fig)

    def test_yaxis_ticks_report_true_ping_numbers_not_local_index(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, ping_stride=20, ax=ax)
            # local mesh row 10 corresponds to true ping 10*20=200
            label = ax.yaxis.get_major_formatter()(10)
            assert label == "200"
        finally:
            plt.close(fig)

    def test_no_break_markers_by_default(self):
        # plot_waterfall itself always draws one port|starboard divider
        # axvline for x_axis="sample" — that's the baseline to compare
        # against, not zero.
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, ping_stride=20, ax=ax)
            assert len(ax.lines) == 1
        finally:
            plt.close(fig)

    def test_break_indices_draw_one_line_each(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(
                ds, ping_stride=20, break_ping_indices=[300, 700], ax=ax,
            )
            # 1 baseline divider (from plot_waterfall) + 2 break markers
            assert len(ax.lines) == 3
        finally:
            plt.close(fig)

    def test_break_line_placed_at_stride_adjusted_position(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(
                ds, ping_stride=20, break_ping_indices=[300], ax=ax,
            )
            # last line added is the break marker (added after the
            # baseline divider inside plot_waterfall)
            line = ax.lines[-1]
            # axhline at true ping 300 -> local y = 300/20 = 15
            y_data = line.get_ydata()
            assert y_data[0] == pytest.approx(15.0)
        finally:
            plt.close(fig)

    def test_title_reports_total_pings_and_stride(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, ping_stride=20, ax=ax)
            title = ax.get_title()
            assert "1000" in title
            assert "1:20" in title
        finally:
            plt.close(fig)

    def test_custom_title_overrides_default(self):
        ds = make_minimal_sidescan_dataset(n_pings=1000)
        fig, ax = plt.subplots()
        try:
            plot_survey_overview(ds, title="My Survey", ax=ax)
            assert ax.get_title() == "My Survey"
        finally:
            plt.close(fig)


class TestPlotWaterfallSlantRangeAxis:
    """x_axis='slant_range' -- always finite (unlike ground_range, which
    is NaN in the water column), signed by channel matching
    ground_range's own port-negative/starboard-positive convention."""

    def _ds_with_slant_range(self):
        from sidescanpy.processing.geometric import compute_geometric_ranges

        ds = make_minimal_sidescan_dataset()
        return compute_geometric_ranges(ds)

    def test_missing_slant_range_m_raises(self):
        ds = make_minimal_sidescan_dataset()
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="slant_range"):
                plot_waterfall(ds, x_axis="slant_range", ax=ax)
        finally:
            plt.close(fig)

    def test_renders_without_error(self):
        ds = self._ds_with_slant_range()
        fig, ax = plt.subplots()
        try:
            plot_waterfall(ds, x_axis="slant_range", ax=ax)
            assert "Slant range" in ax.get_xlabel()
        finally:
            plt.close(fig)

    def test_port_and_starboard_offsets_are_oppositely_signed(self):
        """Same construction as _compute_x_values' ground_range branch --
        port's plotted x-range must be entirely <= 0, starboard's
        entirely >= 0, mirroring the convention slant_range_m itself
        does not carry (it's always a positive magnitude)."""
        from sidescanpy.viz.waterfall import _compute_x_values

        ds = self._ds_with_slant_range()
        x_port, x_stbd = _compute_x_values(ds, "slant_range")
        assert np.all(x_port <= 0)
        assert np.all(x_stbd >= 0)
        # magnitudes match the underlying (always-positive) slant_range_m
        sr = ds["slant_range_m"].values
        np.testing.assert_allclose(-x_port, sr[:, 0, :])
        np.testing.assert_allclose(x_stbd, sr[:, 1, :])
