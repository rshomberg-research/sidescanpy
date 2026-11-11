"""
Unit tests for gridded raster visualization.

Tests matplotlib plotting functions (verify they don't crash and return
correct types; visual verification is manual).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import PowerNorm
from pyproj import CRS

from sidescanpy.core.units import mask_floor_clip
from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.viz.gridded import (
    _axis_labels,
    _crs_label,
    plot_gridded_raster,
)


@pytest.fixture
def sample_raster():
    """Create a test raster with n_contributing_samples."""
    grid = GridDefinition(
        bbox=(39.0, -71.0, 40.0, -70.0),
        resolution_m=5000.0,
    )
    backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

    raster = SideScanGriddedRaster(
        grid_def=grid,
        backscatter=backscatter,
        channel_name="455 kHz",
    )

    # Add n_contributing_samples (as would be done by resample_channel_to_grid)
    n_contrib = np.random.randint(10, 100, size=backscatter.shape)
    n_contrib[np.isnan(backscatter)] = 0
    raster.dataset["n_contributing_samples"] = (["ns", "ew"], n_contrib)
    raster.dataset["n_contributing_samples"].attrs["units"] = "count"

    return raster


@pytest.fixture
def utm_raster():
    """A raster in a projected (UTM) CRS, for axis-label/CRS-info tests."""
    grid = GridDefinition(
        bbox=(4320000.0, 400000.0, 4330000.0, 410000.0),
        resolution_m=1000.0,
        crs=CRS.from_epsg(26919),
    )
    backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
    return SideScanGriddedRaster(
        grid_def=grid, backscatter=backscatter, channel_name="455 kHz",
    )


class TestCrsAwareLabels:
    """Axis labels and CRS info adapt to the raster's CRS."""

    def test_axis_labels_geographic(self, sample_raster):
        xlabel, ylabel = _axis_labels(sample_raster)
        assert "Longitude" in xlabel
        assert "Latitude" in ylabel

    def test_axis_labels_projected(self, utm_raster):
        xlabel, ylabel = _axis_labels(utm_raster)
        assert "Easting" in xlabel and "(m)" in xlabel
        assert "Northing" in ylabel and "(m)" in ylabel

    def test_crs_label_includes_epsg(self, utm_raster):
        label = _crs_label(utm_raster)
        assert "26919" in label

    def test_plot_gridded_raster_uses_easting_northing_for_utm(self, utm_raster):
        fig, ax = plot_gridded_raster(utm_raster)
        try:
            assert "Easting" in ax.get_xlabel()
            assert "Northing" in ax.get_ylabel()
            assert "26919" in ax.get_title()
        finally:
            plt.close(fig)

    def test_plot_gridded_raster_uses_lat_lon_for_geographic(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster, title=None)
        try:
            assert "Longitude" in ax.get_xlabel()
            assert "Latitude" in ax.get_ylabel()
        finally:
            plt.close(fig)


class TestPlotGriddedRasterPercentiles:
    """vmin_pct/vmax_pct default to plot_waterfall's 1/99, not a
    different hardcoded value — keeping waterfall and gridded exports
    of the same data comparably normalized (see docstring in
    plot_gridded_raster for why a seemingly small percentile
    difference can matter a lot)."""

    def test_default_percentiles_match_1_99(self, sample_raster):
        finite = sample_raster.backscatter[np.isfinite(sample_raster.backscatter)]
        expected_vmin = np.percentile(finite, 1.0)
        expected_vmax = np.percentile(finite, 99.0)

        fig, ax = plot_gridded_raster(sample_raster)
        try:
            im = ax.collections[0] if ax.collections else ax.images[0]
            assert im.norm.vmin == pytest.approx(expected_vmin)
            assert im.norm.vmax == pytest.approx(expected_vmax)
        finally:
            plt.close(fig)

    def test_custom_percentiles_respected(self, sample_raster):
        finite = sample_raster.backscatter[np.isfinite(sample_raster.backscatter)]
        expected_vmin = np.percentile(finite, 5.0)
        expected_vmax = np.percentile(finite, 95.0)

        fig, ax = plot_gridded_raster(sample_raster, vmin_pct=5.0, vmax_pct=95.0)
        try:
            im = ax.collections[0] if ax.collections else ax.images[0]
            assert im.norm.vmin == pytest.approx(expected_vmin)
            assert im.norm.vmax == pytest.approx(expected_vmax)
        finally:
            plt.close(fig)

    def test_explicit_vmin_vmax_overrides_percentiles(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster, vmin=0.1, vmax=0.9)
        try:
            im = ax.collections[0] if ax.collections else ax.images[0]
            assert im.norm.vmin == pytest.approx(0.1)
            assert im.norm.vmax == pytest.approx(0.9)
        finally:
            plt.close(fig)


class TestMaskFloorClip:
    """Floor-clip artifact values (``core.units.FLOOR_DB``) are
    not real backscatter returns and should be masked to NaN, not treated
    as legitimate low-end data, matching viz/waterfall.py's handling."""

    def test_zero_and_neg120_masked_to_nan(self):
        bs = np.array([0.0, -120.0, 5.3, -40.0, 0.1])
        result = mask_floor_clip(bs)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[[2, 3, 4]], bs[[2, 3, 4]])

    def test_no_floor_values_returns_equivalent_array(self):
        bs = np.array([5.3, -40.0, 0.1])
        result = mask_floor_clip(bs)
        np.testing.assert_array_equal(result, bs)


class TestPlotGriddedRasterExcludesFloorClip:
    def test_vmin_not_pulled_down_by_floor_clip_zeros(self):
        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
        backscatter = np.random.uniform(
            10.0, 20.0, size=(grid.num_cells_ns, grid.num_cells_ew)
        )
        backscatter[: grid.num_cells_ns // 2, :] = 0.0
        raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter)

        fig, ax = plot_gridded_raster(raster)
        try:
            im = ax.collections[0] if ax.collections else ax.images[0]
            # If the floor-clip zeros leaked into the percentile calc,
            # vmin would be pulled down near 0 instead of staying within
            # the real [10, 20) population.
            assert im.norm.vmin >= 9.0
        finally:
            plt.close(fig)


class TestPlotGriddedRaster:
    """Test plot_gridded_raster()."""

    def test_basic_plot(self, sample_raster):
        """Create a basic gridded plot."""
        fig, ax = plot_gridded_raster(sample_raster)

        assert fig is not None
        assert ax is not None
        assert len(ax.images) > 0 or len(ax.collections) > 0

    def test_plot_with_custom_cmap(self, sample_raster):
        """Use custom colormap."""
        fig, ax = plot_gridded_raster(
            sample_raster,
            cmap="RdYlBu_r",
        )

        assert fig is not None
        plt.close(fig)

    def test_nan_renders_black(self, sample_raster):
        # 2026-09-28: standardized NaN-cell rendering to black across
        # every sidescanpy output (user request) -- matplotlib's own
        # default is transparent, which shows through to white in most
        # exports.
        fig, ax = plot_gridded_raster(sample_raster)
        mesh = ax.collections[-1]
        np.testing.assert_array_equal(mesh.get_cmap().get_bad(), (0.0, 0.0, 0.0, 1.0))
        plt.close(fig)

    def test_plot_with_vmin_vmax(self, sample_raster):
        """Specify colorbar range."""
        fig, ax = plot_gridded_raster(
            sample_raster,
            vmin=0.2,
            vmax=0.8,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_title(self, sample_raster):
        """Use custom title."""
        fig, ax = plot_gridded_raster(
            sample_raster,
            title="Test gridded raster",
        )

        assert ax.get_title() == "Test gridded raster"
        plt.close(fig)

    def test_plot_no_colorbar(self, sample_raster):
        """Omit colorbar."""
        fig, ax = plot_gridded_raster(
            sample_raster,
            show_colorbar=False,
        )

        assert fig is not None
        plt.close(fig)

    def test_colorbar_label_reflects_raster_units(self):
        """
        Colorbar label shows the raster's actual units, not a hardcoded
        claim. Regression test: plot_gridded_raster() used to hardcode
        "Backscatter (linear)" regardless of what was gridded, silently
        mislabeling TVG-corrected (dB) data.
        """
        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

        linear_raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, units="linear")
        fig, ax = plot_gridded_raster(linear_raster)
        cbar_label = ax.collections[0].colorbar.ax.get_ylabel()
        assert "linear" in cbar_label
        plt.close(fig)

        db_raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, units="dB")
        fig, ax = plot_gridded_raster(db_raster)
        cbar_label = ax.collections[0].colorbar.ax.get_ylabel()
        assert "dB" in cbar_label
        assert "linear" not in cbar_label
        plt.close(fig)

    def test_plot_existing_axes(self, sample_raster):
        """Plot on existing axes."""
        fig, ax = plt.subplots()
        fig_ret, ax_ret = plot_gridded_raster(sample_raster, fig=fig, ax=ax)

        assert ax_ret is ax
        plt.close(fig)

    def test_scale_bar_shown_by_default_for_projected_crs(self, utm_raster):
        fig, ax = plot_gridded_raster(utm_raster)
        assert len(ax.texts) >= 1
        plt.close(fig)

    def test_scale_bar_can_be_disabled(self, utm_raster):
        fig, ax = plot_gridded_raster(utm_raster, show_scale_bar=False)
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_scale_bar_skipped_for_geographic_crs(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster)
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_plot_with_nan_data(self):
        """Plot raster with partial NaN."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        backscatter[:10, :] = np.nan  # Mask top rows

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        fig, ax = plot_gridded_raster(raster)
        assert fig is not None
        plt.close(fig)


class TestPlotGriddedRasterGamma:
    """gamma is a display-only PowerNorm applied to the color mapping —
    the plotted array and colorbar stay in the raster's real units,
    never renormalized to [0, 1]. gamma=1.0 must remain byte-for-byte
    identical to pre-gamma behavior (plain vmin/vmax, no norm object)."""

    def test_default_gamma_is_linear_not_powernorm(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster)
        mesh = ax.collections[0]
        assert not isinstance(mesh.norm, PowerNorm)
        plt.close(fig)

    def test_gamma_applies_powernorm_with_matching_bounds(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster, gamma=2.0)
        mesh = ax.collections[0]
        assert isinstance(mesh.norm, PowerNorm)
        assert mesh.norm.gamma == 2.0
        vmin, vmax = mesh.get_clim()
        assert vmin < vmax
        plt.close(fig)

    def test_gamma_does_not_change_underlying_plotted_values(self, sample_raster):
        fig1, ax1 = plot_gridded_raster(sample_raster)
        fig2, ax2 = plot_gridded_raster(sample_raster, gamma=3.0)
        try:
            np.testing.assert_array_equal(
                ax1.collections[0].get_array().data, ax2.collections[0].get_array().data
            )
        finally:
            plt.close(fig1)
            plt.close(fig2)

    def test_colorbar_label_stays_real_units_with_gamma(self):
        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        db_raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, units="dB")

        fig, ax = plot_gridded_raster(db_raster, gamma=2.5)
        cbar_label = ax.collections[0].colorbar.ax.get_ylabel()
        assert "dB" in cbar_label
        plt.close(fig)


class TestBasemapIntegration:
    """show_land/show_locator -- see tests/unit/viz/test_basemap.py for
    the basemap module itself; these confirm the wiring into
    plot_gridded_raster specifically."""

    cartopy = pytest.importorskip("cartopy", reason="requires the optional 'maps' extra")

    def test_show_land_draws_a_background_collection(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster, show_land=True)
        assert len(ax.collections) >= 1  # land patch collection + pcolormesh
        plt.close(fig)

    def test_show_land_works_for_projected_crs_too(self, utm_raster):
        fig, ax = plot_gridded_raster(utm_raster, show_land=True)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_show_locator_adds_an_inset_axes(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster, show_locator=True)
        # inset_axes() registers as a child of ax, not a top-level fig.axes entry
        assert len(ax.child_axes) >= 1
        plt.close(fig)


    def test_default_false_adds_no_inset_and_no_land_collection(self, sample_raster):
        fig, ax = plot_gridded_raster(sample_raster)
        assert len(ax.child_axes) == 0
        plt.close(fig)


class TestPlotGridCoverage:
    """plot_grid_coverage: backscatter color, opacity from samples per cell."""

    def test_title_reports_coverage(self, sample_raster):
        from sidescanpy.viz import plot_grid_coverage

        fig, ax = plot_grid_coverage(sample_raster)
        assert "cells with data" in ax.get_title()
        plt.close(fig)

    def test_alpha_follows_sample_count(self, sample_raster):
        from sidescanpy.viz import plot_grid_coverage

        counts = sample_raster.dataset["n_contributing_samples"].values
        fig, ax = plot_grid_coverage(sample_raster, full_opacity_count=counts.max())
        rgba = ax.images[0].get_array()
        ys = sample_raster.y_values
        if ys.ndim == 2:
            ys = ys[:, 0]
        if ys[0] > ys[-1]:
            counts = counts[::-1]
        np.testing.assert_allclose(rgba[..., 3], counts / counts.max())
        plt.close(fig)

    def test_default_full_opacity_is_95th_percentile(self, sample_raster):
        from sidescanpy.viz import plot_grid_coverage

        counts = sample_raster.dataset["n_contributing_samples"].values
        fig, ax = plot_grid_coverage(sample_raster)
        alpha = ax.images[0].get_array()[..., 3]
        assert alpha.max() == 1.0
        np.testing.assert_allclose(np.mean(alpha == 1.0), np.mean(counts >= np.percentile(counts, 95)), atol=0.02)
        plt.close(fig)

    def test_raster_without_counts_is_opaque_where_finite(self):
        from sidescanpy.viz import plot_grid_coverage

        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
        backscatter = np.full((grid.num_cells_ns, grid.num_cells_ew), np.nan)
        backscatter.ravel()[::10] = 1.0
        fig, ax = plot_grid_coverage(SideScanGriddedRaster(grid_def=grid, backscatter=backscatter))
        alpha = ax.images[0].get_array()[..., 3]
        assert set(np.unique(alpha)) == {0.0, 1.0}
        plt.close(fig)

    def test_draws_on_given_axes_with_crs_labels(self, utm_raster):
        from sidescanpy.viz import plot_grid_coverage

        fig, ax = plt.subplots()
        out_fig, out_ax = plot_grid_coverage(utm_raster, fig=fig, ax=ax, title="t")
        assert out_ax is ax and out_fig is fig
        assert ax.get_xlabel() == "Easting (m)" and ax.get_title() == "t"
        plt.close(fig)
