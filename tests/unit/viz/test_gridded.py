"""
Unit tests for gridded raster visualization.

Tests matplotlib plotting functions (verify they don't crash and return
correct types; visual verification is manual).
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.viz.gridded import (
    plot_gridded_raster,
    plot_gridded_with_coverage,
    plot_gridded_multiple_methods,
    export_gridded_png,
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


class TestPlotGriddedWithCoverage:
    """Test plot_gridded_with_coverage()."""

    def test_basic_plot_with_coverage(self, sample_raster):
        """Create a plot with coverage transparency."""
        fig, ax = plot_gridded_with_coverage(sample_raster)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_coverage_title_includes_percentage(self, sample_raster):
        """Title shows data coverage."""
        fig, ax = plot_gridded_with_coverage(sample_raster)

        title = ax.get_title()
        assert "cov:" in title or "coverage" in title
        plt.close(fig)

    def test_coverage_with_sparse_data(self):
        """Coverage plot with sparse data (many NaN)."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )
        backscatter = np.full((grid.num_cells_ns, grid.num_cells_ew), np.nan)
        # Fill only 10% of cells
        n_fill = (grid.num_cells_ns * grid.num_cells_ew) // 10
        idx = np.random.choice(
            grid.num_cells_ns * grid.num_cells_ew, n_fill, replace=False
        )
        backscatter.ravel()[idx] = np.random.rand(n_fill)

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        fig, ax = plot_gridded_with_coverage(raster)
        assert fig is not None
        plt.close(fig)


class TestPlotMultipleMethods:
    """Test plot_gridded_multiple_methods()."""

    def test_multi_method_plot(self, sample_raster):
        """Plot multiple resampling methods."""
        rasters = {
            "mean": sample_raster,
            "max": sample_raster,
            "median": sample_raster,
        }

        fig, axes = plot_gridded_multiple_methods(rasters)

        assert fig is not None
        assert axes.shape == (1, 3)
        plt.close(fig)

    def test_multi_method_many_subplots(self, sample_raster):
        """Many resampling methods (>3)."""
        rasters = {
            f"method_{i}": sample_raster for i in range(6)
        }

        fig, axes = plot_gridded_multiple_methods(rasters)

        assert fig is not None
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_multi_method_shared_range(self, sample_raster):
        """Shared colorbar range across methods."""
        rasters = {
            "mean": sample_raster,
            "max": sample_raster,
        }

        fig, axes = plot_gridded_multiple_methods(
            rasters,
            vmin=0.2,
            vmax=0.8,
        )

        assert fig is not None
        plt.close(fig)


class TestExportGriddedPng:
    """Test export_gridded_png()."""

    def test_export_png(self, sample_raster, tmp_path):
        """Export raster to PNG."""
        output_path = tmp_path / "test_raster.png"

        export_gridded_png(sample_raster, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_png_custom_dpi(self, sample_raster, tmp_path):
        """Export with custom DPI."""
        output_path = tmp_path / "test_raster_dpi.png"

        export_gridded_png(sample_raster, str(output_path), dpi=300)

        assert output_path.exists()

    def test_export_png_custom_cmap(self, sample_raster, tmp_path):
        """Export with custom colormap."""
        output_path = tmp_path / "test_raster_cmap.png"

        export_gridded_png(
            sample_raster,
            str(output_path),
            cmap="hot",
        )

        assert output_path.exists()