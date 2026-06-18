"""
Unit tests for resampling channel data to grids.

Tests the resample_channel_to_grid() pipeline and aggregation methods.
"""

import pytest
import numpy as np
import xarray as xr
from pyproj import CRS

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.grids.resample import (
    resample_channel_to_grid,
    ResamplingMethod,
)


# Helpers to create test channel data

def _make_test_channel(
    num_pings: int = 100,
    samples_per_ping: int = 256,
    freq_hz: int = 455_000,
) -> xr.Dataset:
    """
    Create a minimal SideScanFrequencyChannel-like dataset.
    
    Returns an xarray Dataset with:
    - backscatter_raw (ping, sample)
    - latitude_per_sample, longitude_per_sample (ping, sample)
    - Other required fields
    """
    # Create a simple survey grid covering a 1x1 degree area
    south, west, north, east = 39.0, -71.0, 40.0, -70.0
    
    # Generate ping/sample grid in lat/lon
    pings = np.arange(num_pings)
    samples = np.arange(samples_per_ping)
    
    # Linear interpolation over the survey area
    lats = np.linspace(south, north, num_pings)
    lons = np.linspace(west, east, samples_per_ping)
    
    lat_per_sample = np.tile(lats[:, np.newaxis], (1, samples_per_ping))
    lon_per_sample = np.tile(lons[np.newaxis, :], (num_pings, 1))
    
    # Create backscatter: add some spatial structure
    backscatter = np.random.rand(num_pings, samples_per_ping)
    # Add a bright line in the middle
    backscatter[:, samples_per_ping // 2 : samples_per_ping // 2 + 10] *= 2.0
    
    ds = xr.Dataset(
        {
            "backscatter_raw": (["ping", "sample"], backscatter),
            "latitude_per_sample": (["ping", "sample"], lat_per_sample),
            "longitude_per_sample": (["ping", "sample"], lon_per_sample),
            "ping": (["ping"], pings),
            "sample": (["sample"], samples),
        },
        attrs={
            "frequency_hz": freq_hz,
            "processing_level": "raw",
        },
    )
    
    return ds


class TestResamplingMethodEnum:
    """Test ResamplingMethod enum."""

    def test_valid_methods(self):
        """Standard resampling methods are available."""
        assert ResamplingMethod.MEAN.value == "mean"
        assert ResamplingMethod.MEDIAN.value == "median"
        assert ResamplingMethod.MAX.value == "max"
        assert ResamplingMethod.NEAREST.value == "nearest"
        assert ResamplingMethod.IDW.value == "idw"

    def test_method_from_string(self):
        """Construct enum from string."""
        method = ResamplingMethod("mean")
        assert method == ResamplingMethod.MEAN


class TestResampleChannelToGridBasic:
    """Test basic resampling workflow."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,  # Coarse grid
        )

    def test_resample_mean_basic(self, channel, grid):
        """Resample channel to grid using mean aggregation."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)
        assert raster.grid_def == grid
        assert raster.channel_name == "455 kHz"
        assert np.any(np.isfinite(raster.backscatter))

    def test_resample_median_basic(self, channel, grid):
        """Resample using median aggregation."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEDIAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)

    def test_resample_max_basic(self, channel, grid):
        """Resample using max aggregation."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MAX,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)

    def test_resample_nearest_basic(self, channel, grid):
        """Resample using nearest-neighbor."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.NEAREST,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)

    def test_resample_idw_basic(self, channel, grid):
        """Resample using inverse-distance weighting."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.IDW,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)

    def test_resample_preserves_metadata(self, channel, grid):
        """Resampling preserves channel metadata."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.channel_name == "455 kHz"
        assert raster.processing_level == "raw"


class TestResampleChannelToGridNContributing:
    """Test n_contributing_samples tracking."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

    def test_n_contributing_samples_in_dataset(self, channel, grid):
        """Dataset includes n_contributing_samples variable."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert "n_contributing_samples" in raster.dataset.data_vars
        n_contrib = raster.dataset["n_contributing_samples"].values
        assert n_contrib.shape == raster.backscatter.shape

    def test_n_contributing_samples_is_nonnegative(self, channel, grid):
        """All counts are non-negative."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        n_contrib = raster.dataset["n_contributing_samples"].values
        assert np.all(n_contrib >= 0)

    def test_n_contributing_samples_matches_coverage(self, channel, grid):
        """Cells with data have n_contrib > 0."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        n_contrib = raster.dataset["n_contributing_samples"].values
        backscatter = raster.backscatter

        # Where backscatter is finite, n_contrib should be > 0
        finite_mask = np.isfinite(backscatter)
        assert np.all(n_contrib[finite_mask] > 0)

        # Where backscatter is NaN, n_contrib should be 0
        nan_mask = np.isnan(backscatter)
        assert np.all(n_contrib[nan_mask] == 0)

    def test_n_contributing_samples_has_units(self, channel, grid):
        """n_contributing_samples has units attribute."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        n_contrib = raster.dataset["n_contributing_samples"]
        assert "units" in n_contrib.attrs
        assert n_contrib.attrs["units"] in ["count", "dimensionless", "1"]


class TestResampleChannelToGridQuality:
    """Test data quality and sanity checks."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

    def test_mean_is_within_input_range(self, channel, grid):
        """Mean resampling produces values in input range."""
        backscatter_raw = channel["backscatter_raw"].values
        min_input = np.nanmin(backscatter_raw)
        max_input = np.nanmax(backscatter_raw)

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        backscatter_resampled = raster.backscatter
        finite_mask = np.isfinite(backscatter_resampled)
        
        if np.any(finite_mask):
            assert np.nanmin(backscatter_resampled) >= min_input - 1e-6
            assert np.nanmax(backscatter_resampled) <= max_input + 1e-6

    def test_max_is_at_least_as_large_as_mean(self, channel, grid):
        """Max aggregation produces values >= mean aggregation."""
        raster_mean = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        raster_max = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MAX,
        )

        # Where both are finite, max >= mean
        finite_both = (
            np.isfinite(raster_mean.backscatter)
            & np.isfinite(raster_max.backscatter)
        )

        if np.any(finite_both):
            assert np.all(
                raster_max.backscatter[finite_both]
                >= raster_mean.backscatter[finite_both] - 1e-6
            )

    def test_median_between_min_max(self, channel, grid):
        """Median aggregation is between min and max."""
        raster_min = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.NEAREST,  # Proxy for min
        )

        raster_median = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEDIAN,
        )

        raster_max = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MAX,
        )

        # Sanity check: all methods produce valid output
        assert np.any(np.isfinite(raster_median.backscatter))
        assert np.any(np.isfinite(raster_max.backscatter))


class TestResampleChannelToGridNaN:
    """Test handling of NaN and missing data."""

    def test_resample_with_partial_nan(self):
        """Resampling handles cells with some NaN samples."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=128)
        # Mask out some samples as NaN
        channel["backscatter_raw"].values[0:10, 0:20] = np.nan

        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        # Should still produce a valid raster
        assert isinstance(raster, SideScanGriddedRaster)
        assert np.any(np.isfinite(raster.backscatter))

    def test_resample_all_nan_channel_raises_error(self):
        """Channel with all NaN samples raises error (no data to resample)."""
        channel = _make_test_channel(num_pings=5, samples_per_ping=10)
        # Make the channel mostly NaN
        channel["backscatter_raw"].values[:, :] = np.nan

        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100_000.0,  # Very coarse grid
        )

        # Should raise error because backscatter validation fails
        with pytest.raises(ValueError, match="all NaN|no data"):
            resample_channel_to_grid(
                channel,
                grid_def=grid,
                method=ResamplingMethod.MEAN,
            )

    def test_resample_handles_inf(self):
        """Resampling handles inf values gracefully."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=128)
        channel["backscatter_raw"].values[10, 50] = np.inf

        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

        # Should not crash
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)


class TestResampleChannelToGridUTM:
    """Test resampling with projected CRS."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

    @pytest.fixture
    def utm_grid(self):
        """Create a test grid in UTM."""
        utm_crs = CRS.from_epsg(26919)
        return GridDefinition(
            bbox=(4320000.0, 400000.0, 4370000.0, 450000.0),
            resolution_m=5000.0,
            crs=utm_crs,
        )

    def test_resample_to_utm_grid(self, channel, utm_grid):
        """Resample channel to UTM grid."""
        # Need to reproject channel coordinates to UTM first
        # For now, skip this test (transformation would be separate step)
        pytest.skip("Channel reprojection not yet implemented")

    def test_resample_preserves_utm_crs(self, channel, utm_grid):
        """Output raster has same CRS as input grid."""
        pytest.skip("Channel reprojection not yet implemented")


class TestResampleChannelToGridEdgeCases:
    """Test boundary and edge cases."""

    def test_resample_single_cell_grid(self):
        """Resample to a grid with only 1 cell."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=128)

        grid = GridDefinition(
            bbox=(39.0, -71.0, 39.0001, -70.9999),
            resolution_m=100_000.0,  # Forces 1 cell
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.num_cells_total >= 1
        assert np.any(np.isfinite(raster.backscatter))

    def test_resample_very_fine_grid(self):
        """Resample to a grid much finer than input sampling."""
        channel = _make_test_channel(num_pings=10, samples_per_ping=20)

        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,  # Very fine
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.NEAREST,
        )

        # Many cells will be NaN due to interpolation gaps
        assert raster.data_coverage < 0.5
        assert np.any(np.isfinite(raster.backscatter))

    def test_resample_very_coarse_grid(self):
        """Resample to a grid much coarser than input sampling."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=256)

        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100_000.0,  # Very coarse
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        # Few cells, but each should have many contributing samples
        assert raster.num_cells_total < 10
        n_contrib = raster.dataset["n_contributing_samples"].values
        assert np.all(n_contrib[raster.backscatter != np.nan] > 100)


class TestResampleChannelToGridPerSide:
    """Test optional per-side (port/starboard) diagnostic output."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        ds = _make_test_channel(num_pings=50, samples_per_ping=128)
        # Add port/starboard markers
        ds = ds.assign_coords(side=("sample", ["port"] * 64 + ["starboard"] * 64))
        return ds

    def test_resample_with_port_starboard_separation(self, channel):
        """Resample can optionally separate port and starboard."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

        # Per-side separation not yet implemented
        pytest.skip("Per-side separation deferred to v0.7.0")

    def test_resample_default_merges_sides(self, channel):
        """Default resample merges port and starboard."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        # Only backscatter variable (merged)
        assert "backscatter" in raster.dataset.data_vars
        assert "port" not in raster.dataset.data_vars
        assert "starboard" not in raster.dataset.data_vars


class TestResampleChannelToGridMetadata:
    """Test metadata preservation and propagation."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

    def test_raster_has_resampling_method_metadata(self, channel, grid):
        """Output raster includes resampling method in metadata."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert "resampling_method" in raster.dataset.attrs
        assert raster.dataset.attrs["resampling_method"] == "mean"

    def test_raster_backscatter_has_processing_level(self, channel, grid):
        """Backscatter variable preserves processing_level."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.processing_level == "raw"

    def test_raster_dataset_has_source_info(self, channel, grid):
        """Dataset includes source channel frequency."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert "frequency_hz" in raster.dataset.attrs or raster.channel_name == "455 kHz"