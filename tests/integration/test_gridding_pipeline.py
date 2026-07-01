"""
Integration tests for the gridding pipeline.

Tests the full workflow: channel → grid definition → resampling → raster.
"""

import pytest
import numpy as np
import xarray as xr
from pyproj import CRS

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.resample import ResamplingMethod, resample_channel_to_grid


def _make_realistic_channel(
    num_pings: int = 200,
    samples_per_ping: int = 512,
    freq_hz: int = 455_000,
) -> xr.Dataset:
    """
    Create a realistic test channel with spatial structure.

    Simulates a survey over a 2x2 degree area with:
    - Port and starboard sides (channel dim) with realistic geometry
    - Bathymetric features (ridges, valleys)
    - Water column gap at nadir

    Matches the production schema (core/schema.py REQUIRED_DIMS): channel=2.
    Port and starboard share identical lat/lon here since these tests only
    check aggregate output, not spatial separation (see
    TestResampleChannelToGrid3DValidation in tests/unit/grids/test_resample.py
    for that).
    """
    pings = np.arange(num_pings)
    channels = np.arange(2)
    samples = np.arange(samples_per_ping)

    # Survey area: 2x2 degrees
    south, west, north, east = 39.0, -72.0, 41.0, -70.0

    # Linear interpolation for pings (along-track)
    lats = np.linspace(south, north, num_pings)

    # Samples span port to starboard
    # Assume slant range increases away from nadir
    # Create realistic lat/lon per sample
    lat_per_sample = np.zeros((num_pings, 2, samples_per_ping))
    lon_per_sample = np.zeros((num_pings, 2, samples_per_ping))

    for i, ping_lat in enumerate(lats):
        # Longitude varies across samples (port to starboard)
        lons = np.linspace(west, east, samples_per_ping)
        lat_per_sample[i, :, :] = ping_lat
        lon_per_sample[i, :, :] = lons

    # Create backscatter with realistic features
    backscatter = np.random.rand(num_pings, 2, samples_per_ping) * 0.5

    # Add a bright ridge feature in the middle
    ridge_center = samples_per_ping // 2
    ridge_width = 20
    for i in range(num_pings):
        ridge_intensity = 0.3 + 0.2 * np.sin(i * np.pi / num_pings)
        backscatter[
            i,
            :,
            ridge_center - ridge_width : ridge_center + ridge_width,
        ] += ridge_intensity

    # Water column gap at nadir (very high slant range)
    backscatter[:, :, :50] = np.nan
    backscatter[:, :, -50:] = np.nan

    # Add some noise spikes
    noise_idx = np.random.choice(
        num_pings * 2 * samples_per_ping, size=100, replace=False
    )
    backscatter.ravel()[noise_idx] = np.random.rand(100)

    ds = xr.Dataset(
        {
            "backscatter_raw": (["ping", "channel", "sample"], backscatter),
            "latitude_per_sample": (["ping", "channel", "sample"], lat_per_sample),
            "longitude_per_sample": (["ping", "channel", "sample"], lon_per_sample),
            "ping": (["ping"], pings),
            "channel": (["channel"], channels),
            "sample": (["sample"], samples),
        },
        attrs={
            "frequency_hz": freq_hz,
            "processing_level": "raw",
        },
    )

    return ds


class TestGriddingPipelineBasic:
    """Test complete gridding workflow."""

    @pytest.fixture
    def channel(self):
        """Create a realistic test channel."""
        return _make_realistic_channel(num_pings=100, samples_per_ping=256)

    @pytest.fixture
    def grid(self):
        """Create a test grid covering the channel survey area."""
        return GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=5000.0,
        )

    def test_end_to_end_resample_mean(self, channel, grid):
        """Full pipeline: channel → grid → raster (mean)."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        # Verify output structure
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)
        assert "n_contributing_samples" in raster.dataset.data_vars
        assert raster.crs == CRS.from_epsg(4326)
        assert raster.channel_name == "455 kHz"

        # Verify coverage
        coverage = raster.data_coverage
        assert 0.1 < coverage < 0.9  # Some cells filled, some empty

    def test_end_to_end_multiple_methods(self, channel, grid):
        """Resample with all methods and compare outputs."""
        rasters = {}
        for method in ResamplingMethod:
            rasters[method.value] = resample_channel_to_grid(
                channel,
                grid_def=grid,
                method=method,
            )

        # All rasters should have same shape
        base_shape = rasters["mean"].backscatter.shape
        for method_name, raster in rasters.items():
            assert raster.backscatter.shape == base_shape, f"{method_name} mismatch"

        # Max should be >= mean (element-wise, where both finite)
        mean_raster = rasters["mean"].backscatter
        max_raster = rasters["max"].backscatter
        finite_both = np.isfinite(mean_raster) & np.isfinite(max_raster)
        if np.any(finite_both):
            assert np.all(max_raster[finite_both] >= mean_raster[finite_both] - 1e-6)

    def test_raster_dataset_integrity(self, channel, grid):
        """Verify xarray dataset is well-formed."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        ds = raster.dataset

        # Check required variables
        assert "backscatter" in ds.data_vars
        assert "n_contributing_samples" in ds.data_vars

        # Check required coordinates
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

        # Check attributes
        assert "crs" in ds.attrs
        assert "resolution_m" in ds.attrs
        assert "resampling_method" in ds.attrs

    def test_n_contributing_samples_consistency(self, channel, grid):
        """n_contributing_samples should match backscatter validity."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        backscatter = raster.backscatter
        n_contrib = raster.dataset["n_contributing_samples"].values

        # Where backscatter is finite, n_contrib > 0
        finite_mask = np.isfinite(backscatter)
        assert np.all(n_contrib[finite_mask] > 0)

        # Where backscatter is NaN, n_contrib == 0
        nan_mask = np.isnan(backscatter)
        assert np.all(n_contrib[nan_mask] == 0)


class TestGriddingPipelineCoarseVsFine:
    """Test gridding at different resolutions."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_realistic_channel(num_pings=100, samples_per_ping=256)

    def test_coarse_grid_high_coverage(self, channel):
        """Coarse grid should have high coverage."""
        coarse_grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=50_000.0,  # 50 km
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=coarse_grid,
            method=ResamplingMethod.MEAN,
        )

        coverage = raster.data_coverage
        assert coverage > 0.8  # Mostly filled

    def test_fine_grid_lower_coverage(self, channel):
        """Fine grid should have lower coverage (interpolation gaps)."""
        fine_grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=1000.0,  # 1 km
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=fine_grid,
            method=ResamplingMethod.MEAN,
        )

        coverage = raster.data_coverage
        assert coverage < 0.8  # Sparse

    def test_coarse_more_cells_than_samples(self, channel):
        """Coarse grid cells average many samples."""
        coarse_grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=50_000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=coarse_grid,
            method=ResamplingMethod.MEAN,
        )

        n_contrib = raster.dataset["n_contributing_samples"].values
        filled_cells = n_contrib[n_contrib > 0]
        assert np.mean(filled_cells) > 100  # Average many samples


class TestGriddingPipelineQuality:
    """Test output quality and sanity."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_realistic_channel(num_pings=100, samples_per_ping=256)

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=5000.0,
        )

    def test_gridded_data_preserves_range(self, channel, grid):
        """Gridded values within input data range."""
        backscatter_raw = channel["backscatter_raw"].values
        min_raw = np.nanmin(backscatter_raw)
        max_raw = np.nanmax(backscatter_raw)

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        backscatter_grid = raster.backscatter
        finite_mask = np.isfinite(backscatter_grid)
        
        if np.any(finite_mask):
            min_grid = np.min(backscatter_grid[finite_mask])
            max_grid = np.max(backscatter_grid[finite_mask])
            assert min_grid >= min_raw - 1e-6
            assert max_grid <= max_raw + 1e-6

    def test_bright_feature_visible_in_grid(self, channel, grid):
        """Bright ridge feature from channel should appear in grid."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MAX,
        )

        backscatter_grid = raster.backscatter
        
        # Max aggregation should produce some high values
        finite_vals = backscatter_grid[np.isfinite(backscatter_grid)]
        if len(finite_vals) > 0:
            # Check that peak values are present
            assert np.max(finite_vals) > 0.5  # Ridge was added to base

    def test_different_methods_produce_consistent_patterns(self, channel, grid):
        """Different aggregation methods should show similar spatial patterns."""
        raster_mean = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )
        
        raster_median = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEDIAN,
        )
        
        # Correlation between mean and median should be high
        mean_vals = raster_mean.backscatter[np.isfinite(raster_mean.backscatter)]
        median_vals = raster_median.backscatter[np.isfinite(raster_median.backscatter)]
        
        if len(mean_vals) > 10 and len(median_vals) > 10:
            # Get overlapping cells
            finite_both = (
                np.isfinite(raster_mean.backscatter)
                & np.isfinite(raster_median.backscatter)
            )
            if np.sum(finite_both) > 10:
                corr = np.corrcoef(
                    raster_mean.backscatter[finite_both],
                    raster_median.backscatter[finite_both],
                )[0, 1]
                assert corr > 0.5  # Positive correlation


class TestGriddingPipelineEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_single_ping_channel(self):
        """Resample a channel with only 1 ping."""
        channel = _make_realistic_channel(num_pings=1, samples_per_ping=256)
        grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=50_000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster.backscatter, np.ndarray)

    def test_narrow_channel(self):
        """Resample a channel with very few samples."""
        channel = _make_realistic_channel(num_pings=50, samples_per_ping=32)
        grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=10_000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert np.any(np.isfinite(raster.backscatter))

    def test_grid_partially_outside_channel(self):
        """Grid partially outside survey area."""
        channel = _make_realistic_channel(num_pings=100, samples_per_ping=256)
        
        # Grid extends beyond channel bounds
        grid = GridDefinition(
            bbox=(38.0, -73.0, 42.0, -69.0),  # Larger than channel
            resolution_m=10_000.0,
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        # Should have some NaN cells (outside coverage)
        assert np.any(np.isnan(raster.backscatter))
        # But also some data cells
        assert np.any(np.isfinite(raster.backscatter))


class TestGriddingPipelineReprojection:
    """Test gridding with different CRS."""

    @pytest.fixture
    def channel(self):
        """Create a test channel."""
        return _make_realistic_channel(num_pings=100, samples_per_ping=256)

    def test_resample_to_wgs84(self, channel):
        """Resample to WGS84 (default)."""
        grid = GridDefinition(
            bbox=(39.0, -72.0, 41.0, -70.0),
            resolution_m=5000.0,
            crs=CRS.from_epsg(4326),
        )

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.crs == CRS.from_epsg(4326)
        assert np.any(np.isfinite(raster.backscatter))

    def test_resample_to_utm(self, channel):
        """Resample to UTM (requires channel coordinate conversion)."""
        # For now, channels are always in WGS84, so direct UTM resampling
        # would need upstream coordinate transformation
        pytest.skip("Channel UTM conversion not yet implemented")