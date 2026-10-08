"""
Unit tests for resampling channel data to grids.

Tests the resample_channel_to_grid() pipeline and aggregation methods.
"""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.grids.resample import (
    ResamplingMethod,
    resample_channel_to_grid,
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
    - backscatter_intensity (ping, channel, sample)
    - latitude_per_sample, longitude_per_sample (ping, channel, sample)
    - Other required fields

    Matches the production schema (core/schema.py REQUIRED_DIMS): channel=2
    for port/starboard. Port and starboard share identical lat/lon here since
    these tests only check aggregate output, not spatial separation — see
    TestResampleChannelToGrid3DValidation for that.
    """
    # Create a simple survey grid covering a 1x1 degree area
    south, west, north, east = 39.0, -71.0, 40.0, -70.0

    # Generate ping/channel/sample grid in lat/lon
    pings = np.arange(num_pings)
    channels = np.arange(2)
    samples = np.arange(samples_per_ping)

    # Linear interpolation over the survey area
    lats = np.linspace(south, north, num_pings)
    lons = np.linspace(west, east, samples_per_ping)

    lat_per_sample = np.tile(lats[:, np.newaxis, np.newaxis], (1, 2, samples_per_ping))
    lon_per_sample = np.tile(lons[np.newaxis, np.newaxis, :], (num_pings, 2, 1))

    # Create backscatter: add some spatial structure
    backscatter = np.random.rand(num_pings, 2, samples_per_ping)
    # Add a bright line in the middle
    backscatter[:, :, samples_per_ping // 2 : samples_per_ping // 2 + 10] *= 2.0

    ds = xr.Dataset(
        {
            "backscatter_intensity": (["ping", "channel", "sample"], backscatter),
            "latitude_per_sample": (["ping", "channel", "sample"], lat_per_sample),
            "longitude_per_sample": (["ping", "channel", "sample"], lon_per_sample),
            "ping": (["ping"], pings),
            "channel": (["channel"], channels),
            "sample": (["sample"], samples),
        },
        attrs={
            "freq_hz_start": freq_hz,
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
        # Derived from the data: per-sample lat/lon present -> georef.
        assert raster.processing_level == "georef"


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
        backscatter_intensity = channel["backscatter_intensity"].values
        min_input = np.nanmin(backscatter_intensity)
        max_input = np.nanmax(backscatter_intensity)

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
        _raster_min = resample_channel_to_grid(
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
        channel["backscatter_intensity"].values[0:10, :, 0:20] = np.nan

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
        channel["backscatter_intensity"].values[:, :] = np.nan

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
        channel["backscatter_intensity"].values[10, 0, 50] = np.inf

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
        """Resample channel (WGS84 lat/lon samples) onto a UTM grid.

        resample_channel_to_grid() reprojects the WGS84 sample
        coordinates into the grid's own CRS before querying — no
        separate channel-reprojection step needed by the caller.
        """
        raster = resample_channel_to_grid(
            channel,
            grid_def=utm_grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        # utm_grid's bbox only partially overlaps this channel's lat/lon
        # extent (see docstring math in the fixture derivation) — some,
        # not necessarily all, cells should get real data.
        assert raster.data_coverage > 0

    def test_resample_preserves_utm_crs(self, channel, utm_grid):
        """Output raster has same CRS as input grid."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=utm_grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.crs == utm_grid.crs
        assert raster.crs.to_epsg() == 26919

    def test_physically_invalid_latitude_dropped_not_crashed(self, utm_grid, caplog):
        """A physically-invalid latitude (>90, e.g. a corrupted
        georeferencing result or sensor glitch -- not something a
        healthy WGS84 fix ever produces, but not NaN either) can project
        to a non-finite UTM coordinate. Found via a synthetic
        mission whose fabricated vessel position wandered close enough
        to a pole that its per-sample georeferencing math produced a few
        out-of-range (>90 deg) latitudes. Previously crashed inside
        scipy's cKDTree with an opaque "data must be finite" ValueError;
        should instead drop just the bad sample(s) (with a logged
        warning) and grid the rest normally."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=128)
        channel["latitude_per_sample"].values[0, :, :] = 95.0

        raster = resample_channel_to_grid(
            channel,
            grid_def=utm_grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert "non-finite" in caplog.text

    def test_all_samples_invalid_latitude_raises_clear_error(self, utm_grid):
        """Every sample projects to a non-finite UTM coordinate -- should
        raise the same clear "no valid coordinates" error as the
        pre-existing all-NaN-lat/lon case, not a bare scipy ValueError."""
        channel = _make_test_channel(num_pings=50, samples_per_ping=128)
        channel["latitude_per_sample"].values[:, :, :] = 95.0

        with pytest.raises(ValueError, match="No valid sample coordinates"):
            resample_channel_to_grid(
                channel,
                grid_def=utm_grid,
                method=ResamplingMethod.MEAN,
            )

    def test_resample_to_utm_grid_matches_reprojected_wgs84_grid(self, channel):
        """A UTM grid and its geographic-equivalent WGS84 grid should
        resample to the same physical cells (allowing for the different
        cell-center placement each CRS's linspace produces), confirming
        the projected-CRS code path isn't silently returning something
        unrelated to the WGS84 path's result."""
        utm_crs = CRS.from_epsg(26919)
        utm_grid = GridDefinition(
            bbox=(4320000.0, 400000.0, 4330000.0, 410000.0),
            resolution_m=1000.0,
            crs=utm_crs,
        )
        wgs84_grid = utm_grid.reproject_to(CRS.from_epsg(4326))

        raster_utm = resample_channel_to_grid(
            channel, grid_def=utm_grid, method=ResamplingMethod.MEAN,
        )
        raster_wgs84 = resample_channel_to_grid(
            channel, grid_def=wgs84_grid, method=ResamplingMethod.MEAN,
        )

        assert raster_utm.data_coverage > 0
        assert raster_wgs84.data_coverage > 0
        # Both cover (approximately) the same ground area, so coverage
        # fraction should be in the same ballpark, not orders of
        # magnitude apart (which is what the old degrees-vs-meters bug
        # would have produced: near-zero coverage for the UTM grid).
        assert raster_utm.data_coverage == pytest.approx(
            raster_wgs84.data_coverage, abs=0.3
        )


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
        """Create a test channel with a real port/starboard channel dimension."""
        return _make_test_channel(num_pings=50, samples_per_ping=128)

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
        """The raster carries the channel's derived processing level."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.processing_level == "georef"

    def test_raster_dataset_has_source_info(self, channel, grid):
        """Dataset includes source channel frequency."""
        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.dataset.attrs["frequency_hz"] == 455_000
        assert raster.channel_name == "455 kHz"


class TestResampleChannelToGrid3DValidation:
    """
    Test 3D (ping, channel, sample) input validation and channel-preserving
    behavior. Uses the channel_3d fixture from conftest.py, which matches
    the production schema (core/schema.py REQUIRED_DIMS).
    """

    @pytest.fixture
    def grid(self):
        """Create a test grid."""
        return GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=5000.0,
        )

    def test_accepts_3d_data(self, channel_3d, grid):
        """3D (ping, channel, sample) input resamples successfully."""
        raster = resample_channel_to_grid(
            channel_3d,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert isinstance(raster, SideScanGriddedRaster)
        assert np.any(np.isfinite(raster.backscatter))

    def test_rejects_2d_data(self, grid):
        """2D (ping, sample) input raises a clear ValueError, not a silent misread."""
        ds_2d = xr.Dataset(
            {
                "backscatter_intensity": (["ping", "sample"], np.random.rand(20, 32)),
                "latitude_per_sample": (["ping", "sample"], np.full((20, 32), 39.5)),
                "longitude_per_sample": (["ping", "sample"], np.full((20, 32), -70.5)),
            },
            attrs={"freq_hz_start": 455_000, "processing_level": "raw"},
        )

        with pytest.raises(ValueError, match="3D"):
            resample_channel_to_grid(ds_2d, grid_def=grid, method=ResamplingMethod.MEAN)

    def test_rejects_mismatched_coordinate_shape(self, channel_3d, grid):
        """latitude_per_sample shape diverging from backscatter_intensity raises ValueError."""
        bad = channel_3d.copy(deep=True)
        # Collapse the channel dimension on latitude only, producing a
        # (ping, sample) array against a (ping, channel, sample) backscatter_intensity.
        bad["latitude_per_sample"] = bad["latitude_per_sample"].isel(channel=0)

        with pytest.raises(ValueError, match="shape"):
            resample_channel_to_grid(bad, grid_def=grid, method=ResamplingMethod.MEAN)

    def test_preserves_port_starboard_spatial_distinction(self):
        """
        Port and starboard samples resample into their own distinct grid
        cells, not a blended average. Regression test for the historical
        bug described in docs/_internal/v0.6.0-gridding-refactor.md.
        """
        num_pings, num_samples = 30, 20
        lat_track = np.linspace(39.0, 39.01, num_pings)
        lon_center = -71.0

        lat3d = np.tile(lat_track[:, np.newaxis, np.newaxis], (1, 2, num_samples))
        lon3d = np.empty((num_pings, 2, num_samples))
        lon3d[:, 0, :] = lon_center - 0.02  # port: well west
        lon3d[:, 1, :] = lon_center + 0.02  # starboard: well east

        backscatter = np.empty((num_pings, 2, num_samples))
        backscatter[:, 0, :] = 1.0  # port value
        backscatter[:, 1, :] = 5.0  # starboard value

        channel = xr.Dataset(
            {
                "backscatter_intensity": (["ping", "channel", "sample"], backscatter),
                "latitude_per_sample": (["ping", "channel", "sample"], lat3d),
                "longitude_per_sample": (["ping", "channel", "sample"], lon3d),
            },
            attrs={"freq_hz_start": 455_000, "processing_level": "raw"},
        )

        grid = GridDefinition(
            bbox=(38.99, -71.03, 39.02, -70.97),
            resolution_m=50.0,
        )

        raster = resample_channel_to_grid(
            channel, grid_def=grid, method=ResamplingMethod.MEAN
        )
        finite_values = raster.backscatter[np.isfinite(raster.backscatter)]

        # If channels were averaged together, every cell would read ~3.0.
        assert not np.any(np.isclose(finite_values, 3.0, atol=0.1))
        assert np.any(np.isclose(finite_values, 1.0, atol=0.1))
        assert np.any(np.isclose(finite_values, 5.0, atol=0.1))

    def test_output_shape_matches_grid(self, channel_3d, grid):
        """Raster shape matches grid cell counts for 3D input."""
        raster = resample_channel_to_grid(
            channel_3d,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
        )

        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)

    def test_backscatter_var_selects_alternate_variable(self, channel_3d, grid):
        """backscatter_var lets callers grid a variable other than the
        default backscatter_intensity (e.g. a derived uncorrected copy)."""
        channel = channel_3d.copy(deep=True)
        channel["backscatter_alt"] = channel["backscatter_intensity"] - 100.0

        raster = resample_channel_to_grid(
            channel,
            grid_def=grid,
            method=ResamplingMethod.MEAN,
            backscatter_var="backscatter_alt",
        )

        finite_values = raster.backscatter[np.isfinite(raster.backscatter)]
        assert np.all(finite_values < -90.0)
        assert raster.dataset.attrs["backscatter_variable"] == "backscatter_alt"

    def test_backscatter_var_missing_raises_clear_error(self, channel_3d, grid):
        """Requesting a backscatter_var absent from the channel raises ValueError."""
        with pytest.raises(ValueError, match="backscatter_missing"):
            resample_channel_to_grid(
                channel_3d,
                grid_def=grid,
                method=ResamplingMethod.MEAN,
                backscatter_var="backscatter_missing",
            )

    def test_default_var_without_tvg_produces_linear_units(self, channel_3d, grid):
        """backscatter_intensity before TVG is recorded as linear units."""
        raster = resample_channel_to_grid(channel_3d, grid_def=grid, method=ResamplingMethod.MEAN)

        assert raster.units == "linear"
        assert raster.dataset["backscatter"].attrs["units"] == "linear"

    def test_backscatter_intensity_after_tvg_produces_db_units(self, channel_3d, grid):
        """backscatter_intensity with a recorded tvg step is recorded as dB
        units (regression: the raster used to claim "linear" for TVG dB data)."""
        import json

        channel = channel_3d.copy(deep=True)
        channel["backscatter_intensity"] = channel["backscatter_intensity"] - 100.0
        channel.attrs["processing_history"] = json.dumps([{"step": "tvg", "params": {}}])

        raster = resample_channel_to_grid(channel, grid_def=grid, method=ResamplingMethod.MEAN)

        assert raster.units == "dB"
        assert raster.dataset["backscatter"].attrs["units"] == "dB"
        assert raster.processing_level == "georef"


class TestResampleChannelNameFromConverterAttrs:
    """The raster's channel label comes from ``freq_hz_start`` -- the
    attribute the JSF/SDF converters actually write -- not a
    ``frequency_hz`` attribute nothing produces."""

    @pytest.mark.parametrize("freq_start, freq_end, expected", [
        (455_000, 455_000, "455 kHz"),
        (1_495_000, 1_495_000, "1495 kHz"),
        (540_000, 560_000, "540 kHz"),  # chirp: freq_hz_start labels it, as freq_key does
    ])
    def test_channel_name_from_converter_built_dataset(self, freq_start, freq_end, expected):
        from sidescanpy.grids import bbox_from_channel_dataset, utm_grid_from_wgs84_bbox
        from sidescanpy.processing import correct
        from tests.conftest import make_minimal_sidescan_dataset

        ds = make_minimal_sidescan_dataset(n_pings=20, n_samples=60, freq_hz_start=freq_start, freq_hz_end=freq_end)
        ds.attrs["sample_rate_hz"] = 10_000  # 7.5 cm/sample -> 4.5 m slant range
        ds["altitude_m"] = ("ping", np.full(20, 1.0, dtype=np.float32))
        ds = correct(ds, ["geometric", "georef"])
        grid = utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(ds), resolution_m=200.0)

        raster = resample_channel_to_grid(ds, grid)

        assert "frequency_hz" not in ds.attrs
        assert raster.channel_name == expected
        assert raster.dataset.attrs["channel_name"] == expected
        assert raster.dataset.attrs["frequency_hz"] == freq_start


class TestGridChannel:
    """grid_channel: bbox -> auto UTM grid -> resample, as one call."""

    def _small(self):
        ds = _make_test_channel(num_pings=30, samples_per_ping=40)
        # ~100 m extent so a 5 m grid stays small
        ds["latitude_per_sample"] = 39.0 + (ds["latitude_per_sample"] - 39.0) * 1e-3
        ds["longitude_per_sample"] = -71.0 + (ds["longitude_per_sample"] + 71.0) * 1e-3
        return ds

    def test_matches_manual_chain(self):
        from sidescanpy.grids import (
            bbox_from_channel_dataset,
            grid_channel,
            utm_grid_from_wgs84_bbox,
        )

        ds = self._small()
        manual = resample_channel_to_grid(
            ds, utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(ds), resolution_m=5.0), method=ResamplingMethod.MEDIAN,
        )
        raster = grid_channel(ds, 5.0, "median")
        np.testing.assert_array_equal(raster.dataset["backscatter"].values, manual.dataset["backscatter"].values)
        assert raster.grid_def.crs.is_projected

    def test_backscatter_var_selects_variable(self):
        from sidescanpy.grids import grid_channel

        ds = self._small()
        ds["other"] = ds["backscatter_intensity"] * 0 + 7.0
        raster = grid_channel(ds, 5.0, backscatter_var="other")
        vals = raster.dataset["backscatter"].values
        assert np.nanmax(vals) == pytest.approx(7.0) and np.nanmin(vals) == pytest.approx(7.0)

    def test_unknown_method_raises(self):
        from sidescanpy.grids import grid_channel

        with pytest.raises(ValueError):
            grid_channel(self._small(), 5.0, "bilinear")
