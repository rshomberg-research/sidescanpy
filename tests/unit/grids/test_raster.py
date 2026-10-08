"""
Unit tests for SideScanGriddedRaster.

Tests the xarray-validated wrapper for gridded backscatter data.
"""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster


class TestSideScanGriddedRasterInstantiation:
    """Test basic construction and validation."""

    def test_minimal_construction_from_grid_and_data(self):
        """Create a gridded raster from GridDefinition + 2D backscatter array."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        # Create dummy backscatter data matching grid cell count
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        assert raster.grid_def == grid
        assert raster.backscatter.shape == (grid.num_cells_ns, grid.num_cells_ew)
        assert isinstance(raster.dataset, xr.Dataset)

    def test_construction_with_metadata(self):
        """Include channel name and processing level."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
            channel_name="455 kHz",
            processing_level="radiometric",
        )

        assert raster.channel_name == "455 kHz"
        assert raster.processing_level == "radiometric"

    def test_construction_with_utm_grid(self):
        """Grid can be in projected CRS (e.g., UTM)."""
        utm_crs = CRS.from_epsg(26919)
        grid = GridDefinition(
            bbox=(400000.0, 4320000.0, 450000.0, 4370000.0),
            resolution_m=5.0,
            crs=utm_crs,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        assert raster.crs == utm_crs


class TestSideScanGriddedRasterValidation:
    """Test schema validation."""

    def test_rejects_mismatched_shape(self):
        """Backscatter shape must match grid cell count."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        # Wrong shape
        backscatter = np.random.rand(10, 20)

        with pytest.raises(ValueError, match="shape|mismatch|cells"):
            SideScanGriddedRaster(
                grid_def=grid,
                backscatter=backscatter,
            )

    def test_rejects_1d_backscatter(self):
        """Backscatter must be 2D."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns * grid.num_cells_ew)

        with pytest.raises(ValueError, match="2D|dimension"):
            SideScanGriddedRaster(
                grid_def=grid,
                backscatter=backscatter,
            )

    def test_rejects_3d_backscatter(self):
        """Backscatter must be 2D, not multi-channel."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew, 2)

        with pytest.raises(ValueError, match="2D|dimension"):
            SideScanGriddedRaster(
                grid_def=grid,
                backscatter=backscatter,
            )

    def test_rejects_nan_only_backscatter(self):
        """Cannot create raster with all NaN backscatter."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.full((grid.num_cells_ns, grid.num_cells_ew), np.nan)

        with pytest.raises(ValueError, match="all NaN|no data"):
            SideScanGriddedRaster(
                grid_def=grid,
                backscatter=backscatter,
            )


class TestSideScanGriddedRasterProperties:
    """Test accessor methods and derived properties."""

    @pytest.fixture
    def sample_raster(self):
        """Create a test raster."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
            channel_name="455 kHz",
        )

    def test_crs_property(self, sample_raster):
        """CRS accessible via property."""
        assert sample_raster.crs == CRS.from_epsg(4326)

    def test_bbox_property(self, sample_raster):
        """Bounding box accessible via property."""
        bbox = sample_raster.bbox
        assert bbox == (39.0, -71.0, 40.0, -70.0)

    def test_resolution_m_property(self, sample_raster):
        """Resolution accessible via property."""
        assert sample_raster.resolution_m == 1000.0

    def test_bounds_dict(self, sample_raster):
        """Bounds as dict with keys south/west/north/east."""
        bounds = sample_raster.bounds_dict
        assert bounds["south"] == 39.0
        assert bounds["west"] == -71.0
        assert bounds["north"] == 40.0
        assert bounds["east"] == -70.0

    def test_num_cells_ns(self, sample_raster):
        """Number of cells in north-south direction."""
        assert sample_raster.num_cells_ns == sample_raster.grid_def.num_cells_ns

    def test_num_cells_ew(self, sample_raster):
        """Number of cells in east-west direction."""
        assert sample_raster.num_cells_ew == sample_raster.grid_def.num_cells_ew

    def test_num_cells_total(self, sample_raster):
        """Total number of cells."""
        assert sample_raster.num_cells_total == sample_raster.grid_def.num_cells_total

    def test_extent_m_ns(self, sample_raster):
        """Extent in north-south direction."""
        extent = sample_raster.extent_m_ns
        assert extent > 100_000  # ~1 degree

    def test_extent_m_ew(self, sample_raster):
        """Extent in east-west direction."""
        extent = sample_raster.extent_m_ew
        assert extent > 80_000  # ~1 degree at latitude

    def test_data_coverage(self, sample_raster):
        """Fraction of non-NaN cells."""
        coverage = sample_raster.data_coverage
        assert 0.0 <= coverage <= 1.0
        # Our random raster should have ~100% coverage
        assert coverage > 0.99


class TestSideScanGriddedRasterXarraySchema:
    """Test that xarray Dataset has expected structure."""

    @pytest.fixture
    def sample_raster(self):
        """Create a test raster."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
            channel_name="455 kHz",
            processing_level="radiometric",
        )

    def test_dataset_has_backscatter_variable(self, sample_raster):
        """Dataset includes 'backscatter' as a data variable."""
        assert "backscatter" in sample_raster.dataset.data_vars
        assert sample_raster.dataset["backscatter"].shape == (
            sample_raster.num_cells_ns,
            sample_raster.num_cells_ew,
        )

    def test_dataset_has_grid_coords(self, sample_raster):
        """Dataset includes latitude and longitude coordinates."""
        assert "latitude" in sample_raster.dataset.coords
        assert "longitude" in sample_raster.dataset.coords
        lat = sample_raster.dataset["latitude"].values
        lon = sample_raster.dataset["longitude"].values
        # Should be 2D coordinate arrays
        assert lat.ndim == 2
        assert lon.ndim == 2
        assert lat.shape == (sample_raster.num_cells_ns, sample_raster.num_cells_ew)
        assert lon.shape == (sample_raster.num_cells_ns, sample_raster.num_cells_ew)

    def test_dataset_has_required_attrs(self, sample_raster):
        """Dataset has required global attributes."""
        attrs = sample_raster.dataset.attrs
        assert "crs" in attrs
        assert "crs_epsg" in attrs or "crs_wkt" in attrs
        assert "resolution_m" in attrs
        assert "channel_name" in attrs
        assert "processing_level" in attrs

    def test_backscatter_variable_has_units(self, sample_raster):
        """Backscatter variable includes units attribute."""
        bs = sample_raster.dataset["backscatter"]
        assert "units" in bs.attrs
        assert bs.attrs["units"] in ["dB", "linear", "dB re 1 μPa²"]

    def test_lat_lon_have_units(self, sample_raster):
        """Lat/lon coordinates have units."""
        assert "units" in sample_raster.dataset["latitude"].attrs
        assert "units" in sample_raster.dataset["longitude"].attrs
        assert sample_raster.dataset["latitude"].attrs["units"] == "degrees_north"
        assert sample_raster.dataset["longitude"].attrs["units"] == "degrees_east"


class TestSideScanGriddedRasterUnits:
    """
    Test units propagation. Regression coverage for a bug where
    SideScanGriddedRaster hardcoded units="linear" regardless of what was
    actually gridded, and viz/gridded.py hardcoded a matching "(linear)"
    colorbar label — silently mislabeling TVG-corrected (dB) data.
    """

    @pytest.fixture
    def grid(self):
        return GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)

    def test_default_units_is_linear(self, grid):
        """Default units remains 'linear' for backward compatibility."""
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter)

        assert raster.units == "linear"
        assert raster.dataset["backscatter"].attrs["units"] == "linear"

    def test_explicit_units_propagate_to_dataset(self, grid):
        """Explicit units="dB" is reflected in the dataset attrs."""
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, units="dB")

        assert raster.units == "dB"
        assert raster.dataset["backscatter"].attrs["units"] == "dB"

    def test_units_survive_slice_by_indices(self):
        """slice_by_indices() preserves the source raster's units."""
        fine_grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=1000.0)
        backscatter = np.random.rand(fine_grid.num_cells_ns, fine_grid.num_cells_ew)
        raster = SideScanGriddedRaster(grid_def=fine_grid, backscatter=backscatter, units="dB")

        sliced = raster.slice_by_indices(ns_slice=slice(10, 30), ew_slice=slice(15, 35))
        assert sliced.units == "dB"

    def test_units_survive_dict_round_trip(self, grid):
        """to_dict()/from_dict() preserves units."""
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, units="dB")

        d = raster.to_dict(include_data=True)
        restored = SideScanGriddedRaster.from_dict(d)

        assert restored.units == "dB"


class TestSideScanGriddedRasterReprojection:
    """Test reprojection to different CRS."""

    @pytest.fixture
    def sample_raster(self):
        """Create a test raster in WGS84."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )




    def test_reproject_same_crs_is_noop(self, sample_raster):
        """Reprojecting to same CRS returns equivalent raster."""
        same_crs_raster = sample_raster.reproject_to(CRS.from_epsg(4326))

        assert same_crs_raster.crs == sample_raster.crs
        np.testing.assert_array_equal(
            same_crs_raster.backscatter,
            sample_raster.backscatter,
        )

class TestSideScanGriddedRasterSlicing:
    """Test spatial subsetting operations."""

    @pytest.fixture
    def sample_raster(self):
        """Create a test raster."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.arange(
            grid.num_cells_ns * grid.num_cells_ew, dtype=float
        ).reshape(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

    def test_slice_by_bbox(self, sample_raster):
        """Slice raster by geographic bounding box."""
        subset = sample_raster.slice_by_bbox(
            bbox=(39.2, -70.8, 39.8, -70.2)
        )

        assert isinstance(subset, SideScanGriddedRaster)
        assert subset.backscatter.shape[0] < sample_raster.backscatter.shape[0]
        assert subset.backscatter.shape[1] < sample_raster.backscatter.shape[1]

    def test_slice_by_bbox_preserves_grid_def(self, sample_raster):
        """Subset bbox should match requested bounds."""
        target_bbox = (39.2, -70.8, 39.8, -70.2)
        subset = sample_raster.slice_by_bbox(bbox=target_bbox)

        south, west, north, east = subset.bbox
        target_south, target_west, target_north, target_east = target_bbox

        # Subset bounds should contain target (due to cell alignment)
        assert south >= target_south - 0.01
        assert west >= target_west - 0.01
        assert north <= target_north + 0.01
        assert east <= target_east + 0.01

    def test_slice_raises_on_no_overlap(self, sample_raster):
        """Slicing with no geographic overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap|intersect"):
            sample_raster.slice_by_bbox(
                bbox=(50.0, -60.0, 51.0, -59.0)  # Far away
            )

    def test_slice_by_indices(self, sample_raster):
        """Slice by direct array indices (ns_slice, ew_slice)."""
        subset = sample_raster.slice_by_indices(
            ns_slice=slice(10, 30),
            ew_slice=slice(15, 35),
        )

        assert isinstance(subset, SideScanGriddedRaster)
        assert subset.backscatter.shape == (20, 20)


class TestSideScanGriddedRasterSerialization:
    """Test to/from dict for config storage."""

    @pytest.fixture
    def sample_raster(self):
        """Create a test raster."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
            channel_name="455 kHz",
            processing_level="radiometric",
        )

    def test_to_dict_metadata(self, sample_raster):
        """Serialize metadata (not data) to dict."""
        d = sample_raster.to_dict(include_data=False)

        assert d["grid_def"]["bbox"] == sample_raster.bbox
        assert d["grid_def"]["resolution_m"] == sample_raster.resolution_m
        assert d["channel_name"] == "455 kHz"
        assert d["processing_level"] == "radiometric"

    def test_to_dict_with_data(self, sample_raster):
        """Serialize metadata + backscatter data to dict."""
        d = sample_raster.to_dict(include_data=True)

        assert "backscatter" in d
        # Should be a list (JSON-serializable)
        assert isinstance(d["backscatter"], (list, dict, np.ndarray))


class TestSideScanGriddedRasterEdgeCases:
    """Test boundary and edge cases."""

    def test_very_small_raster(self):
        """Grid with only 1 cell."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 39.0001, -70.9999),
            resolution_m=100_000.0,
        )
        backscatter = np.array([[1.0]])

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        assert raster.num_cells_total == 1

    def test_raster_with_some_nan(self):
        """Raster with partial NaN coverage (e.g., water-column gap)."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        # Mask out top portion (water column)
        backscatter[:10, :] = np.nan

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        assert raster.data_coverage < 1.0
        assert raster.data_coverage > 0.5

    def test_raster_with_infs(self):
        """Raster with inf values (should be allowed)."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        backscatter[0, 0] = np.inf

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
        )

        assert np.isinf(raster.backscatter[0, 0])

    def test_default_processing_level(self):
        """Default processing level if not specified."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)

        raster = SideScanGriddedRaster(
            grid_def=grid,
            backscatter=backscatter,
            # no processing_level specified
        )

        assert raster.processing_level == "raw"  # or other sensible default


class TestSideScanGriddedRasterProjectedCRS:
    """A projected (e.g. UTM) grid gets northing/easting coords in meters,
    not latitude/longitude in degrees — the coordinate NAME changes, not
    just its label/units, since a variable named "latitude" with meter
    values would be actively misleading to anyone inspecting the dataset."""

    @pytest.fixture
    def utm_raster(self):
        grid = GridDefinition(
            bbox=(4320000.0, 400000.0, 4330000.0, 410000.0),
            resolution_m=1000.0,
            crs=CRS.from_epsg(26919),
        )
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        return SideScanGriddedRaster(grid_def=grid, backscatter=backscatter)

    def test_is_geographic_false_for_utm(self, utm_raster):
        assert utm_raster.is_geographic is False

    def test_coord_names_are_northing_easting(self, utm_raster):
        assert utm_raster.y_coord_name == "northing"
        assert utm_raster.x_coord_name == "easting"

    def test_dataset_uses_northing_easting_not_lat_lon(self, utm_raster):
        assert "northing" in utm_raster.dataset.coords
        assert "easting" in utm_raster.dataset.coords
        assert "latitude" not in utm_raster.dataset.coords
        assert "longitude" not in utm_raster.dataset.coords

    def test_northing_easting_units_are_meters(self, utm_raster):
        assert utm_raster.dataset["northing"].attrs["units"] == "m"
        assert utm_raster.dataset["easting"].attrs["units"] == "m"

    def test_northing_easting_values_are_in_projected_range(self, utm_raster):
        # bbox was given directly in EPSG:26919 meters — cell centers
        # should fall within that range, not look like lat/lon degrees.
        assert utm_raster.y_values.min() >= 4_320_000
        assert utm_raster.y_values.max() <= 4_330_000
        assert utm_raster.x_values.min() >= 400_000
        assert utm_raster.x_values.max() <= 410_000

    def test_crs_epsg_in_attrs(self, utm_raster):
        assert utm_raster.dataset.attrs["crs_epsg"] == 26919

    def test_geographic_raster_still_uses_lat_lon(self):
        """Unchanged behavior for the default WGS84 case."""
        grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=1000.0)
        backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
        raster = SideScanGriddedRaster(grid_def=grid, backscatter=backscatter)

        assert raster.is_geographic is True
        assert raster.y_coord_name == "latitude"
        assert raster.x_coord_name == "longitude"
        assert "northing" not in raster.dataset.coords
        assert raster.dataset["latitude"].attrs["units"] == "degrees_north"
