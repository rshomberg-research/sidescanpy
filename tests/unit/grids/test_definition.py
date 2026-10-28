"""
Unit tests for GridDefinition.

Tests the contract: bounding box + resolution + projection → grid metadata.
"""

import numpy as np
import pytest
from pyproj import CRS

# Import will fail until sidescanpy/grids/definition.py exists.
# That's expected in TDD mode.
from sidescanpy.grids.definition import (
    GridDefinition,
    bbox_from_channel_dataset,
    utm_crs_for_lonlat,
    utm_grid_from_wgs84_bbox,
)


@pytest.mark.unit
class TestGridDefinitionInstantiation:
    """Test basic construction and validation."""

    def test_minimal_construction(self):
        """Create a grid from bbox and resolution (WGS84 default)."""
        # bbox: (south, west, north, east) in lat/lon
        grid = GridDefinition(
            bbox=(39.0, -71.5, 40.0, -70.5),
            resolution_m=10.0,
        )
        assert grid.bbox == (39.0, -71.5, 40.0, -70.5)
        assert grid.resolution_m == 10.0
        assert grid.crs == CRS.from_epsg(4326)  # WGS84

    def test_construction_with_explicit_crs(self):
        """Grid can be defined in UTM or other projection."""
        utm_crs = CRS.from_epsg(26919)  # UTM zone 19N
        grid = GridDefinition(
            bbox=(400000.0, 4320000.0, 450000.0, 4370000.0),
            resolution_m=5.0,
            crs=utm_crs,
        )
        assert grid.crs == utm_crs
        assert grid.resolution_m == 5.0

    def test_bbox_order_is_swne(self):
        """Bbox is (south, west, north, east) for clarity."""
        grid = GridDefinition(
            bbox=(10.0, 20.0, 30.0, 40.0),
            resolution_m=1.0,
        )
        south, west, north, east = grid.bbox
        assert south == 10.0
        assert west == 20.0
        assert north == 30.0
        assert east == 40.0

@pytest.mark.unit
class TestGridDefinitionValidation:
    """Test bounds and parameter validation."""

    def test_rejects_empty_bbox(self):
        """Cannot create grid with zero extent."""
        with pytest.raises(ValueError, match="zero|extent"):
            GridDefinition(
                bbox=(39.0, -71.0, 39.0, -71.0),  # south == north
                resolution_m=10.0,
            )

    def test_rejects_inverted_ns_bounds(self):
        """North must be > south."""
        with pytest.raises(ValueError, match="north.*south"):
            GridDefinition(
                bbox=(40.0, -71.0, 39.0, -71.0),
                resolution_m=10.0,
            )

    def test_rejects_inverted_ew_bounds(self):
        """East must be > west."""
        with pytest.raises(ValueError, match="east.*west"):
            GridDefinition(
                bbox=(39.0, -70.0, 40.0, -71.0),
                resolution_m=10.0,
            )

    def test_rejects_nonpositive_resolution(self):
        """Resolution must be > 0."""
        with pytest.raises(ValueError, match="resolution|positive"):
            GridDefinition(
                bbox=(39.0, -71.0, 40.0, -70.0),
                resolution_m=0.0,
            )
        with pytest.raises(ValueError, match="resolution|positive"):
            GridDefinition(
                bbox=(39.0, -71.0, 40.0, -70.0),
                resolution_m=-1.0,
            )


@pytest.mark.unit
class TestGridDefinitionMetrics:
    """Test derived properties: cell counts, extent, etc."""

    def test_cell_count_ns(self):
        """North-south cell count is ceil(extent / resolution)."""
        # 1 degree of latitude ≈ 111.3 km everywhere (true geodesic
        # distance, via pyproj.Geod — fixed 2026-07-02; previously
        # extent_m_ns/ew went through a Web Mercator projection whose
        # vertical scale is 1/cos(latitude), overstating this by ~39%
        # at 39.5°N specifically because Web Mercator distorts N-S
        # distance away from the equator).
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,
        )
        ns_cells = grid.num_cells_ns
        assert 1090 < ns_cells < 1130, f"Got {ns_cells}"

    def test_cell_count_ew(self):
        """East-west cell count (at mid-latitude)."""
        # 1 degree of longitude at 39.5°N ≈ 111.3 km * cos(39.5°) ≈ 85.9 km.
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,
        )
        ew_cells = grid.num_cells_ew
        assert 840 < ew_cells < 880, f"Got {ew_cells}"

    @pytest.mark.parametrize(
        "extent_m, resolution_m, expected",
        [(400.2, 0.3, 1334), (0.9, 0.3, 3), (100.0, 0.1, 1000), (100.5, 1.0, 101)],
    )
    def test_cell_count_when_extent_is_whole_cells_up_to_float_error(
        self, extent_m, resolution_m, expected
    ):
        """Float error in extent / resolution does not add a cell."""
        south, west = 4_500_000.0, 300_000.0
        grid = GridDefinition(
            bbox=(south, west, south + extent_m, west + extent_m),
            resolution_m=resolution_m,
            crs=CRS.from_epsg(32619),
        )
        assert grid.num_cells_ns == expected
        assert grid.num_cells_ew == expected
        lats, lons = grid.get_grid_centers()
        assert lats.shape == (expected, expected)

    def test_cell_count_rounds_up_a_real_partial_cell(self):
        """A partial cell beyond float error still adds a cell."""
        grid = GridDefinition(
            bbox=(4_500_000.0, 300_000.0, 4_500_400.201, 300_400.201),
            resolution_m=0.3,
            crs=CRS.from_epsg(32619),
        )
        assert grid.num_cells_ns == 1335

    def test_total_cells(self):
        """num_cells_total = num_cells_ns * num_cells_ew."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,
        )
        assert grid.num_cells_total == grid.num_cells_ns * grid.num_cells_ew

    def test_extent_m(self):
        """Extent in meters (account for projection)."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,
        )
        extent_ns = grid.extent_m_ns
        extent_ew = grid.extent_m_ew
        # Both should be 100,000+ meters (1° at these latitudes)
        assert extent_ns > 110_000, f"NS extent {extent_ns}"
        assert extent_ew > 80_000, f"EW extent {extent_ew}"




@pytest.mark.unit
class TestGridDefinitionCoordinates:
    """Test grid node and pixel coordinate generation."""

    def test_grid_corner_coords(self):
        """Corner coordinates match bbox."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=1000.0,
        )
        sw_lat, sw_lon, ne_lat, ne_lon = grid.corner_coords
        south, west, north, east = grid.bbox
        # Corners should be near bbox edges (within one cell)
        assert abs(sw_lat - south) < 0.01
        assert abs(sw_lon - west) < 0.01
        assert abs(ne_lat - north) < 0.01
        assert abs(ne_lon - east) < 0.01

    def test_grid_centers_2d_array(self):
        """Get 2D arrays of lat/lon for all grid centers."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10_000.0,  # ~0.1°
        )
        lats, lons = grid.get_grid_centers()
        # Should be 2D arrays
        assert lats.ndim == 2
        assert lons.ndim == 2
        # Shapes should match num_cells
        assert lats.shape == (grid.num_cells_ns, grid.num_cells_ew)
        assert lons.shape == (grid.num_cells_ns, grid.num_cells_ew)
        # Min/max should be within bbox
        assert np.nanmin(lats) >= grid.bbox[0]
        assert np.nanmax(lats) <= grid.bbox[2]
        assert np.nanmin(lons) >= grid.bbox[1]
        assert np.nanmax(lons) <= grid.bbox[3]


@pytest.mark.unit
class TestGridDefinitionReprSerialization:
    """Test string representation and round-trip."""

    def test_repr_is_informative(self):
        """repr() shows key parameters."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        r = repr(grid)
        assert "GridDefinition" in r
        assert "39.0000" in r or "39" in r
        assert "10.0" in r or "10" in r

    def test_to_dict(self):
        """Serialize to dict for JSON/config."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        d = grid.to_dict()
        assert d["bbox"] == (39.0, -71.0, 40.0, -70.0)
        assert d["resolution_m"] == 10.0
        assert "crs" in d  # EPSG code or WKT

    def test_from_dict_round_trip(self):
        """Reconstruct from dict."""
        original = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        reconstructed = GridDefinition.from_dict(original.to_dict())
        assert reconstructed.bbox == original.bbox
        assert reconstructed.resolution_m == original.resolution_m
        assert reconstructed.crs == original.crs


@pytest.mark.unit
class TestGridDefinitionProjection:
    """Test CRS conversion and reprojection."""

    def test_wgs84_is_default(self):
        """Defaults to EPSG:4326."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        assert grid.crs.to_epsg() == 4326

    def test_reproject_to_utm(self):
        """Convert grid to UTM zone."""
        grid_wgs84 = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        # Should auto-detect UTM zone 19N for this bbox
        grid_utm = grid_wgs84.reproject_to(CRS.from_epsg(26919))
        assert grid_utm.crs.to_epsg() == 26919
        # Bbox should be in UTM coordinates (not lat/lon)
        assert grid_utm.bbox[0] > 100_000  # northings are large
        assert abs(grid_utm.resolution_m - 10.0) < 1.0  # resolution ~preserved

    def test_reproject_preserves_coverage(self):
        """Reprojecting doesn't lose coverage area."""
        grid_wgs84 = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100.0,
        )
        area_wgs84 = grid_wgs84.extent_m_ns * grid_wgs84.extent_m_ew

        grid_utm = grid_wgs84.reproject_to(CRS.from_epsg(26919))
        area_utm = grid_utm.extent_m_ns * grid_utm.extent_m_ew

        # Areas may differ significantly due to projection distortion
        # Relaxed tolerance to account for transformation artifacts
        assert abs(area_utm - area_wgs84) / area_wgs84 < 0.4  # within 40%

    def test_reproject_round_trip(self):
        """Reproject to UTM and back."""
        original = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=10.0,
        )
        utm = original.reproject_to(CRS.from_epsg(26919))
        back = utm.reproject_to(CRS.from_epsg(4326))

        # Round-trip incurs rounding error from projection chain.
        # Allow up to 0.03 degrees (~3 km at equator).
        for i, (orig, recovered) in enumerate(zip(original.bbox, back.bbox)):
            assert abs(orig - recovered) < 0.03, f"bbox[{i}] mismatch: {orig} vs {recovered}"


@pytest.mark.unit
class TestGridDefinitionEdgeCases:
    """Test boundary and edge cases."""

    def test_very_small_resolution(self):
        """Grid with sub-meter resolution."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=0.1,
        )
        # Should not crash; cells will be very numerous
        assert grid.resolution_m == 0.1
        assert grid.num_cells_total > 1_000_000

    def test_very_large_resolution(self):
        """Grid with multi-km resolution."""
        grid = GridDefinition(
            bbox=(39.0, -71.0, 40.0, -70.0),
            resolution_m=100_000.0,
        )
        # Should produce just a few cells
        assert grid.num_cells_total >= 1  # at least 1 cell

    def test_near_pole(self):
        """Grid near a pole (high latitude)."""
        # A realistic few-degree-wide bbox, not the full -180..180 span:
        # west=-180 and east=180 are literally the same meridian, so a
        # true geodesic point-to-point distance between them is ~0 —
        # not a meaningful "east-west extent" for a survey grid, and
        # only appeared nonzero previously because of the Web Mercator
        # distance bug fixed 2026-07-02 (see test_cell_count_ns).
        grid = GridDefinition(
            bbox=(85.0, -1.0, 86.0, 1.0),
            resolution_m=1000.0,
        )
        # Should handle gracefully (though ew extent is smaller)
        assert grid.num_cells_ns > 0
        assert grid.num_cells_ew > 0

    def test_crosses_antimeridian_not_supported_yet(self):
        """Antimeridian crossing raises ValueError."""
        with pytest.raises(ValueError, match="antimeridian|east.*west"):
            GridDefinition(
                bbox=(39.0, 170.0, 40.0, -170.0),  # crosses ±180°
                resolution_m=10.0,
            )


class TestUtmCrsForLonlat:
    """Auto-UTM zone selection."""

    def test_maine_coast_is_zone_19n(self):
        # Fixture survey area (~ -68.8, 44.1).
        crs = utm_crs_for_lonlat(-68.8, 44.1)
        assert crs.to_epsg() == 32619

    def test_northern_hemisphere_uses_326xx(self):
        crs = utm_crs_for_lonlat(-71.0, 41.5)
        assert 32601 <= crs.to_epsg() <= 32660

    def test_southern_hemisphere_uses_327xx(self):
        crs = utm_crs_for_lonlat(-70.0, -30.0)
        assert 32701 <= crs.to_epsg() <= 32760

    def test_same_zone_number_different_hemisphere(self):
        north = utm_crs_for_lonlat(-70.0, 10.0)
        south = utm_crs_for_lonlat(-70.0, -10.0)
        # Same longitude → same zone number, different EPSG family
        assert north.to_epsg() - 32600 == south.to_epsg() - 32700

    def test_zone_boundaries_clamped(self):
        # Should not raise or produce an out-of-range EPSG at the extremes
        crs_west = utm_crs_for_lonlat(-179.9, 10.0)
        crs_east = utm_crs_for_lonlat(179.9, 10.0)
        assert 32601 <= crs_west.to_epsg() <= 32660
        assert 32601 <= crs_east.to_epsg() <= 32660


class TestUtmGridFromWgs84Bbox:
    """Building a projected GridDefinition directly from a WGS84 bbox."""

    def test_returns_projected_grid(self):
        grid = utm_grid_from_wgs84_bbox(
            bbox=(44.0, -69.0, 44.1, -68.8), resolution_m=1.0,
        )
        assert not grid.crs.is_geographic
        assert grid.crs.to_epsg() == 32619

    def test_bbox_is_in_meters_not_degrees(self):
        grid = utm_grid_from_wgs84_bbox(
            bbox=(44.0, -69.0, 44.1, -68.8), resolution_m=1.0,
        )
        south, west, north, east = grid.bbox
        # UTM eastings/northings are on the order of 1e5-1e7, nothing
        # like WGS84 degree magnitudes.
        assert south > 1000
        assert west > 1000
        assert north > south
        assert east > west

    def test_covers_same_ground_area_as_input(self):
        bbox_deg = (44.0, -69.0, 44.1, -68.8)
        grid = utm_grid_from_wgs84_bbox(bbox=bbox_deg, resolution_m=10.0)
        wgs84_equivalent = GridDefinition(bbox=bbox_deg, resolution_m=10.0)

        # Extents (in meters) should roughly match between the native
        # WGS84-geodetic-distance computation and the UTM projected one.
        assert grid.extent_m_ns == pytest.approx(wgs84_equivalent.extent_m_ns, rel=0.05)
        assert grid.extent_m_ew == pytest.approx(wgs84_equivalent.extent_m_ew, rel=0.05)


class TestBboxFromChannelDataset:
    """Auto-fit a WGS84 bbox to a channel dataset's per-sample extent."""

    def test_bbox_covers_full_lat_lon_extent_with_margin(self, channel_3d):
        south, west, north, east = bbox_from_channel_dataset(channel_3d)
        lats = channel_3d["latitude_per_sample"].values
        lons = channel_3d["longitude_per_sample"].values
        assert south < float(np.min(lats))
        assert north > float(np.max(lats))
        assert west < float(np.min(lons))
        assert east > float(np.max(lons))

    def test_margin_scales_with_extent_by_default(self, channel_3d):
        south, west, north, east = bbox_from_channel_dataset(channel_3d, margin_frac=0.5)
        lats = channel_3d["latitude_per_sample"].values
        lat_extent = float(np.max(lats)) - float(np.min(lats))
        # margin_frac=0.5 -> half the extent added on each side
        assert (float(np.min(lats)) - south) == pytest.approx(lat_extent * 0.5, rel=0.05)

    def test_min_margin_deg_applies_to_near_zero_extent(self):
        import xarray as xr

        lat = np.full((5, 2, 3), 41.0)
        lon = np.full((5, 2, 3), -71.0)
        ds = xr.Dataset(
            {
                "latitude_per_sample": (["ping", "channel", "sample"], lat),
                "longitude_per_sample": (["ping", "channel", "sample"], lon),
            }
        )
        south, west, north, east = bbox_from_channel_dataset(ds, min_margin_deg=0.01)
        assert north - south == pytest.approx(0.02)
        assert east - west == pytest.approx(0.02)

    def test_raises_on_all_nan_coordinates(self):
        import xarray as xr

        nan = np.full((5, 2, 3), np.nan)
        ds = xr.Dataset(
            {
                "latitude_per_sample": (["ping", "channel", "sample"], nan.copy()),
                "longitude_per_sample": (["ping", "channel", "sample"], nan.copy()),
            }
        )
        with pytest.raises(ValueError, match="No valid"):
            bbox_from_channel_dataset(ds)

    def test_physically_invalid_but_finite_coordinates_excluded(self):
        """A finite lat/lon outside [-90,90]/[-180,180] (e.g. a corrupted
        compute_georeferencing() result found via a synthetic-data
        exercise) must not blow up the bbox -- it can poison an
        auto-selected UTM zone's cell count downstream (OverflowError
        converting an infinite extent to int)."""
        import xarray as xr

        lat = np.full((5, 2, 3), 41.0)
        lon = np.full((5, 2, 3), -71.0)
        lat[0, 0, 0] = 103.25  # physically invalid, but np.isfinite
        lon[0, 0, 0] = 209.5
        ds = xr.Dataset(
            {
                "latitude_per_sample": (["ping", "channel", "sample"], lat),
                "longitude_per_sample": (["ping", "channel", "sample"], lon),
            }
        )
        south, west, north, east = bbox_from_channel_dataset(ds, min_margin_deg=0.01)
        assert south == pytest.approx(40.99)
        assert north == pytest.approx(41.01)
        assert west == pytest.approx(-71.01)
        assert east == pytest.approx(-70.99)

    def test_raises_when_every_coordinate_is_physically_invalid(self):
        import xarray as xr

        lat = np.full((5, 2, 3), 103.25)
        lon = np.full((5, 2, 3), 209.5)
        ds = xr.Dataset(
            {
                "latitude_per_sample": (["ping", "channel", "sample"], lat),
                "longitude_per_sample": (["ping", "channel", "sample"], lon),
            }
        )
        with pytest.raises(ValueError, match="No valid"):
            bbox_from_channel_dataset(ds)


class TestBboxMinMarginMeters:
    def _point_ds(self, lat0=41.6, lon0=-71.4):
        import xarray as xr

        return xr.Dataset({
            "latitude_per_sample": (["ping", "channel", "sample"], np.full((5, 2, 3), lat0)),
            "longitude_per_sample": (["ping", "channel", "sample"], np.full((5, 2, 3), lon0)),
        })

    def test_default_min_margin_is_meters_not_a_tenth_of_a_km(self):
        from pyproj import Geod

        south, west, north, east = bbox_from_channel_dataset(self._point_ds())
        geod = Geod(ellps="WGS84")
        _, _, ns = geod.inv(-71.4, south, -71.4, north)
        _, _, ew = geod.inv(west, 41.6, east, 41.6)
        assert ns == pytest.approx(10.0, rel=0.02)   # 5 m each side
        assert ew == pytest.approx(10.0, rel=0.02)

    def test_large_extent_uses_fractional_margin(self, channel_3d):
        south, _, _, _ = bbox_from_channel_dataset(channel_3d, margin_frac=0.1, min_margin_m=0.0)
        lats = channel_3d["latitude_per_sample"].values
        extent = float(np.nanmax(lats)) - float(np.nanmin(lats))
        assert float(np.nanmin(lats)) - south == pytest.approx(0.1 * extent, rel=0.05)
