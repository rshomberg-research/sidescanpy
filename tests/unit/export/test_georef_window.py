"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.export.georef_window import extract_georef_window, locate_point
from sidescanpy.grids.raster import SideScanGriddedRaster


def _real_center(ds):
    lat = ds["latitude_per_sample"].values
    lon = ds["longitude_per_sample"].values
    finite = np.isfinite(lat) & np.isfinite(lon)
    idx = tuple(a[len(a) // 2] for a in np.nonzero(finite))
    return float(lat[idx]), float(lon[idx])


class TestExtractGeorefWindow:
    def test_returns_gridded_raster(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        raster, _meta = extract_georef_window(georef_channel, 16, resolution_m=5.0, center_lat=center_lat, center_lon=center_lon)
        assert isinstance(raster, SideScanGriddedRaster)

    def test_grid_has_finite_cells(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        raster, _meta = extract_georef_window(georef_channel, 16, resolution_m=5.0, center_lat=center_lat, center_lon=center_lon)
        assert np.isfinite(raster.dataset["backscatter"].values).any()

    def test_output_is_exactly_square(self, georef_channel):
        # regression guard: size_px/resolution_m alone determine the grid,
        # not a WGS84-bbox roundtrip that could round to a slightly
        # different cell count per axis.
        center_lat, center_lon = _real_center(georef_channel)
        raster, _meta = extract_georef_window(georef_channel, 24, resolution_m=3.0, center_lat=center_lat, center_lon=center_lon)
        assert raster.dataset["backscatter"].values.shape == (24, 24)

    def test_center_is_reported_at_the_true_middle(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        _raster, meta = extract_georef_window(georef_channel, 16, resolution_m=5.0, center_lat=center_lat, center_lon=center_lon)
        assert meta["center_row"] == pytest.approx(8.0, abs=0.5)
        assert meta["center_col"] == pytest.approx(8.0, abs=0.5)

    def test_no_finite_latlon_raises(self, geometric_channel):
        # geometric_channel never had compute_georeferencing run, so
        # latitude_per_sample/longitude_per_sample don't exist at all
        with pytest.raises((ValueError, KeyError)):
            extract_georef_window(geometric_channel, 8, resolution_m=5.0, center_lat=41.5, center_lon=-71.5)


class TestLocatePoint:
    def test_matches_the_centering_point(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        _raster, meta = extract_georef_window(georef_channel, 16, resolution_m=5.0, center_lat=center_lat, center_lon=center_lon)
        row, col = locate_point(center_lat, center_lon, meta)
        assert row == pytest.approx(meta["center_row"], abs=1e-6)
        assert col == pytest.approx(meta["center_col"], abs=1e-6)


class TestGeorefGriddata:
    def test_griddata_reproduces_a_linear_field(self, georef_channel):
        """Linear interpolation is exact for a field linear in position."""
        ds = georef_channel.copy()
        lat = ds["latitude_per_sample"].values
        ds["backscatter_intensity"] = (ds["backscatter_intensity"].dims, (lat - np.nanmin(lat)) * 1e5)
        center_lat, center_lon = _real_center(ds)
        raster, meta = extract_georef_window(ds, 16, resolution_m=2.0, center_lat=center_lat, center_lon=center_lon,
                                             interp_method="linear", max_search_radius_m=10.0)
        grid = raster.dataset["backscatter"].values
        finite = np.isfinite(grid)
        assert finite.sum() > 20
        # Values rise with northing (row) and not with easting (column).
        rows = np.nanmean(np.where(finite, grid, np.nan), axis=1)
        assert np.all(np.diff(rows[np.isfinite(rows)]) > 0)

    def test_griddata_leaves_cells_far_from_samples_empty(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        raster, _ = extract_georef_window(georef_channel, 64, resolution_m=20.0, center_lat=center_lat,
                                          center_lon=center_lon, interp_method="linear", max_search_radius_m=5.0)
        grid = raster.dataset["backscatter"].values
        assert np.isnan(grid).any() and np.isfinite(grid).any()

    def test_griddata_records_its_method(self, georef_channel):
        center_lat, center_lon = _real_center(georef_channel)
        raster, _ = extract_georef_window(georef_channel, 16, resolution_m=5.0, center_lat=center_lat,
                                          center_lon=center_lon, interp_method="linear")
        assert raster.dataset.attrs["resampling_method"] == "griddata-linear"


class TestInterpolateScattered:
    def test_result_does_not_depend_on_coordinate_offset(self):
        """Dense anisotropic samples at UTM-sized coordinates interpolate as
        accurately as near the origin (triangulation precision): before
        the coordinates were centred, the median error grew tenfold."""
        from sidescanpy.export.geometric_window import interpolate_scattered

        a, c = np.meshgrid(np.arange(60) * 0.30, np.arange(600) * 0.024, indexing="ij")
        th = np.radians(245)
        x, y = a * np.cos(th) - c * np.sin(th), a * np.sin(th) + c * np.cos(th)
        values = (np.sin(x * 3) + np.cos(y * 2)).ravel()
        query = np.c_[np.linspace(y.min() + 2, y.max() - 2, 400), np.linspace(x.min() + 2, x.max() - 2, 400)]
        truth = np.sin(query[:, 1] * 3) + np.cos(query[:, 0] * 2)
        points = np.c_[y.ravel(), x.ravel()]
        offset = np.array([3.36e6, 6.0e5])
        local = interpolate_scattered(points, values, query, max_fill_distance_m=0.5)
        far = interpolate_scattered(points + offset, values, query + offset, max_fill_distance_m=0.5)
        assert np.isfinite(far).sum() > 200
        assert np.nanmax(np.abs(far - local)) < 0.05
        assert np.nanmedian(np.abs(far - truth)) < 0.05
