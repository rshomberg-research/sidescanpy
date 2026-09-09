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
