"""
Unit tests for sidescanpy.viz.basemap -- land/water background and
locator thumbnail, built on the vendored 1:50m Natural Earth land
shapefile. Skipped entirely if the optional 'maps' extra (cartopy)
isn't installed, matching this project's convention for optional-extra
gated tests.

2026 rshomberg@uri.edu
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from pyproj import CRS

cartopy = pytest.importorskip("cartopy", reason="requires the optional 'maps' extra")

from sidescanpy.viz import basemap  # noqa: E402

# A real coastline area (Rhode Island Sound, near a real
# survey) -- deliberately not open ocean, so land is actually expected
# to render, not just "no crash."
_LON, _LAT = -71.41, 41.57


class TestAddLandBackground:
    def test_wgs84_sets_axes_limits_to_bbox_plus_margin(self):
        fig, ax = plt.subplots()
        bbox = (41.55, -71.45, 41.60, -71.35)  # (south, west, north, east)
        basemap.add_land_background(ax, bbox, CRS.from_epsg(4326))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # margin_frac=0.08 default widens the view beyond the raw bbox
        assert xlim[0] < bbox[1] and xlim[1] > bbox[3]
        assert ylim[0] < bbox[0] and ylim[1] > bbox[2]
        plt.close(fig)

    def test_utm_projected_crs_also_draws_without_error(self):
        from pyproj import Transformer

        from sidescanpy.grids.definition import utm_crs_for_lonlat

        crs = utm_crs_for_lonlat(_LON, _LAT)
        t = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
        x0, y0 = t.transform(_LON - 0.05, _LAT - 0.05)
        x1, y1 = t.transform(_LON + 0.05, _LAT + 0.05)
        fig, ax = plt.subplots()
        basemap.add_land_background(ax, (min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1)), crs)
        # some land collection was added (real coastline crosses this area)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_autoscale_disabled_so_later_plotting_does_not_override_view(self):
        fig, ax = plt.subplots()
        bbox = (41.55, -71.45, 41.60, -71.35)
        basemap.add_land_background(ax, bbox, CRS.from_epsg(4326))
        xlim_before = ax.get_xlim()
        ax.plot([-90, 90], [-80, 80])  # would blow the view wide open if autoscale were still on
        assert ax.get_xlim() == xlim_before
        plt.close(fig)

    def test_missing_cartopy_raises_clear_import_error(self, monkeypatch):
        monkeypatch.setattr(basemap, "_HAS_CARTOPY", False)
        fig, ax = plt.subplots()
        with pytest.raises(ImportError, match="maps"):
            basemap.add_land_background(ax, (41.55, -71.45, 41.60, -71.35), CRS.from_epsg(4326))
        plt.close(fig)


class TestPlotLocatorThumbnail:
    def test_creates_own_figure_when_none_given(self):
        fig, ax = basemap.plot_locator_thumbnail(_LON, _LAT, region_km=40)
        assert fig is not None
        assert ax.get_xticks().size == 0 or list(ax.get_xticks()) == []
        plt.close(fig)

    def test_draws_into_provided_axes(self):
        fig, ax = plt.subplots()
        inset = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
        returned_fig, returned_ax = basemap.plot_locator_thumbnail(_LON, _LAT, ax=inset, fig=fig)
        assert returned_ax is inset
        assert returned_fig is fig
        plt.close(fig)

    def test_small_mark_bbox_draws_a_point_not_a_rectangle(self):
        fig, ax = basemap.plot_locator_thumbnail(
            _LON, _LAT, region_km=50.0,
            mark_bbox_wgs84=(_LON - 0.0001, _LAT - 0.0001, _LON + 0.0001, _LAT + 0.0001),
        )
        from matplotlib.patches import Rectangle
        assert not any(isinstance(p, Rectangle) for p in ax.patches)
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_large_mark_bbox_draws_a_rectangle(self):
        fig, ax = basemap.plot_locator_thumbnail(
            _LON, _LAT, region_km=50.0,
            mark_bbox_wgs84=(_LON - 0.05, _LAT - 0.05, _LON + 0.05, _LAT + 0.05),
        )
        from matplotlib.patches import Rectangle
        assert any(isinstance(p, Rectangle) for p in ax.patches)
        plt.close(fig)
