"""
Unit tests for sidescanpy.viz.track — survey track + scan-line plotting.

2026 rshomberg@uri.edu
"""


import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyproj import CRS

from sidescanpy.viz.track import _project, plot_track_and_scanlines


def _make_track(n_pings=50, n_rungs=10):
    lat = np.linspace(41.50, 41.51, n_pings)
    lon = np.linspace(-71.40, -71.39, n_pings)
    idx = np.linspace(0, n_pings - 1, n_rungs).astype(int)
    # Port/starboard offset by a small constant lat delta as a stand-in
    # for a real across-track swath.
    port_lat, port_lon = lat[idx] - 0.0005, lon[idx]
    stbd_lat, stbd_lon = lat[idx] + 0.0005, lon[idx]
    return lat, lon, port_lat, port_lon, stbd_lat, stbd_lon


class TestProject:
    def test_geographic_crs_is_identity_lon_lat(self):
        lat = np.array([41.5, 41.6])
        lon = np.array([-71.4, -71.3])
        x, y = _project(lat, lon, CRS.from_epsg(4326))
        np.testing.assert_array_equal(x, lon)
        np.testing.assert_array_equal(y, lat)

    def test_projected_crs_returns_meter_scale_values(self):
        lat = np.array([41.5, 41.6])
        lon = np.array([-71.4, -71.3])
        x, y = _project(lat, lon, CRS.from_epsg(32619))
        assert (x > 100_000).all()
        assert (y > 1_000_000).all()


class TestPlotTrackAndScanlines:
    def test_basic_plot_returns_fig_ax(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        try:
            assert fig is not None
            assert ax is not None
            assert len(ax.lines) >= 1  # track line at minimum
            assert len(ax.collections) == 1  # scan-line LineCollection
        finally:
            plt.close(fig)

    def test_auto_utm_used_when_crs_not_given(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        try:
            assert "Easting" in ax.get_xlabel()
            assert "Northing" in ax.get_ylabel()
            assert "32619" in ax.get_title()  # fixture lon/lat (~ -68.8°) -> zone 19N
        finally:
            plt.close(fig)

    def test_explicit_geographic_crs(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(
            lat, lon, pl, po, sl, so, crs=CRS.from_epsg(4326),
        )
        try:
            assert "Longitude" in ax.get_xlabel()
            assert "Latitude" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_title_reports_ping_and_rung_counts(self):
        lat, lon, pl, po, sl, so = _make_track(n_pings=123, n_rungs=7)
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        try:
            assert "123" in ax.get_title()
            assert "7" in ax.get_title()
        finally:
            plt.close(fig)

    def test_empty_scanlines_still_plots_track(self):
        lat, lon, _, _, _, _ = _make_track()
        empty = np.array([])
        fig, ax = plot_track_and_scanlines(lat, lon, empty, empty, empty, empty)
        try:
            assert len(ax.lines) >= 1
        finally:
            plt.close(fig)

    def test_plot_existing_axes(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plt.subplots()
        try:
            fig2, ax2 = plot_track_and_scanlines(lat, lon, pl, po, sl, so, fig=fig, ax=ax)
            assert fig2 is fig
            assert ax2 is ax
        finally:
            plt.close(fig)

    def test_scale_bar_shown_by_default_for_projected_crs(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        try:
            assert len(ax.texts) >= 1
        finally:
            plt.close(fig)

    def test_scale_bar_can_be_disabled(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so, show_scale_bar=False)
        try:
            assert len(ax.texts) == 0
        finally:
            plt.close(fig)

    def test_scale_bar_skipped_for_geographic_crs(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(
            lat, lon, pl, po, sl, so, crs=CRS.from_epsg(4326),
        )
        try:
            assert len(ax.texts) == 0
        finally:
            plt.close(fig)


class TestBasemapIntegration:
    """show_land/show_locator -- see tests/unit/viz/test_basemap.py for
    the basemap module itself; these confirm the wiring into
    plot_track_and_scanlines specifically."""

    cartopy = pytest.importorskip("cartopy", reason="requires the optional 'maps' extra")

    def test_show_land_draws_a_background_collection(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so, show_land=True)
        # the scan-line LineCollection plus the land PatchCollection
        assert len(ax.collections) >= 2
        plt.close(fig)

    def test_show_locator_adds_an_inset_axes(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so, show_locator=True)
        # inset_axes() registers as a child of ax, not a top-level fig.axes entry
        assert len(ax.child_axes) >= 1
        plt.close(fig)


    def test_default_false_adds_no_inset(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        assert len(ax.child_axes) == 0
        plt.close(fig)
