"""
Unit tests for sidescanpy.viz.track — survey track + scan-line plotting.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyproj import CRS

from sidescanpy.viz.track import (
    _project,
    build_survey_track,
    export_track_and_scanlines_png,
    plot_track_and_scanlines,
)


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
            assert "32619" in ax.get_title()  # site-b-area lon/lat -> zone 19N
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


class TestExportTrackAndScanlinesPng:
    def test_export_creates_file(self, tmp_path):
        lat, lon, pl, po, sl, so = _make_track()
        out_path = tmp_path / "track.png"
        export_track_and_scanlines_png(lat, lon, pl, po, sl, so, str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestBuildSurveyTrack:
    def test_none_when_no_file_has_frequency(self, sample_multi_frequency_survey):
        surveys = {Path("a.ss.h5"): sample_multi_frequency_survey}
        result = build_survey_track(surveys, "hz_999999", scanline_stride=5)
        assert result is None

    def test_returns_track_and_scanline_arrays(self, sample_multi_frequency_survey):
        surveys = {Path("a.ss.h5"): sample_multi_frequency_survey}
        n_pings = sample_multi_frequency_survey.get_frequency("hz_1495000").dataset.sizes["ping"]
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)
        assert result is not None
        assert len(result["track_lat"]) == n_pings
        assert len(result["track_lon"]) == n_pings
        expected_n_rungs = len(np.arange(0, n_pings, 3))
        assert len(result["scan_port_lat"]) == expected_n_rungs
        assert len(result["scan_stbd_lat"]) == expected_n_rungs

    def test_multi_file_concatenates_in_time_order(self, sample_multi_frequency_survey):
        from sidescanpy.core import SideScanSurvey

        ds_a = sample_multi_frequency_survey.get_frequency("hz_1495000").dataset.copy()
        ds_b = ds_a.copy()
        # Shift file b's timestamps later so file a must sort first
        ds_b = ds_b.assign_coords(time=ds_b["time"] + np.timedelta64(1, "h"))
        ds_b["latitude"] = ds_a["latitude"] + 10.0  # distinguishable values

        survey_a = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey_a.add_frequency("hz_1495000", ds_a)
        survey_b = SideScanSurvey(source_files=["b.jsf"], source_format="JSF", survey_name="b")
        survey_b.add_frequency("hz_1495000", ds_b)

        surveys = {Path("b.ss.h5"): survey_b, Path("a.ss.h5"): survey_a}
        n_pings = ds_a.sizes["ping"]
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)
        # file a (earlier timestamps) must appear first in the concatenated track
        np.testing.assert_array_equal(result["track_lat"][:n_pings], ds_a["latitude"].values)
        np.testing.assert_array_equal(result["track_lat"][n_pings:], ds_b["latitude"].values)

    def test_mismatched_sample_window_lengths_do_not_nan_out_rungs(self):
        """Regression test: a real survey's two files can have
        differently-sized sample axes (e.g. one file's channel was
        trimmed further by the far-range noise crop than the other's).
        The last-sample "sample" coordinate then differs by value
        across files (e.g. index 79 vs. 99); concatenating without
        accounting for that used to align on the mismatched label and
        NaN out whichever file didn't have that particular value."""
        from sidescanpy.core import SideScanSurvey

        from tests.conftest import make_minimal_sidescan_dataset

        # altitude_m must be well below the (very short, default-fixture)
        # slant range reached by the last sample, or every sample is
        # legitimately water-column-masked regardless of this bug.
        # A real .ss.h5-loaded dataset carries an explicit indexed
        # "sample" coordinate (assigned by the HDF5 backend on open,
        # not by this in-memory fixture) — assign one here too, since
        # that indexed label is exactly what the old code aligned on.
        ds_a = make_minimal_sidescan_dataset(n_pings=10, n_samples=100)
        ds_a["altitude_m"] = (["ping"], np.full(10, 0.05, dtype=np.float32))
        ds_a = ds_a.assign_coords(sample=np.arange(ds_a.sizes["sample"]))
        ds_b = make_minimal_sidescan_dataset(n_pings=10, n_samples=80)
        ds_b["altitude_m"] = (["ping"], np.full(10, 0.05, dtype=np.float32))
        ds_b = ds_b.assign_coords(sample=np.arange(ds_b.sizes["sample"]))

        survey_a = SideScanSurvey(source_files=["a.jsf"], source_format="JSF", survey_name="a")
        survey_a.add_frequency("hz_1495000", ds_a)
        survey_b = SideScanSurvey(source_files=["b.jsf"], source_format="JSF", survey_name="b")
        survey_b.add_frequency("hz_1495000", ds_b)

        surveys = {Path("a.ss.h5"): survey_a, Path("b.ss.h5"): survey_b}
        result = build_survey_track(surveys, "hz_1495000", scanline_stride=3)

        assert not np.isnan(result["scan_port_lat"]).any()
        assert not np.isnan(result["scan_stbd_lat"]).any()


class TestBasemapIntegration:
    """show_land/show_locator -- see tests/unit/viz/test_basemap.py for
    the basemap module itself; these confirm the wiring into
    plot_track_and_scanlines/export_track_and_scanlines_png specifically."""

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

    def test_export_with_land_and_locator(self, tmp_path):
        lat, lon, pl, po, sl, so = _make_track()
        out = tmp_path / "track_basemap.png"
        export_track_and_scanlines_png(lat, lon, pl, po, sl, so, str(out), show_land=True, show_locator=True)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_default_false_adds_no_inset(self):
        lat, lon, pl, po, sl, so = _make_track()
        fig, ax = plot_track_and_scanlines(lat, lon, pl, po, sl, so)
        assert len(ax.child_axes) == 0
        plt.close(fig)
