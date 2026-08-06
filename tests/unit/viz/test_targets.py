"""
Unit tests for sidescanpy.viz.targets — track + scan-lines with target
catalog positions overlaid.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS

from sidescanpy.targets.core import Target
from sidescanpy.viz.targets import export_track_with_targets_png, plot_track_with_targets


def _make_track(n_pings=50, n_rungs=10):
    lat = np.linspace(41.50, 41.51, n_pings)
    lon = np.linspace(-71.40, -71.39, n_pings)
    idx = np.linspace(0, n_pings - 1, n_rungs).astype(int)
    port_lat, port_lon = lat[idx] - 0.0005, lon[idx]
    stbd_lat, stbd_lon = lat[idx] + 0.0005, lon[idx]
    return {
        "track_lat": lat, "track_lon": lon,
        "scan_port_lat": port_lat, "scan_port_lon": port_lon,
        "scan_stbd_lat": stbd_lat, "scan_stbd_lon": stbd_lon,
    }


def _make_target(target_id, lat=None, lon=None):
    return Target(
        target_id=target_id,
        source_format="jsf_discover",
        catalog_source=f"Target-{target_id}.target",
        latitude=lat,
        longitude=lon,
    )


class TestPlotTrackWithTargets:
    def test_basic_plot_returns_fig_ax(self):
        track = _make_track()
        targets = [_make_target("0001", 41.505, -71.395), _make_target("0002", 41.506, -71.394)]
        fig, ax = plot_track_with_targets(track, targets)
        try:
            assert fig is not None
            assert ax is not None
        finally:
            plt.close(fig)

    def test_scatter_count_matches_georeferenced_targets(self):
        track = _make_track()
        targets = [
            _make_target("0001", 41.505, -71.395),
            _make_target("0002", 41.506, -71.394),
            _make_target("0003", None, None),  # no position, excluded
        ]
        fig, ax = plot_track_with_targets(track, targets)
        try:
            assert "Targets (n=2)" in ax.get_legend().get_texts()[-1].get_text()
        finally:
            plt.close(fig)

    def test_no_georeferenced_targets_still_plots_track(self):
        track = _make_track()
        targets = [_make_target("0001", None, None)]
        fig, ax = plot_track_with_targets(track, targets)
        try:
            assert len(ax.lines) >= 1
        finally:
            plt.close(fig)

    def test_title_notes_target_overlay(self):
        track = _make_track()
        fig, ax = plot_track_with_targets(track, [])
        try:
            assert "target catalog overlay" in ax.get_title()
        finally:
            plt.close(fig)

    def test_explicit_geographic_crs(self):
        track = _make_track()
        targets = [_make_target("0001", 41.505, -71.395)]
        fig, ax = plot_track_with_targets(track, targets, crs=CRS.from_epsg(4326))
        try:
            assert "Longitude" in ax.get_xlabel()
        finally:
            plt.close(fig)


class TestExportTrackWithTargetsPng:
    def test_export_creates_file(self, tmp_path):
        track = _make_track()
        targets = [_make_target("0001", 41.505, -71.395)]
        out_path = tmp_path / "track_with_targets.png"
        export_track_with_targets_png(track, targets, str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
