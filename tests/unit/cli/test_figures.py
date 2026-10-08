"""
Unit tests for sidescanpy.cli._figures -- the PNG writers behind
``sidescan-export grid/track`` and ``sidescan-convert targets``.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest
from PIL import Image

from sidescanpy.cli import _figures
from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.processing.track import SurveyTrack
from sidescanpy.targets.core import Target


@pytest.fixture
def sample_raster():
    grid = GridDefinition(bbox=(39.0, -71.0, 40.0, -70.0), resolution_m=5000.0)
    backscatter = np.random.rand(grid.num_cells_ns, grid.num_cells_ew)
    return SideScanGriddedRaster(grid_def=grid, backscatter=backscatter, channel_name="455 kHz")


def _make_track(n_pings=50, n_rungs=10) -> SurveyTrack:
    lat = np.linspace(41.50, 41.51, n_pings)
    lon = np.linspace(-71.40, -71.39, n_pings)
    idx = np.linspace(0, n_pings - 1, n_rungs).astype(int)
    return SurveyTrack(
        track_lat=lat, track_lon=lon,
        scan_port_lat=lat[idx] - 0.0005, scan_port_lon=lon[idx],
        scan_stbd_lat=lat[idx] + 0.0005, scan_stbd_lon=lon[idx],
    )


def _make_narrow_track(n_pings=50) -> SurveyTrack:
    """A long/skinny track: latitude spans ~50km, longitude barely moves."""
    lat = np.linspace(41.00, 41.45, n_pings)
    lon = np.full(n_pings, -71.40)
    idx = np.linspace(0, n_pings - 1, 10).astype(int)
    return SurveyTrack(
        track_lat=lat, track_lon=lon,
        scan_port_lat=lat[idx], scan_port_lon=lon[idx] - 0.0002,
        scan_stbd_lat=lat[idx], scan_stbd_lon=lon[idx] + 0.0002,
    )


def _make_target(target_id, lat=None, lon=None):
    return Target(
        target_id=target_id,
        source_format="jsf_discover",
        catalog_source=f"Target-{target_id}.target",
        latitude=lat,
        longitude=lon,
    )


class TestSaveGriddedPng:
    def test_writes_png(self, sample_raster, tmp_path):
        out = tmp_path / "raster.png"
        _figures.save_gridded_png(sample_raster, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_dpi(self, sample_raster, tmp_path):
        low, high = tmp_path / "low.png", tmp_path / "high.png"
        _figures.save_gridded_png(sample_raster, str(low), dpi=50)
        _figures.save_gridded_png(sample_raster, str(high), dpi=200)
        assert Image.open(high).size[0] > Image.open(low).size[0]

    def test_custom_cmap_and_gamma(self, sample_raster, tmp_path):
        out = tmp_path / "raster.png"
        _figures.save_gridded_png(sample_raster, str(out), cmap="hot", gamma=2.5)
        assert out.stat().st_size > 0


class TestSaveTrackPng:
    def test_writes_png(self, tmp_path):
        out = tmp_path / "track.png"
        _figures.save_track_png(_make_track(), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saved_dimensions_constant_regardless_of_track_shape(self, tmp_path):
        """Every saved track.png has the same pixel size whatever the
        survey's aspect ratio (no bbox_inches="tight" cropping)."""
        wide, narrow = tmp_path / "wide.png", tmp_path / "narrow.png"
        _figures.save_track_png(_make_track(), str(wide))
        _figures.save_track_png(_make_narrow_track(), str(narrow))
        assert Image.open(wide).size == Image.open(narrow).size


class TestSaveTrackWithTargetsPng:
    def test_writes_png(self, tmp_path):
        out = tmp_path / "track_with_targets.png"
        _figures.save_track_with_targets_png(_make_track(), [_make_target("0001", 41.505, -71.395)], str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saved_dimensions_constant_regardless_of_track_shape(self, tmp_path):
        targets = [_make_target("0001", 41.505, -71.395)]
        wide, narrow = tmp_path / "wide.png", tmp_path / "narrow.png"
        _figures.save_track_with_targets_png(_make_track(), targets, str(wide))
        _figures.save_track_with_targets_png(_make_narrow_track(), targets, str(narrow))
        assert Image.open(wide).size == Image.open(narrow).size


class TestSaveTargetPositionScatterPng:
    def test_writes_file_when_positions_exist(self, tmp_path):
        out = tmp_path / "scatter.png"
        assert _figures.save_target_position_scatter_png([_make_target("0001", 41.5, -71.4)], str(out)) is True
        assert out.stat().st_size > 0

    def test_no_file_written_when_no_positions(self, tmp_path):
        out = tmp_path / "scatter.png"
        assert _figures.save_target_position_scatter_png([_make_target("0001")], str(out)) is False
        assert not out.exists()


class TestSaveWaterfallPng:
    def test_draws_one_box_per_rect(self, tmp_path, monkeypatch):
        from tests.unit.synthetic import make_raw_dataset

        ds = make_raw_dataset(n_pings=20, n_samples=50, backscatter=np.random.rand(20, 2, 50).astype(np.float32) + 1)
        drawn = []
        orig_close = _figures.plt.close

        def _capture_close(fig):
            ax = fig.axes[0]
            drawn.append((len(ax.patches), [t.get_text() for t in ax.texts]))
            orig_close(fig)

        monkeypatch.setattr(_figures.plt, "close", _capture_close)
        out = tmp_path / "wf.png"
        _figures.save_waterfall_png(
            ds, out, x_axis="sample", y_axis="ping", ping_offset=0, cmap="copper",
            vmin=None, vmax=None, vmin_pct=1.0, vmax_pct=99.0, gamma=1.0, dpi=50,
            box_rects=[("T1", 5.0, 5.0, 4.0, 4.0), ("T2", 20.0, 10.0, 4.0, 4.0)],
        )
        assert out.stat().st_size > 0
        assert drawn == [(2, ["T1", "T2"])]


class TestSaveTvgFitPng:
    def test_writes_two_panel_figure(self, tmp_path):
        from sidescanpy.analysis.tvg import fit_range_decay
        from sidescanpy.processing import compute_geometric_ranges
        from tests.unit.synthetic import make_raw_dataset

        n_samples = 400
        decay = 100.0 * np.exp(-np.arange(n_samples) / 120.0) + 1.0
        rng = np.random.default_rng(0)
        data = decay[None, None, :] * (1.0 + 0.02 * rng.standard_normal((30, 2, n_samples)))
        ds = compute_geometric_ranges(make_raw_dataset(backscatter=data.astype(np.float32)))
        result = fit_range_decay(ds, bin_width_m=5.0, usable_range_m=250.0)
        if result.fit is None:
            pytest.skip("synthetic decay did not produce a fit")
        out = tmp_path / "tvg_fit.png"
        _figures.save_tvg_fit_png(result, out, suptitle="synthetic", residual_xlabel="Slant range (m)")
        assert out.stat().st_size > 0


class TestBasemapIntegration:
    cartopy = pytest.importorskip("cartopy", reason="requires the optional 'maps' extra")

    def test_gridded_with_land_and_locator(self, sample_raster, tmp_path):
        out = tmp_path / "raster_basemap.png"
        _figures.save_gridded_png(sample_raster, str(out), show_land=True, show_locator=True)
        assert out.stat().st_size > 0

    def test_track_with_land_and_locator(self, tmp_path):
        out = tmp_path / "track_basemap.png"
        _figures.save_track_png(_make_track(), str(out), show_land=True, show_locator=True)
        assert out.stat().st_size > 0

    def test_track_with_targets_land_and_locator(self, tmp_path):
        out = tmp_path / "track_targets_basemap.png"
        _figures.save_track_with_targets_png(
            _make_track(), [_make_target("0001", 41.505, -71.395)], str(out), show_land=True, show_locator=True,
        )
        assert out.stat().st_size > 0
