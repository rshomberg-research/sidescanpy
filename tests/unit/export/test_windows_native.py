"""
Unit tests for sidescanpy.export.windows.native: physically square chips and
dual-swath tiles cut from the native ping x sample grid, and the pixel <->
native coordinate mapping used to round-trip annotations.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.export.windows.native import (
    NativeWindow,
    along_track_spacing_m,
    native_chip,
    swath_tiles,
)
from tests.conftest import make_minimal_sidescan_dataset

M_PER_PING = 0.24


def _survey(n_pings=400, n_samples=500, sample_rate_hz=31250.0):
    """Synthetic channel: 0.24 m along-track spacing, 0.024 m samples, value = 50 dB."""
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, backscatter_val=50.0)
    ds.attrs["sample_rate_hz"] = sample_rate_hz
    ds["latitude"] = ("ping", 44.0 + np.arange(n_pings) * M_PER_PING / 111_320.0)
    ds["longitude"] = ("ping", np.full(n_pings, -68.9))
    return ds


class TestAlongTrackSpacing:
    def test_spacing_from_positions(self):
        assert along_track_spacing_m(_survey()) == pytest.approx(M_PER_PING, rel=0.01)

    def test_repeated_fixes_do_not_shrink_spacing(self):
        ds = _survey()
        lat = ds["latitude"].values.copy()
        lat = np.repeat(lat[::10], 10)[: len(lat)]          # each fix held for 10 pings
        ds["latitude"] = ("ping", lat)
        assert along_track_spacing_m(ds) == pytest.approx(M_PER_PING, rel=0.05)


class TestNativeChip:
    def test_square_in_metres_and_sized(self):
        ds = _survey()
        img, win = native_chip(ds, ping=200, channel=1, sample=250, size_m=6.0, size_px=128)
        assert img.shape == (128, 128)
        assert (win.ping_stop - win.ping_start) * win.m_per_ping == pytest.approx(6.0, abs=M_PER_PING)
        assert (win.sample_stop - win.sample_start) * win.m_per_sample == pytest.approx(6.0, abs=0.024)

    def test_target_maps_to_centre(self):
        ds = _survey()
        _, win = native_chip(ds, ping=200, channel=1, sample=250, size_m=6.0, size_px=128)
        x, y = win.to_pixel(200.5, 250.5, 1)
        assert x == pytest.approx(64, abs=3) and y == pytest.approx(64, abs=6)

    def test_feature_lands_where_mapped(self):
        ds = _survey()
        ds["backscatter_intensity"][210, 1, 260] = 100.0
        img, win = native_chip(ds, ping=200, channel=1, sample=250, size_m=6.0, size_px=250)
        x, y = win.to_pixel(210.5, 260.5, 1)
        yy, xx = np.unravel_index(np.nanargmax(img), img.shape)
        assert abs(xx - x) <= 2 and abs(yy - y) <= 3

    def test_edge_is_nan_padded_and_target_stays_centred(self):
        ds = _survey()
        img, win = native_chip(ds, ping=3, channel=0, sample=5, size_m=6.0, size_px=100)
        assert np.isnan(img[0, 0]) and np.isfinite(img[60, 60])
        x, y = win.to_pixel(3.5, 5.5, 0)
        assert x == pytest.approx(50, abs=3) and y == pytest.approx(50, abs=6)

    def test_nan_cells_stay_nan_after_resize(self):
        ds = _survey()
        ds["backscatter_intensity"][:, 1, :240] = np.nan
        img, win = native_chip(ds, ping=200, channel=1, sample=250, size_m=6.0, size_px=128)
        x_edge, _ = win.to_pixel(200, 240, 1)
        assert np.isnan(img[64, : int(x_edge) - 2]).all()
        assert np.isfinite(img[64, int(x_edge) + 2:]).all()

    def test_round_trip(self):
        ds = _survey()
        _, win = native_chip(ds, ping=200, channel=0, sample=250, size_m=10.0, size_px=200)
        p, s, c = win.to_native(*win.to_pixel(205.0, 230.0, 0))
        assert (p, s, c) == (pytest.approx(205.0), pytest.approx(230.0), 0)


class TestSwathTiles:
    def test_tiles_cover_the_survey(self):
        ds = _survey(n_pings=400)
        tiles = swath_tiles(ds, tile_m=24.0, px_m=0.1, overlap_m=4.8)
        starts = [w.ping_start for _, w in tiles]
        assert starts[0] == 0 and tiles[-1][1].ping_stop == 400
        assert all(b - a == 80 for a, b in zip(starts, starts[1:]))  # 100-ping tiles, 20-ping overlap

    def test_dual_layout_port_mirrored_on_the_left(self):
        ds = _survey(n_pings=100)
        ds["backscatter_intensity"][50, 0, 100] = 100.0   # port, 2.4 m out
        ds["backscatter_intensity"][50, 1, 100] = 100.0   # starboard, 2.4 m out
        (img, win), = swath_tiles(ds, tile_m=24.0, px_m=0.024)
        side_w = img.shape[1] // 2
        hot = np.argwhere(img > 75)
        xs = sorted(set(hot[:, 1]))
        assert xs[0] < side_w <= xs[-1]
        assert abs((side_w - xs[0]) - (xs[-1] - side_w)) <= 2           # symmetric about the centre
        assert win.to_native(xs[0] + 0.5, hot[0, 0] + 0.5)[2] == 0
        assert win.to_native(xs[-1] + 0.5, hot[0, 0] + 0.5)[2] == 1

    def test_box_round_trip(self):
        ds = _survey(n_pings=200)
        (_, win), *_ = swath_tiles(ds, tile_m=48.0, px_m=0.1)
        x0, y0 = win.to_pixel(40, 120, 1)
        x1, y1 = win.to_pixel(50, 160, 1)
        box = win.box_to_native(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        assert box["channel"] == 1
        assert box["ping_start"] == pytest.approx(40) and box["ping_stop"] == pytest.approx(50)
        assert box["sample_start"] == pytest.approx(120) and box["sample_stop"] == pytest.approx(160)

    def test_port_box_round_trip(self):
        ds = _survey(n_pings=200)
        (_, win), *_ = swath_tiles(ds, tile_m=48.0, px_m=0.1)
        x0, y0 = win.to_pixel(40, 120, 0)
        x1, y1 = win.to_pixel(50, 160, 0)
        box = win.box_to_native(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        assert box["channel"] == 0
        assert (box["sample_start"], box["sample_stop"]) == (pytest.approx(120), pytest.approx(160))

    def test_window_serialises(self):
        ds = _survey(n_pings=100)
        (_, win), = swath_tiles(ds, tile_m=24.0, px_m=0.1)
        assert NativeWindow.from_dict(win.to_dict()) == win
