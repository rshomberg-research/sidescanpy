"""
Unit tests for sidescanpy.analysis.shadows: shadow + highlight object
proposals on synthetic speckled seabed with objects of known geometry.

2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.analysis import ShadowProposals, detect_shadow_proposals
from tests.conftest import make_minimal_sidescan_dataset

M_PER_PING = 0.24
M_PER_SAMPLE = 0.012          # 1500 / (2 * 62500)
ALTITUDE = 7.0
SEABED_DB = 65.0


def _seabed(n_pings=200, n_samples=2500, seed=0):
    """Speckled flat seabed in dB: 0.24 m pings, 1.2 cm samples, 7 m altitude, water column NaN."""
    rng = np.random.default_rng(seed)
    ds = make_minimal_sidescan_dataset(n_pings=n_pings, n_samples=n_samples, backscatter_val=SEABED_DB)
    ds.attrs["sample_rate_hz"] = 62500.0
    ds["latitude"] = ("ping", 44.0 + np.arange(n_pings) * M_PER_PING / 111_320.0)
    ds["longitude"] = ("ping", np.full(n_pings, -68.9))
    ds["altitude_m"] = ("ping", np.full(n_pings, ALTITUDE, dtype=np.float32))
    v = SEABED_DB + rng.normal(0, 4.0, (n_pings, 2, n_samples)).astype(np.float32)
    v[:, :, : int(ALTITUDE / M_PER_SAMPLE)] = np.nan
    ds["backscatter_intensity"].values[:] = v
    return ds


def _object(ds, *, channel, ping, n_pings, slant_m, height_m=0.5, highlight_m=0.3, shadow_drop_db=20.0,
            highlight_gain_db=12.0):
    """Highlight then shadow on ``n_pings`` pings; shadow length from the flat-seabed geometry."""
    s0 = int(slant_m / M_PER_SAMPLE)
    hl = int(highlight_m / M_PER_SAMPLE)
    r_shadow = slant_m + highlight_m
    ls = height_m * r_shadow / (ALTITUDE - height_m)
    n_sh = int(ls / M_PER_SAMPLE)
    v = ds["backscatter_intensity"].values
    v[ping:ping + n_pings, channel, s0:s0 + hl] += highlight_gain_db
    v[ping:ping + n_pings, channel, s0 + hl:s0 + hl + n_sh] -= shadow_drop_db
    return {"ping_start": ping, "ping_stop": ping + n_pings, "sample_start": s0, "sample_stop": s0 + hl + n_sh,
            "shadow_m": n_sh * M_PER_SAMPLE}


def _iou(row, box):
    pa = max(0, min(row.ping_stop, box["ping_stop"]) - max(row.ping_start, box["ping_start"]))
    sa = max(0, min(row.sample_stop, box["sample_stop"]) - max(row.sample_start, box["sample_start"]))
    inter = pa * sa
    a = (row.ping_stop - row.ping_start) * (row.sample_stop - row.sample_start)
    b = (box["ping_stop"] - box["ping_start"]) * (box["sample_stop"] - box["sample_start"])
    return inter / (a + b - inter)


class TestDetectShadowProposals:
    def test_finds_object_with_box_and_height(self):
        ds = _seabed()
        box = _object(ds, channel=1, ping=80, n_pings=6, slant_m=15.0, height_m=0.5)
        r = detect_shadow_proposals(ds, m_per_ping=M_PER_PING)
        assert isinstance(r, ShadowProposals)
        df = r.proposals[r.proposals.channel == 1]
        best = max(df.itertuples(), key=lambda row: _iou(row, box))
        assert _iou(best, box) > 0.6
        assert best.height_m == pytest.approx(0.5, abs=0.15)
        assert best.highlight_fraction > 0.5
        assert best.score > 0.5

    def test_objects_on_both_sides(self):
        ds = _seabed()
        port = _object(ds, channel=0, ping=40, n_pings=5, slant_m=12.0)
        stbd = _object(ds, channel=1, ping=140, n_pings=5, slant_m=20.0)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        for ch, box in ((0, port), (1, stbd)):
            assert max(_iou(row, box) for row in df[df.channel == ch].itertuples()) > 0.6

    def test_speckle_alone_gives_no_confident_proposal(self):
        df = detect_shadow_proposals(_seabed(), m_per_ping=M_PER_PING).proposals
        assert (df.score > 0.5).sum() == 0

    def test_single_ping_shadow_is_rejected(self):
        ds = _seabed()
        box = _object(ds, channel=1, ping=100, n_pings=1, slant_m=15.0)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        assert all(_iou(row, box) < 0.3 for row in df.itertuples())

    def test_shadow_without_highlight_scores_lower(self):
        ds = _seabed()
        with_hl = _object(ds, channel=1, ping=50, n_pings=6, slant_m=15.0)
        no_hl = _object(ds, channel=1, ping=130, n_pings=6, slant_m=15.0, highlight_m=0.0)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        a = max(df.itertuples(), key=lambda row: _iou(row, with_hl))
        b = max(df.itertuples(), key=lambda row: _iou(row, no_hl))
        assert b.highlight_fraction < a.highlight_fraction and b.score < a.score

    def test_blanked_water_column_is_never_shadow(self):
        df = detect_shadow_proposals(_seabed(), m_per_ping=M_PER_PING).proposals
        assert (df.sample_start >= int(ALTITUDE / M_PER_SAMPLE) - 10).all()

    def test_dark_band_at_water_column_edge_is_dropped(self):
        ds = _seabed()
        s0 = int(ALTITUDE / M_PER_SAMPLE)
        ds["backscatter_intensity"].values[60:90, 1, s0:s0 + 40] -= 20.0      # 7 m x 0.5 m dark band along the edge
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        assert not ((df.channel == 1) & (df.ping_start < 90) & (df.ping_stop > 60) & (df.sample_start < s0 + 40)).any()

    def test_too_tall_object_is_dropped(self):
        ds = _seabed()
        box = _object(ds, channel=1, ping=80, n_pings=6, slant_m=10.0, height_m=2.5)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING, max_height_m=1.0).proposals
        assert all(_iou(row, box) < 0.3 for row in df.itertuples())

    def test_shallow_short_shadow_near_nadir_found_by_weak_pass(self):
        ds = _seabed()
        box = _object(ds, channel=1, ping=80, n_pings=6, slant_m=8.5, height_m=0.15, shadow_drop_db=5.0,
                      highlight_gain_db=8.0)
        strong = detect_shadow_proposals(ds, m_per_ping=M_PER_PING, weak_shadow_db=None).proposals
        assert all(_iou(row, box) < 0.3 for row in strong.itertuples())
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        best = max(df.itertuples(), key=lambda row: _iou(row, box))
        assert _iou(best, box) > 0.4
        assert best.weak

    def test_weak_shadow_without_highlight_is_dropped(self):
        ds = _seabed()
        box = _object(ds, channel=1, ping=80, n_pings=6, slant_m=8.5, height_m=0.15, shadow_drop_db=5.0,
                      highlight_m=0.0)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        assert all(_iou(row, box) < 0.3 for row in df.itertuples())

    def test_score_uses_height_so_range_does_not_matter(self):
        ds = _seabed()
        near = _object(ds, channel=1, ping=50, n_pings=6, slant_m=9.0, height_m=0.4)
        far = _object(ds, channel=1, ping=130, n_pings=6, slant_m=25.0, height_m=0.4)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        a = max(df.itertuples(), key=lambda row: _iou(row, near))
        b = max(df.itertuples(), key=lambda row: _iou(row, far))
        assert a.shadow_length_m < 0.6 * b.shadow_length_m
        assert a.score == pytest.approx(b.score, abs=0.08)

    def test_result_serialises(self):
        ds = _seabed()
        _object(ds, channel=1, ping=80, n_pings=6, slant_m=15.0)
        d = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).to_dict()
        assert set(d["proposals"]) >= {"ping_start", "sample_stop", "score"}
        assert d["cell_samples"] == 5


class TestPlotShadowProposals:
    def test_boxes_drawn_in_native_and_window_coordinates(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from sidescanpy.export.windows.native import swath_tiles
        from sidescanpy.viz import plot_shadow_proposals

        ds = _seabed()
        _object(ds, channel=1, ping=80, n_pings=6, slant_m=15.0)
        r = detect_shadow_proposals(ds, m_per_ping=M_PER_PING)
        fig, ax = plt.subplots()
        plot_shadow_proposals(r, ax=ax, channel=1, min_score=0.5)
        assert len(ax.patches) == (r.proposals[(r.proposals.channel == 1)].score >= 0.5).sum() >= 1
        (img, win), = swath_tiles(ds, tile_m=1e6, px_m=0.05, m_per_ping=M_PER_PING)
        fig2, ax2 = plt.subplots()
        ax2.imshow(img)
        plot_shadow_proposals(r, ax=ax2, window=win, min_score=0.5)
        x, y = ax2.patches[0].get_xy()
        assert img.shape[1] / 2 <= x < img.shape[1] and 0 <= y < img.shape[0]   # starboard half
        plt.close("all")


def _seabed_with_water_column(seed=0, water_db=40.0):
    """As _seabed, but the water column holds dark speckle (not NaN), as in data that was not blanked."""
    ds = _seabed(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_wc = int(ALTITUDE / M_PER_SAMPLE)
    v = ds["backscatter_intensity"].values
    v[:, :, :n_wc] = water_db + rng.normal(0, 6.0, v[:, :, :n_wc].shape).astype(np.float32)
    return ds, n_wc


class TestShadowWaterColumn:
    def test_dark_water_column_gives_no_proposals(self):
        ds, n_wc = _seabed_with_water_column()
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        assert not (df.shadow_sample_start < n_wc).any()
        assert not (df.sample_start < n_wc - 10).any()

    def test_object_still_found_with_water_column_masked(self):
        ds, _ = _seabed_with_water_column()
        box = _object(ds, channel=1, ping=80, n_pings=6, slant_m=15.0, height_m=0.5)
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        best = max(df[df.channel == 1].itertuples(), key=lambda row: _iou(row, box))
        assert _iou(best, box) > 0.6
        assert best.height_m == pytest.approx(0.5, abs=0.15)

    def test_mask_water_column_false_keeps_old_behaviour(self):
        ds, n_wc = _seabed_with_water_column()
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING, mask_water_column=False).proposals
        assert (df.shadow_sample_start < n_wc).any()

    def test_water_column_not_masked_where_altitude_unknown(self):
        ds, n_wc = _seabed_with_water_column()
        ds["altitude_m"].values[:] = np.nan
        df = detect_shadow_proposals(ds, m_per_ping=M_PER_PING).proposals
        assert (df.shadow_sample_start < n_wc).any()

    def test_input_values_not_modified(self):
        ds, _ = _seabed_with_water_column()
        before = ds["backscatter_intensity"].values.copy()
        detect_shadow_proposals(ds, m_per_ping=M_PER_PING)
        np.testing.assert_array_equal(ds["backscatter_intensity"].values, before)
