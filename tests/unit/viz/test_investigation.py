"""
Unit tests for sidescanpy.viz.investigation.

Focused on the real fixes made 2026-08-24: the beam-geometry far-ray
sign bug (both port and starboard rays should draw, not just one
duplicated) and the nadir-zoom water-column shading merge. Not
exhaustive smoke coverage of every plot function -- this module had no
test file at all before this pass.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sidescanpy.analysis.investigation import (
    StageResult,
    stage2_5_beam_geometry,
    stage2_nadir,
)
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.viz.investigation import (
    plot_stage2_5_beam_geometry,
    plot_stage2_nadir_zoom,
)


def _make_raw_dataset(*, n_pings: int = 20, n_samples: int = 200, sample_rate_hz: float = 1000.0) -> xr.Dataset:
    """Minimal valid raw sidescan dataset -- mirrors
    test_investigation.py's (analysis) own helper of the same name."""
    backscatter = np.ones((n_pings, 2, n_samples), dtype=np.float32)
    coords = {
        "ping": np.arange(n_pings),
        "channel": np.arange(2),
        "sample": np.arange(n_samples),
        "ping_number": (["ping"], np.arange(n_pings, dtype=np.int32)),
        "time": (["ping"], pd.date_range("2026-01-01", periods=n_pings, freq="1s").values),
        "channel_id": (["channel"], ["port", "starboard"]),
    }
    data_vars = {
        "backscatter_intensity": (["ping", "channel", "sample"], backscatter.copy()),
        "backscatter_raw": (["ping", "channel", "sample"], backscatter),
        "latitude": (["ping"], np.full(n_pings, 41.5, dtype=np.float32)),
        "longitude": (["ping"], np.full(n_pings, -71.5, dtype=np.float32)),
        "heading_deg": (["ping"], np.full(n_pings, 90.0, dtype=np.float32)),
        "pitch_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "roll_deg": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "depth_m": (["ping"], np.full(n_pings, 50.0, dtype=np.float32)),
        "altitude_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "fathometer_reading_m": (["ping"], np.full(n_pings, 10.0, dtype=np.float32)),
        "heave_m": (["ping"], np.zeros(n_pings, dtype=np.float32)),
        "sound_vel_ms": (["ping"], np.full(n_pings, 1500.0, dtype=np.float32)),
    }
    attrs = {
        "freq_hz_start": 1_495_000.0,
        "freq_hz_end": 1_495_000.0,
        "sample_rate_hz": sample_rate_hz,
        "pulse_length_us": 100.0,
        "processing_level": "raw",
        "processing_history": "[]",
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


class TestPlotStage2_5BeamGeometry:
    def test_far_ray_drawn_on_both_sides(self):
        """Regression test for the real bug: usable_range_m used to be
        stored unsigned for both port/starboard far_ray entries, so the
        top-panel diagram only ever drew a far ray toward the
        (arbitrarily) positive/starboard side, twice. After the sign
        fix, the top-panel axis must show real line data at both a
        negative x (port) and a positive x (starboard)."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 25.0})
        stage2_5 = stage2_5_beam_geometry(ds, stage2, stage3)

        fig, (ax_geo, ax_ping) = plot_stage2_5_beam_geometry(ds, stage2_5)
        try:
            all_x = np.concatenate([line.get_xdata() for line in ax_geo.get_lines()])
            assert np.any(all_x < -1.0), "expected a far ray drawn toward the port (negative) side"
            assert np.any(all_x > 1.0), "expected a far ray drawn toward the starboard (positive) side"
        finally:
            plt.close(fig)

    def test_wedge_shading_present_on_both_sides(self):
        """The illuminated-beam wedge fill must also land on both
        sides -- the pre-fix version's double-signing bug could put
        both wedges on the same side or invert one entirely."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 25.0})
        stage2_5 = stage2_5_beam_geometry(ds, stage2, stage3)

        fig, (ax_geo, ax_ping) = plot_stage2_5_beam_geometry(ds, stage2_5)
        try:
            patches = ax_geo.patches
            assert len(patches) >= 2, "expected at least one filled wedge per side"
            xs = [v for p in patches for v in p.get_xy()[:, 0]]
            assert any(x < -1.0 for x in xs)
            assert any(x > 1.0 for x in xs)
        finally:
            plt.close(fig)

    def test_no_stage_number_in_title(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        stage3 = StageResult(stage="stage3_usable_range", metrics={"usable_range_m": 25.0})
        stage2_5 = stage2_5_beam_geometry(ds, stage2, stage3)

        fig, (ax_geo, ax_ping) = plot_stage2_5_beam_geometry(ds, stage2_5)
        try:
            assert "Stage" not in ax_geo.get_title()
        finally:
            plt.close(fig)


class TestPlotStage2NadirZoom:
    def test_water_column_span_covers_both_sides_to_first_bottom_return(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)
        boundary = stage2.data.get("first_bottom_return_sample")
        assert boundary is not None

        fig, ax = plot_stage2_nadir_zoom(ds, stage2=stage2, nadir_n_samples=5)
        try:
            spans = [p for p in ax.patches if p.get_label() == "water column returns"]
            assert len(spans) == 1, "expected one unified water-column span, not separate window/gap regions"
            x0 = spans[0].get_x()
            x1 = x0 + spans[0].get_width()
            bottom_this = boundary[ds.sizes["ping"] // 2, 0]
            assert min(x0, x1) == pytest.approx(-bottom_this, abs=1.0)
            assert max(x0, x1) == pytest.approx(bottom_this, abs=1.0)
        finally:
            plt.close(fig)

    def test_altitude_and_bottom_return_lines_still_drawn(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_stage2_nadir_zoom(ds, stage2=stage2, nadir_n_samples=5)
        try:
            labels = [l.get_label() for l in ax.get_lines()]
            assert "altitude" in labels
            assert "first bottom return" in labels
        finally:
            plt.close(fig)

    def test_no_stage_number_in_title(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        fig, ax = plot_stage2_nadir_zoom(ds)
        try:
            assert "Stage" not in ax.get_title()
        finally:
            plt.close(fig)

    def test_falls_back_to_nadir_window_span_without_stage2(self):
        """No stage2 given -- still shades something labeled "water
        column returns" (the raw blanking window), doesn't crash or
        silently omit the shading."""
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        fig, ax = plot_stage2_nadir_zoom(ds, nadir_n_samples=5)
        try:
            spans = [p for p in ax.patches if p.get_label() == "water column returns"]
            assert len(spans) == 1
        finally:
            plt.close(fig)
