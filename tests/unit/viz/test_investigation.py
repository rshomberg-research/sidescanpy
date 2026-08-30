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
    stage4_decay,
)
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db
from sidescanpy.viz.investigation import (
    plot_stage2_5_beam_geometry,
    plot_stage2_nadir_zoom,
    plot_stage2_waterfall,
    plot_stage4_residuals,
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


class TestPlotStage2Waterfall:
    """ping_offset added 2026-08-31 -- a caller plotting an already-sliced
    window (e.g. one section of a larger survey) needs the traced
    boundary lines to land on the same real ping numbers as the
    underlying waterfall image, not the image's own 0-based offset and
    the trace's separate, always-0-based one silently disagreeing."""

    def test_default_ping_offset_is_zero_based_unchanged(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_stage2_waterfall(ds, stage2=stage2)
        try:
            bottom_line = next(line for line in ax.get_lines() if line.get_label() == "first bottom return")
            assert bottom_line.get_ydata()[0] == 0
        finally:
            plt.close(fig)

    def test_ping_offset_shifts_both_image_and_traced_boundary(self):
        ds = _make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        stage2 = stage2_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_stage2_waterfall(ds, stage2=stage2, ping_offset=1000)
        try:
            bottom_line = next(line for line in ax.get_lines() if line.get_label() == "first bottom return")
            # Real bug this guards: before ping_offset existed on this
            # function, the traced line was always drawn at a 0-based
            # np.arange(n_pings) regardless of what offset the waterfall
            # image itself was given -- landing this at 1000 (not 0)
            # confirms the trace and the image now agree.
            assert bottom_line.get_ydata()[0] == 1000
            assert ax.get_ylim()[1] >= 1000 or ax.get_ylim()[0] >= 1000  # y-axis inverted, either bound could be the max
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


def _make_scaled_spreading_dataset(*, scale: float, n_pings: int = 30, n_samples: int = 300) -> xr.Dataset:
    """Raw data that follows `spreading_only`'s decay shape exactly, but
    amplified by `scale` relative to the literal (scale=1.0) model --
    e.g. `scale=1.6` means the real decay is 1.6x steeper than the
    textbook spreading-loss term alone. Built so `fit_hardware_tvg_curve`'s
    free-fit scale for the `spreading_only` candidate converges close to
    `scale` (a near-perfect free fit) while its fixed-scale (1.0)
    residual stays large -- the two `plot_stage4_residuals` modes should
    then plot visibly different data for the same candidate, which is
    exactly the distinction the 2026-08-25 fix (`use_fixed_scale`)
    exists to make correct."""
    ds = _make_raw_dataset(n_pings=n_pings, n_samples=n_samples)
    ds = compute_geometric_ranges(ds)
    params = TVGParams(spreading_loss_db=20.0, absorption_db_per_m=0.0)
    ground_range = ds["ground_range_m"].values
    loss_db = scale * tvg_gain_db(np.clip(ground_range, params.min_range_m, None), params)
    ref = np.nanmin(loss_db[np.isfinite(loss_db)])
    amplitude = 10.0 ** (-(loss_db - ref) / 20.0)
    amplitude = np.nan_to_num(amplitude, nan=0.0).astype(np.float32)
    ds["backscatter_raw"] = (["ping", "channel", "sample"], amplitude)
    return ds


class TestPlotStage4Residuals:
    """Regression coverage for the real bug a user caught 2026-08-25:
    this plot kept using the free-fit scale/RMSE after `plot_candidate_fit`
    (decay_fit.png) was fixed to fixed-scale ranking the day before --
    residuals.png and decay_fit.png silently stopped telling the same
    story. This function had zero test coverage before this fix, which
    is presumably how the drift went unnoticed."""

    def test_no_fit_result_shows_placeholder_title_not_crash(self):
        stage4 = StageResult(stage="stage4_decay", metrics={}, data={})
        fig, ax = plot_stage4_residuals(stage4)
        try:
            assert ax.get_title() == "No fit available"
        finally:
            plt.close(fig)

    def test_fixed_scale_default_uses_fixed_scale_rmse_in_label(self):
        """A candidate whose free-fit scale is meaningfully off from 1.0
        (by construction here) must show `fixed_scale_rmse_db` in its
        legend label under the default `use_fixed_scale=True`, not the
        free-fit `rmse_db` -- the two are picked to differ substantially
        by this fixture's own construction, so this isn't a coincidental
        pass."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        stage4 = stage4_decay(ds, bin_width_m=2.0)
        spreading = next(c for c in stage4.data["fit_result"].candidates if c.name == "spreading_only")
        assert abs(spreading.scale - 1.6) < 0.3  # sanity: free fit actually found the injected scale
        assert spreading.fixed_scale_rmse_db > spreading.rmse_db + 1.0  # and the two RMSEs meaningfully differ

        fig, ax = plot_stage4_residuals(stage4)
        try:
            labels = [line.get_label() for line in ax.get_lines() if "spreading_only" in line.get_label()]
            assert len(labels) == 1
            assert f"{spreading.fixed_scale_rmse_db:.2f}dB" in labels[0]
            assert f"{spreading.rmse_db:.2f}dB" not in labels[0]
        finally:
            plt.close(fig)

    def test_fixed_vs_free_fit_scale_plot_different_residual_values(self):
        """Not just the label -- the actual plotted y-data must change
        too, confirming `model_loss` is really computed at scale=1.0 by
        default rather than `candidate.scale`."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        stage4 = stage4_decay(ds, bin_width_m=2.0)

        fig_fixed, ax_fixed = plot_stage4_residuals(stage4, use_fixed_scale=True)
        fig_free, ax_free = plot_stage4_residuals(stage4, use_fixed_scale=False)
        try:
            def _spreading_ydata(ax):
                line = next(l for l in ax.get_lines() if "spreading_only" in l.get_label())
                return np.asarray(line.get_ydata())

            fixed_y = _spreading_ydata(ax_fixed)
            free_y = _spreading_ydata(ax_free)
            assert not np.allclose(fixed_y, free_y)
            # Free fit (scale close to the injected 1.6) should track the
            # observed curve much more closely than the fixed-scale (1.0)
            # residual, which is deliberately under-scaled by this fixture.
            assert np.sqrt(np.mean(fixed_y**2)) > np.sqrt(np.mean(free_y**2))
        finally:
            plt.close(fig_fixed)
            plt.close(fig_free)

    def test_relative_x_false_shifts_xdata_by_r_ref(self):
        """relative_x=False must produce absolute-range x-data, matching
        plot_candidate_fit's current convention -- needed for a genuinely
        aligned shared x-axis when the two are stacked in one figure."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        stage4 = stage4_decay(ds, bin_width_m=2.0)
        spreading = next(c for c in stage4.data["fit_result"].candidates if c.name == "spreading_only")

        fig_rel, ax_rel = plot_stage4_residuals(stage4, relative_x=True)
        fig_abs, ax_abs = plot_stage4_residuals(stage4, relative_x=False)
        try:
            def _spreading_xdata(ax):
                line = next(l for l in ax.get_lines() if "spreading_only" in l.get_label())
                return np.asarray(line.get_xdata())

            rel_x = _spreading_xdata(ax_rel)
            abs_x = _spreading_xdata(ax_abs)
            assert np.allclose(abs_x - rel_x, spreading.r_ref_m)
        finally:
            plt.close(fig_rel)
            plt.close(fig_abs)

    def test_ax_param_draws_into_given_axes_not_a_new_figure(self):
        """For combining with plot_candidate_fit in one stacked figure
        (fit on top, residuals below, sharing an x-axis)."""
        ds = _make_scaled_spreading_dataset(scale=1.6)
        stage4 = stage4_decay(ds, bin_width_m=2.0)

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        try:
            returned_fig, returned_ax = plot_stage4_residuals(stage4, ax=ax_bottom)
            assert returned_fig is fig
            assert returned_ax is ax_bottom
            assert len(ax_bottom.get_lines()) > 0
            assert len(ax_top.get_lines()) == 0  # untouched
        finally:
            plt.close(fig)
