"""
Tests for sidescanpy.viz.analysis: one plot per analysis result.

2026 rshomberg@uri.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from sidescanpy.analysis.beam import estimate_beam_geometry
from sidescanpy.analysis.nadir import analyze_nadir
from sidescanpy.analysis.navigation import summarize_navigation
from sidescanpy.analysis.outliers import compare_outlier_rejection
from sidescanpy.analysis.usable_range import UsableRangeResult, estimate_usable_range
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.segments import Segment
from sidescanpy.viz.analysis import (
    plot_beam_geometry,
    plot_cross_track_profile,
    plot_nadir_waterfall,
    plot_nadir_zoom,
    plot_navigation_summary,
    plot_outlier_comparison,
    plot_sample_ping,
    plot_segments_overview,
    plot_units_distribution,
    plot_usable_range,
    plot_value_histograms,
)
from tests.unit.synthetic import make_raw_dataset


def _decaying_ds(n_pings: int = 10, n_samples: int = 300) -> xr.Dataset:
    rng = np.random.default_rng(0)
    decay = 100.0 * np.exp(-np.arange(n_samples) / 80.0) + 1.0
    data = decay[None, None, :] * (1.0 + 0.05 * rng.standard_normal((n_pings, 2, n_samples)))
    return compute_geometric_ranges(make_raw_dataset(backscatter=data.astype(np.float32)))


class TestPlotBeamGeometry:
    def test_far_ray_drawn_on_both_sides(self):
        """Regression test for the real bug: usable_range_m used to be
        stored unsigned for both port/starboard far_ray entries, so the
        top-panel diagram only ever drew a far ray toward the
        (arbitrarily) positive/starboard side, twice. After the sign
        fix, the top-panel axis must show real line data at both a
        negative x (port) and a positive x (starboard)."""
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)
        usable = UsableRangeResult(usable_range_m=25.0)
        beam = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable.usable_range_m)

        fig, (ax_geo, ax_ping) = plot_beam_geometry(ds, beam)
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
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)
        usable = UsableRangeResult(usable_range_m=25.0)
        beam = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable.usable_range_m)

        fig, (ax_geo, ax_ping) = plot_beam_geometry(ds, beam)
        try:
            patches = ax_geo.patches
            assert len(patches) >= 2, "expected at least one filled wedge per side"
            xs = [v for p in patches for v in p.get_xy()[:, 0]]
            assert any(x < -1.0 for x in xs)
            assert any(x > 1.0 for x in xs)
        finally:
            plt.close(fig)

    def test_title_has_no_step_number(self):
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)
        usable = UsableRangeResult(usable_range_m=25.0)
        beam = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable.usable_range_m)

        fig, (ax_geo, ax_ping) = plot_beam_geometry(ds, beam)
        try:
            assert "Stage" not in ax_geo.get_title()
        finally:
            plt.close(fig)


class TestPlotNadirWaterfall:
    """ping_offset added 2026-08-31 -- a caller plotting an already-sliced
    window (e.g. one section of a larger survey) needs the traced
    boundary lines to land on the same real ping numbers as the
    underlying waterfall image, not the image's own 0-based offset and
    the trace's separate, always-0-based one silently disagreeing."""

    def test_default_ping_offset_is_zero_based_unchanged(self):
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_nadir_waterfall(ds, nadir=nadir)
        try:
            bottom_line = next(line for line in ax.get_lines() if line.get_label() == "first bottom return")
            assert bottom_line.get_ydata()[0] == 0
        finally:
            plt.close(fig)

    def test_ping_offset_shifts_both_image_and_traced_boundary(self):
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_nadir_waterfall(ds, nadir=nadir, ping_offset=1000)
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


class TestPlotNadirZoom:
    def test_water_column_span_covers_both_sides_to_first_bottom_return(self):
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)
        boundary = nadir.first_bottom_return_sample
        assert boundary is not None

        fig, ax = plot_nadir_zoom(ds, nadir=nadir, nadir_n_samples=5)
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
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        nadir = analyze_nadir(ds, nadir_n_samples=5)

        fig, ax = plot_nadir_zoom(ds, nadir=nadir, nadir_n_samples=5)
        try:
            labels = [l.get_label() for l in ax.get_lines()]
            assert "altitude" in labels
            assert "first bottom return" in labels
        finally:
            plt.close(fig)

    def test_title_has_no_step_number(self):
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        fig, ax = plot_nadir_zoom(ds)
        try:
            assert "Stage" not in ax.get_title()
        finally:
            plt.close(fig)

    def test_falls_back_to_nadir_window_span_without_nadir_result(self):
        """No nadir result given -- still shades something labeled "water
        column returns" (the raw blanking window), doesn't crash or
        silently omit the shading."""
        ds = make_raw_dataset(n_pings=5, n_samples=100)
        ds = compute_geometric_ranges(ds)
        fig, ax = plot_nadir_zoom(ds, nadir_n_samples=5)
        try:
            spans = [p for p in ax.patches if p.get_label() == "water column returns"]
            assert len(spans) == 1
        finally:
            plt.close(fig)


def _make_nav_ds(*, n_pings: int = 10, **overrides) -> xr.Dataset:
    data_vars = {
        "heading_deg": (["ping"], np.linspace(10.0, 100.0, n_pings, dtype=np.float64)),
        "pitch_deg": (["ping"], np.zeros(n_pings)),
        "roll_deg": (["ping"], np.zeros(n_pings)),
        "altitude_m": (["ping"], np.full(n_pings, 5.0)),
        "depth_m": (["ping"], np.full(n_pings, 20.0)),
        "heave_m": (["ping"], np.zeros(n_pings)),
        "latitude": (["ping"], np.linspace(40.0, 40.01, n_pings)),
        "longitude": (["ping"], np.linspace(-70.0, -69.99, n_pings)),
    }
    data_vars.update(overrides)
    return xr.Dataset(data_vars)


class TestPlotNavigationSummary:
    def test_returns_four_axes(self):
        ds = _make_nav_ds()
        fig, axes = plot_navigation_summary(ds)
        assert len(axes) == 4
        plt.close(fig)

    def test_ping_offset_shifts_x_axis(self):
        ds = _make_nav_ds(n_pings=5)
        fig, (ax_heading, _, _, _) = plot_navigation_summary(ds, ping_offset=1000)
        # the heading trace is plotted last, after the convention axhlines
        xdata = ax_heading.lines[-1].get_xdata()
        assert xdata[0] == 1000
        assert xdata[-1] == 1004
        plt.close(fig)

    def test_out_of_range_heading_gets_marked(self):
        ds = _make_nav_ds(n_pings=3, heading_deg=(["ping"], np.array([-400.0, -300.0, -250.0])))
        result = summarize_navigation(ds)
        fig, (ax_heading, _, _, _) = plot_navigation_summary(ds, result=result)
        # a scatter collection was added for the out-of-range points
        assert len(ax_heading.collections) >= 1
        plt.close(fig)

    def test_no_out_of_range_marker_when_heading_in_range(self):
        ds = _make_nav_ds(n_pings=5, heading_deg=(["ping"], np.array([0.0, 90.0, 180.0, 270.0, 359.9])))
        result = summarize_navigation(ds)
        fig, (ax_heading, _, _, _) = plot_navigation_summary(ds, result=result)
        assert len(ax_heading.collections) == 0
        plt.close(fig)

    def test_missing_position_vars_does_not_raise(self):
        ds = _make_nav_ds().drop_vars(["latitude", "longitude"])
        fig, (_, _, _, ax_track) = plot_navigation_summary(ds)
        assert len(ax_track.lines) == 0
        plt.close(fig)

    def test_missing_attitude_vars_does_not_raise(self):
        ds = _make_nav_ds().drop_vars(["pitch_deg", "roll_deg"])
        fig, axes = plot_navigation_summary(ds)
        assert len(axes) == 4
        plt.close(fig)


class TestSmokeAndAxes:
    """Every plot returns (fig, ax/axes), draws into supplied axes, and
    leaves figure-level decoration to the caller in that case."""

    def test_value_histograms_and_units_distribution_titles(self):
        ds = _decaying_ds()
        fig, axes = plot_value_histograms(ds)
        try:
            assert len(axes) == 2
            assert fig._suptitle.get_text().startswith("Value distribution")
        finally:
            plt.close(fig)
        fig, _ = plot_units_distribution(ds)
        try:
            assert fig._suptitle.get_text().startswith("Distribution shape")
        finally:
            plt.close(fig)

    def test_value_histograms_into_given_axes_skips_suptitle(self):
        fig, axes = plt.subplots(1, 2)
        try:
            returned_fig, returned = plot_value_histograms(_decaying_ds(), axes=axes)
            assert returned_fig is fig
            assert returned[0] is axes[0] and returned[1] is axes[1]
            assert fig._suptitle is None
            assert len(axes[0].patches) > 0
        finally:
            plt.close(fig)

    def test_sample_ping_defaults_to_middle_ping(self):
        fig, (ax_port, ax_stbd) = plot_sample_ping(_decaying_ds(n_pings=10))
        try:
            assert fig._suptitle.get_text() == "Sample ping 5"
            assert ax_port.get_title() == "port" and ax_stbd.get_title() == "starboard"
        finally:
            plt.close(fig)

    def test_outlier_comparison_into_given_axes(self):
        ds = _decaying_ds()
        result = compare_outlier_rejection(ds, bin_width_m=2.0)
        fig, axes = plt.subplots(1, 2)
        try:
            returned_fig, _ = plot_outlier_comparison(ds, result, axes=axes)
            assert returned_fig is fig
            assert len(axes[0].patches) > 0
        finally:
            plt.close(fig)

    def test_cross_track_profile_into_given_ax(self):
        fig, ax = plt.subplots()
        try:
            returned_fig, returned_ax = plot_cross_track_profile(_decaying_ds(), ax=ax)
            assert returned_fig is fig and returned_ax is ax
            xdata = ax.get_lines()[0].get_xdata()
            assert xdata.min() < 0 < xdata.max()
        finally:
            plt.close(fig)

    def test_usable_range_marks_boundary_and_accepts_axes(self):
        ds = _decaying_ds()
        result = estimate_usable_range(ds, bin_width_m=1.0)
        fig, axes = plt.subplots(2, 1)
        try:
            returned_fig, (ax_med, ax_std) = plot_usable_range(result, axes=axes)
            assert returned_fig is fig and ax_med is axes[0]
            assert ax_med.get_title() == "Usable range"
        finally:
            plt.close(fig)

    def test_usable_range_without_bins_shows_placeholder(self):
        fig, (ax_med, _) = plot_usable_range(UsableRangeResult())
        try:
            assert ax_med.get_title() == "No binned data"
        finally:
            plt.close(fig)

    def test_segments_overview_shades_excluded_segment(self):
        ds = _decaying_ds(n_pings=10)
        segments = [
            Segment(label="section1", start=0, stop=5, status="clean", reasons=[]),
            Segment(label="excluded1", start=5, stop=10, status="excluded", reasons=["test"]),
        ]
        fig, (ax_alt, ax_pitch) = plot_segments_overview(ds, segments)
        try:
            assert len(ax_alt.patches) == 1 and len(ax_pitch.patches) == 1
        finally:
            plt.close(fig)

    def test_beam_geometry_into_given_axes(self):
        ds = compute_geometric_ranges(make_raw_dataset(n_pings=5, n_samples=100))
        nadir = analyze_nadir(ds, nadir_n_samples=5)
        beam = estimate_beam_geometry(ds, nadir.first_bottom_return_sample, 25.0)
        fig, axes = plt.subplots(2, 1)
        try:
            returned_fig, (ax_geo, _) = plot_beam_geometry(ds, beam, axes=axes)
            assert returned_fig is fig and ax_geo is axes[0]
            assert ax_geo.get_title().startswith("Beam geometry")
        finally:
            plt.close(fig)

    def test_navigation_summary_into_given_axes(self):
        fig, axes = plt.subplots(2, 2)
        try:
            returned_fig, returned = plot_navigation_summary(_make_nav_ds(), axes=axes.ravel())
            assert returned_fig is fig
            assert returned[0].get_title() == "heading_deg"
            assert fig._suptitle is None
        finally:
            plt.close(fig)
