"""
Thin plotting helpers for SideScan sonar data.

Each function renders one result onto a matplotlib Axes (creating a figure
when none is given) and returns the Axes or ``(Figure, Axes)`` so callers
can compose or customize it. Nothing here saves files or runs
corrections: saving, multi-panel layouts and workflow belong to the
caller (see ``examples/`` and ``examples/figures/``). Requires the
``viz`` extra (matplotlib).

Public functions
----------------
- :func:`plot_waterfall` — port+starboard waterfall image with configurable
  x-axis (sample, ground_range, across_track, slant_range) and y-axis (ping, time,
  along_track) modes.
- :func:`resolve_waterfall_axes` — pick x/y axis mode for a correction's
  step set (geometric corrections always render in real-world distance).
- :func:`plot_survey_overview` — compressed, whole-survey waterfall
  (strided ping axis) with optional break markers at caller-supplied
  ping indices, e.g. file boundaries in a multi-file concatenated survey.
- :func:`plot_gridded_raster` — 2D heatmap of one gridded raster.
- :func:`plot_track_and_scanlines` — full-survey nadir track plus
  periodic port/starboard scan-line rungs, in a projected (UTM by
  default) coordinate system. Geometry/coverage only, no backscatter.
  The track comes from :func:`sidescanpy.processing.build_survey_track`.
- :func:`plot_track_with_targets` — the same, with target catalog
  positions overlaid.
- :func:`plot_target_position_scatter` — plain lat/lon scatter of targets.
- :func:`contact_sheet` — grid of image thumbnails, one Figure.
- :func:`render_target_context` — one window image with a box at a target.
- :mod:`sidescanpy.viz.analysis` — one plot per analysis result
  (value histograms, outlier comparison, nadir zoom/waterfall, beam
  geometry, usable range, segments overview, navigation summary).
- :mod:`sidescanpy.viz.tvg` — TVG / range-decay plots (mean vs. range,
  candidate fits, decay fit and residuals, fit-window limits).
- :func:`resolve_cmap` — a colormap that renders NaN black, the
  convention every sidescanpy plot follows.
- :mod:`sidescanpy.viz.scalebar`, :mod:`sidescanpy.viz.basemap` (``maps``
  extra) — overlays used by the map-style plots.

2026 rshomberg@uri.edu
"""

from sidescanpy.viz._cmap import resolve_cmap
from sidescanpy.viz.analysis import (
    plot_beam_geometry,
    plot_cross_track_profile,
    plot_nadir_waterfall,
    plot_nadir_zoom,
    plot_navigation_summary,
    plot_outlier_comparison,
    plot_sample_ping,
    plot_segments_overview,
    plot_shadow_proposals,
    plot_units_distribution,
    plot_usable_range,
    plot_value_histograms,
)
from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context
from sidescanpy.viz.gridded import plot_gridded_raster
from sidescanpy.viz.targets import plot_target_position_scatter, plot_track_with_targets
from sidescanpy.viz.track import plot_track_and_scanlines
from sidescanpy.viz.tvg import (
    plot_altitude_bin_comparison,
    plot_candidate_fit,
    plot_mean_vs_range,
    plot_multi_frequency_comparison,
    plot_per_ping_profiles,
    plot_range_decay_fit,
    plot_range_decay_residuals,
    plot_tvg_gain_curve,
    set_fit_window_limits,
)
from sidescanpy.viz.waterfall import (
    plot_survey_overview,
    plot_waterfall,
    resolve_waterfall_axes,
)

__all__ = [
    "plot_waterfall",
    "plot_survey_overview",
    "resolve_waterfall_axes",
    "plot_gridded_raster",
    "plot_track_and_scanlines",
    "plot_track_with_targets",
    "plot_target_position_scatter",
    "contact_sheet",
    "render_target_context",
    "resolve_cmap",
    # analysis results
    "plot_value_histograms",
    "plot_units_distribution",
    "plot_sample_ping",
    "plot_outlier_comparison",
    "plot_cross_track_profile",
    "plot_nadir_zoom",
    "plot_nadir_waterfall",
    "plot_beam_geometry",
    "plot_usable_range",
    "plot_segments_overview",
    "plot_navigation_summary",
    "plot_shadow_proposals",
    # TVG / range decay
    "plot_mean_vs_range",
    "plot_tvg_gain_curve",
    "plot_per_ping_profiles",
    "plot_multi_frequency_comparison",
    "plot_altitude_bin_comparison",
    "plot_candidate_fit",
    "plot_range_decay_fit",
    "plot_range_decay_residuals",
    "set_fit_window_limits",
]

