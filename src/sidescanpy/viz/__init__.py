"""
Visualization utilities for SideScan sonar data.

Functions return matplotlib Axes (or Figure, Axes) so users can compose
or customize them further.

Public functions
----------------
- :func:`plot_waterfall` — port+starboard waterfall image with configurable
  x-axis (sample, ground_range, across_track, slant_range) and y-axis (ping, time,
  along_track) modes.
- :func:`plot_survey_overview` — compressed, whole-survey waterfall
  (strided ping axis) with optional break markers at caller-supplied
  ping indices, e.g. file boundaries in a multi-file concatenated survey.
- :func:`plot_gridded_raster` — 2D heatmap of gridded backscatter.
- :func:`plot_gridded_with_coverage` — gridded raster with transparency
  showing sample count per cell.
- :func:`plot_gridded_multiple_methods` — side-by-side comparison of
  resampling methods.
- :func:`export_gridded_png` — save gridded raster to PNG.
- ``along_track_distance_m`` (deprecated; moved to
  ``sidescanpy.processing.navigation``) — cumulative along-track distance
  (meters) from per-ping lat/lon; the primitive behind
  ``y_axis="along_track"``, exposed for non-plotting callers (e.g.
  distance-based segmentation).
- :func:`resolve_waterfall_axes` — pick x/y axis mode for a correction's
  step set (geometric corrections always render in real-world distance).
- :func:`plot_track_and_scanlines` — full-survey nadir track plus
  periodic port/starboard scan-line rungs, in a projected (UTM by
  default) coordinate system. Geometry/coverage only, no backscatter.
- :func:`export_track_and_scanlines_png` — save that plot to PNG.
- :func:`build_survey_track` — build the lat/lon track + scan-line
  arrays ``plot_track_and_scanlines`` needs from a multi-file survey dict.
- :func:`plot_track_with_targets` — track + scan-lines with target
  catalog positions overlaid.
- :func:`export_track_with_targets_png` — save that plot to PNG.
- :mod:`sidescanpy.viz.analysis` — one plot per analysis result
  (value histograms, outlier comparison, nadir zoom/waterfall, beam
  geometry, usable range, segments overview, navigation summary).
- :mod:`sidescanpy.viz.tvg` — TVG / range-decay plots (mean vs. range,
  candidate fits, decay fit and residuals, fit-window limits).

Future:
- ``plot_geographic`` — map-style scatter on lat/lon (deferred).

2026 rshomberg@uri.edu
"""

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
from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context
from sidescanpy.viz.gridded import (
    export_gridded_png,
    plot_gridded_multiple_methods,
    plot_gridded_raster,
    plot_gridded_with_coverage,
)
from sidescanpy.viz.targets import export_track_with_targets_png, plot_track_with_targets
from sidescanpy.viz.track import (
    build_survey_track,
    export_track_and_scanlines_png,
    plot_track_and_scanlines,
)
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
    "plot_gridded_with_coverage",
    "plot_gridded_multiple_methods",
    "export_gridded_png",
    "plot_track_and_scanlines",
    "export_track_and_scanlines_png",
    "build_survey_track",
    "contact_sheet",
    "render_target_context",
    "plot_track_with_targets",
    "export_track_with_targets_png",
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


def __getattr__(name: str):
    # Deprecated shim, kept for one release.
    if name == "along_track_distance_m":
        import warnings

        from sidescanpy.processing.navigation import along_track_distance_m

        warnings.warn(
            "sidescanpy.viz.along_track_distance_m has moved to "
            "sidescanpy.processing.navigation",
            DeprecationWarning,
            stacklevel=2,
        )
        return along_track_distance_m
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
