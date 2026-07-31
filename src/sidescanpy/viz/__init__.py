"""
Visualization utilities for SideScan sonar data.

Functions return matplotlib Axes (or Figure, Axes) so users can compose
or customize them further.

Public functions
----------------
- :func:`plot_waterfall` — port+starboard waterfall image with configurable
  x-axis (sample, ground_range, across_track) and y-axis (ping, time,
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
- :func:`along_track_distance_m` — cumulative along-track distance
  (meters) from per-ping lat/lon; the primitive behind
  ``y_axis="along_track"``, exposed for non-plotting callers (e.g.
  distance-based segmentation).
- :func:`plot_track_and_scanlines` — full-survey nadir track plus
  periodic port/starboard scan-line rungs, in a projected (UTM by
  default) coordinate system. Geometry/coverage only, no backscatter.
- :func:`export_track_and_scanlines_png` — save that plot to PNG.

Future:
- ``plot_geographic`` — map-style scatter on lat/lon (deferred).

2026 rshomberg@uri.edu
"""

from sidescanpy.viz.waterfall import plot_waterfall, plot_survey_overview, along_track_distance_m
from sidescanpy.viz.gridded import (
    plot_gridded_raster,
    plot_gridded_with_coverage,
    plot_gridded_multiple_methods,
    export_gridded_png,
)
from sidescanpy.viz.track import plot_track_and_scanlines, export_track_and_scanlines_png
from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context

__all__ = [
    "plot_waterfall",
    "plot_survey_overview",
    "along_track_distance_m",
    "plot_gridded_raster",
    "plot_gridded_with_coverage",
    "plot_gridded_multiple_methods",
    "export_gridded_png",
    "plot_track_and_scanlines",
    "export_track_and_scanlines_png",
    "contact_sheet",
    "render_target_context",
]