"""
Visualization utilities for SideScan sonar data.

Functions return matplotlib Axes (or Figure, Axes) so users can compose
or customize them further.

Public functions
----------------
- :func:`plot_waterfall` — port+starboard waterfall image with configurable
  x-axis (sample, ground_range, across_track) and y-axis (ping, time,
  along_track) modes.
- :func:`plot_gridded_raster` — 2D heatmap of gridded backscatter.
- :func:`plot_gridded_with_coverage` — gridded raster with transparency
  showing sample count per cell.
- :func:`plot_gridded_multiple_methods` — side-by-side comparison of
  resampling methods.
- :func:`export_gridded_png` — save gridded raster to PNG.

Future:
- ``plot_geographic`` — map-style scatter on lat/lon (deferred).

2026 rshomberg@uri.edu
"""

from sidescanpy.viz.waterfall import plot_waterfall
from sidescanpy.viz.gridded import (
    plot_gridded_raster,
    plot_gridded_with_coverage,
    plot_gridded_multiple_methods,
    export_gridded_png,
)

__all__ = [
    "plot_waterfall",
    "plot_gridded_raster",
    "plot_gridded_with_coverage",
    "plot_gridded_multiple_methods",
    "export_gridded_png",
]