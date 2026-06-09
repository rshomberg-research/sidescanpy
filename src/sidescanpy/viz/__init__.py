"""
Visualization utilities for SideScan sonar data.

Functions return matplotlib Axes (or Figure, Axes) so users can compose
or customize them further.

Public functions
----------------
- :func:`plot_waterfall` — port+starboard waterfall image with configurable
  x-axis (sample, ground_range, across_track) and y-axis (ping, time,
  along_track) modes.

Future:
- ``plot_geographic`` — map-style scatter on lat/lon (deferred).

2026 rshomberg@uri.edu
"""

from sidescanpy.viz.waterfall import plot_waterfall

__all__ = ["plot_waterfall"]