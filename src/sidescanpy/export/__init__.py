"""
Dataset-export primitives for SideScan sonar surveys.

Distinct from ``processing/`` (physical corrections: TVG, geometric
ranges, georeferencing) and ``targets/interchange/`` (ML annotation
file formats): this package turns an already-corrected survey channel
into 8-bit training-dataset images — chips or full-channel views,
optionally reprojected into geometric or georeferenced meters — plus
the shared invertible display transform they're rendered through.

Public entry points
-------------------
- :func:`to_uint8` / :func:`from_uint8` — invertible percentile+gamma
  8-bit display transform, with a configurable per-image / precomputed
  (survey- or dataset-wide) normalization scope.
- :func:`compute_reference_stats` — precompute percentile bounds shared
  across many :func:`to_uint8` calls, for survey-/dataset-wide scope.
- :func:`extract_chip` — fixed-size, edge-aware, optionally-jittered
  crop around a point.
- :func:`extract_full_channel` — full-sample-axis window spanning a
  fixed (optionally square-native, to avoid aspect stretch) ping span,
  downsampled to a fixed output size.
- :func:`sample_negative_window` — pick a window location away from
  every cataloged target, for no-target samples.
- :func:`targets_within_window` — every cataloged target whose position
  falls inside an extracted window's extent, for multi-target images.

2026 rshomberg@uri.edu
"""

from sidescanpy.export.eightbit import (
    ReferenceStats,
    ReversalMetadata,
    compute_reference_stats,
    from_uint8,
    to_uint8,
)
from sidescanpy.export.windows import (
    WindowExtent,
    extract_chip,
    extract_full_channel,
    rescale_target_position,
    resize_to,
    sample_negative_window,
    targets_within_window,
)
from sidescanpy.export.geometric_window import extract_geometric_window, extract_geometric_window_binned
from sidescanpy.export.geometric_window import locate_point as locate_point_geometric
from sidescanpy.export.georef_window import extract_georef_window
from sidescanpy.export.georef_window import locate_point as locate_point_georef

__all__ = [
    "ReferenceStats",
    "ReversalMetadata",
    "compute_reference_stats",
    "to_uint8",
    "from_uint8",
    "WindowExtent",
    "extract_chip",
    "extract_full_channel",
    "resize_to",
    "sample_negative_window",
    "targets_within_window",
    "rescale_target_position",
    "extract_geometric_window",
    "extract_geometric_window_binned",
    "extract_georef_window",
    "locate_point_geometric",
    "locate_point_georef",
]
