"""
Dataset-export primitives for SideScan sonar surveys.

Distinct from ``processing/`` (physical corrections: TVG, geometric
ranges, georeferencing) and ``export/ml/`` (ML annotation
file formats): this package turns an already-corrected survey channel
into 8-bit training-dataset images — chips or full-channel views,
optionally reprojected into geometric or georeferenced meters — plus
the shared invertible display transform they're rendered through.

Public entry points
-------------------
- :func:`to_uint8` / :func:`to_uint16` / :func:`from_uint8` — invertible
  percentile+gamma display transform, with a configurable per-image /
  precomputed (survey- or dataset-wide) normalization scope.
- :func:`compute_reference_stats` — precompute percentile bounds shared
  across many :func:`to_uint8` calls, for survey-/dataset-wide scope.
- :mod:`sidescanpy.export.windows` — where an image is cut and what is in
  it: :func:`extract_chip` / :func:`extract_full_channel` (plain arrays),
  :func:`extract_pixel_style_window` / :func:`extract_geo_style_window`
  (one export style at one render level), neighborhood loading, and
  :func:`sample_background_windows` (screened no-target windows).
- :mod:`sidescanpy.export.orientation` — display orientation of exported
  windows and their targets.
- :mod:`sidescanpy.export.augment` — random flips and jitter.
- :mod:`sidescanpy.export.ml` — COCO/YOLO annotation writers.

2026 rshomberg@uri.edu
"""

from sidescanpy.export.augment import (
    augmented_file_name,
    jitter_latlon,
    jitter_offset_m,
    random_flip,
)
from sidescanpy.export.geometric_window import (
    extract_geometric_window,
    extract_geometric_window_binned,
    extract_geometric_window_griddata,
)
from sidescanpy.export.geometric_window import locate_point as locate_point_geometric
from sidescanpy.export.georef_window import extract_georef_window
from sidescanpy.export.georef_window import locate_point as locate_point_georef
from sidescanpy.export.image import (
    ReferenceStats,
    ReversalMetadata,
    compute_reference_stats,
    from_uint8,
    to_uint8,
    to_uint16,
)
from sidescanpy.export.orientation import (
    detrend_by_range,
    flip_geo_vertical,
    orient_for_display,
    orient_geo_window,
    orient_pixel_window,
)
from sidescanpy.export.windows import (
    CHIP_STYLES,
    PIXEL_LEVELS,
    RENDER_LEVELS,
    BackgroundWindow,
    GeoNeighborhood,
    RenderedWindow,
    WindowExtent,
    WindowSpec,
    combine_channels_mirrored,
    extract_chip,
    extract_full_channel,
    extract_geo_style_window,
    extract_pixel_style_window,
    load_geo_neighborhood,
    rescale_target_position,
    resize_to,
    sample_background_windows,
    sample_negative_window,
    targets_within_window,
)

__all__ = [
    "ReferenceStats",
    "ReversalMetadata",
    "compute_reference_stats",
    "to_uint8",
    "to_uint16",
    "from_uint8",
    "augmented_file_name",
    "jitter_latlon",
    "jitter_offset_m",
    "random_flip",
    "detrend_by_range",
    "flip_geo_vertical",
    "orient_for_display",
    "orient_geo_window",
    "orient_pixel_window",
    "CHIP_STYLES",
    "PIXEL_LEVELS",
    "RENDER_LEVELS",
    "BackgroundWindow",
    "GeoNeighborhood",
    "RenderedWindow",
    "WindowExtent",
    "WindowSpec",
    "combine_channels_mirrored",
    "extract_chip",
    "extract_full_channel",
    "extract_geo_style_window",
    "extract_pixel_style_window",
    "load_geo_neighborhood",
    "resize_to",
    "sample_background_windows",
    "sample_negative_window",
    "targets_within_window",
    "rescale_target_position",
    "extract_geometric_window",
    "extract_geometric_window_binned",
    "extract_geometric_window_griddata",
    "extract_georef_window",
    "locate_point_geometric",
    "locate_point_georef",
]
