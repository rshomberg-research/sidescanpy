"""
Export windows: where an image is cut from a survey, and what is in it.

- :mod:`~sidescanpy.export.windows.pixel` -- sample/ping-space crops
  (chips, full-channel views), co-located targets, and the per-style window
  for the ``uncorrected``/``db``/``radiometric`` render levels.
- :mod:`~sidescanpy.export.windows.levels` -- styles, render levels and
  the backscatter variable each level reads.
- :mod:`~sidescanpy.export.windows.neighborhood` -- bounded,
  block-averaged ping neighborhoods for the real-meters levels.
- :mod:`~sidescanpy.export.windows.geo` -- ``geometric``/``georef``
  windows.
- :mod:`~sidescanpy.export.windows.background` -- screened no-target
  windows.
- :mod:`~sidescanpy.export.windows.native` -- chips and dual-swath tiles
  that are square/scaled in metres on the native ping x sample grid, with
  pixel <-> native mapping (annotation round trips).

2026 rshomberg@uri.edu
"""

from sidescanpy.export.windows.background import (
    BackgroundCandidate,
    BackgroundWindow,
    WindowFootprint,
    nan_fraction,
    sample_background_candidates,
    sample_background_windows,
)
from sidescanpy.export.windows.geo import (
    RESAMPLE_METHODS,
    extract_geo_style_window,
    geo_targets_within_window,
    georef_targets_within_window,
    swath_fill_extent,
)
from sidescanpy.export.windows.levels import (
    CHIP_STYLES,
    LEVEL_VARS,
    PIXEL_LEVELS,
    RENDER_LEVELS,
    WindowSpec,
    mask_dropouts,
    mask_nadir_gap_with_uncorrected,
    nominal_sample_interval_m,
    with_db_var,
    with_uncorrected_var,
)
from sidescanpy.export.windows.native import (
    NativeWindow,
    along_track_spacing_m,
    native_chip,
    swath_tiles,
)
from sidescanpy.export.windows.neighborhood import (
    GeoNeighborhood,
    across_values,
    decimate_target,
    geo_decimation,
    load_geo_neighborhood,
    load_swath_neighborhood,
    neighborhood_ping_spacing_m,
)
from sidescanpy.export.windows.pixel import (
    RenderedWindow,
    WindowExtent,
    combine_channels_mirrored,
    extract_chip,
    extract_full_channel,
    extract_pixel_style_window,
    place_targets_in_file,
    rescale_target_position,
    rescale_targets,
    resize_to,
    sample_negative_window,
    target_combined_col,
    targets_within_window,
    with_bbox_size_px,
)

__all__ = [
    "NativeWindow",
    "along_track_spacing_m",
    "native_chip",
    "swath_tiles",
    "BackgroundCandidate",
    "BackgroundWindow",
    "WindowFootprint",
    "nan_fraction",
    "sample_background_candidates",
    "sample_background_windows",
    "RESAMPLE_METHODS",
    "extract_geo_style_window",
    "geo_targets_within_window",
    "georef_targets_within_window",
    "swath_fill_extent",
    "CHIP_STYLES",
    "LEVEL_VARS",
    "PIXEL_LEVELS",
    "RENDER_LEVELS",
    "WindowSpec",
    "mask_dropouts",
    "mask_nadir_gap_with_uncorrected",
    "nominal_sample_interval_m",
    "with_db_var",
    "with_uncorrected_var",
    "GeoNeighborhood",
    "across_values",
    "decimate_target",
    "geo_decimation",
    "load_geo_neighborhood",
    "load_swath_neighborhood",
    "neighborhood_ping_spacing_m",
    "RenderedWindow",
    "WindowExtent",
    "combine_channels_mirrored",
    "extract_chip",
    "extract_full_channel",
    "extract_pixel_style_window",
    "place_targets_in_file",
    "rescale_target_position",
    "rescale_targets",
    "resize_to",
    "sample_negative_window",
    "target_combined_col",
    "targets_within_window",
    "with_bbox_size_px",
]
