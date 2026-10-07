"""Unit tests for sidescanpy.processing module."""

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.interpolate import (
    DEFAULT_FIELD_SOURCES,
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
)
from sidescanpy.processing.radiometric import (
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
)

__all__ = [
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "apply_tvg",
    "TVGParams",
    "absorption_for_frequency",
    "compute_geometric_ranges",
    "DEFAULT_FIELD_SOURCES",
]
