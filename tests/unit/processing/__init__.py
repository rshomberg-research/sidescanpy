"""Unit tests for sidescanpy.processing module."""

from sidescanpy.processing.radiometric import (
    apply_tvg,
    TVGParams,
    absorption_for_frequency,
)
from sidescanpy.processing.interpolate import (
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
    DEFAULT_FIELD_SOURCES,  

)
from sidescanpy.processing.geometric import compute_geometric_ranges

__all__ = [
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "apply_tvg",
    "TVGParams",
    "absorption_for_frequency",
    "compute_geometric_ranges",
    "DEFAULT_FIELD_SOURCES",
]