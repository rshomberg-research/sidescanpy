"""
Processing module for sidescan sonar data.

Provides radiometric, geometric, and georeferencing corrections.
Each correction stage reads from .ss.h5 files, processes in-memory,
and writes corrected .ss.h5 files with updated processing_level and
processing_history attributes.

Main workflow:
    raw → radiometric → geometric → georeferenced

Each stage is independent and can be run in sequence or individually.
"""

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

__all__ = [
    "apply_tvg",
    "TVGParams",
    "absorption_for_frequency",
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "DEFAULT_FIELD_SOURCES",
]