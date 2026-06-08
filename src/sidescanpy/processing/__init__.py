"""
Processing functions for SideScan sonar surveys.

Each function takes a validated input dataset, applies a transform, and
returns a new dataset with updated values, ``processing_level``, and
``processing_history``. Functions are pure (no in-place mutation by
default) unless explicitly marked otherwise.

Public entry points
-------------------
- :func:`interpolate_aux_onto_pings` — replace per-ping nav, attitude,
  and heave fields with values interpolated from the survey's auxiliary
  records onto each ping's timestamp.
- :func:`interpolate_onto_channel` — lower-level: interpolate any
  ``xr.Dataset`` (e.g. an external GPS log) onto one channel's ping
  times.
- :func:`apply_tvg` — apply time-varying gain (spreading + absorption)
  to backscatter. Converts raw linear amplitude to dB.
- :func:`absorption_for_frequency` — frequency-dependent absorption
  defaults for typical coastal seawater.

Conventions
-----------
- Each function asserts the expected input ``processing_level`` and
  raises ``ValueError`` if it doesn't match. Prevents accidental
  double-application.
- ``processing_history`` is appended on every transform via
  ``ds.sidescan.append_to_history(...)``.
- ``backscatter_raw`` is treated as immutable across the pipeline. Only
  ``backscatter_intensity`` is updated by transforms.

See ``docs/processing.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.processing.interpolate import (
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
    DEFAULT_FIELD_SOURCES,
)
from sidescanpy.processing.radiometric import (
    apply_tvg,
    TVGParams,
    absorption_for_frequency,
)

__all__ = [
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "apply_tvg",
    "TVGParams",
    "absorption_for_frequency",
    "DEFAULT_FIELD_SOURCES",
]