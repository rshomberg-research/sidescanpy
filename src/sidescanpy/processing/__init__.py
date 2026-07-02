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
- :func:`run_pipeline` — run the processing pipeline through a target
  ladder level (``none/radiometric/geometric/georef``).
- :func:`run_correction_steps` — run an explicit set of correction
  steps on every channel of a survey, for combinations the ladder
  can't express (e.g. geometric ranges without TVG).
- :func:`apply_correction_steps` — same step-set semantics as
  ``run_correction_steps`` but for a single already-aux-interpolated
  channel dataset, skipping the survey-level aux interpolation.

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

from sidescanpy.processing.geometric import (
    compute_geometric_ranges,
)

from sidescanpy.processing.georeferencing import compute_georeferencing

from sidescanpy.processing.slicing import (
    slice_by_pings,
    slice_by_samples,
    slice_by_time,
    slice_by_bbox,
    slice_by_ground_range,
)

from sidescanpy.processing.pipeline import (
    run_pipeline,
    run_correction_steps,
    apply_correction_steps,
    VALID_PROCESS_LEVELS,
    VALID_STEPS,
)

__all__ = [
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "apply_tvg",
    "TVGParams",
    "absorption_for_frequency",
    "DEFAULT_FIELD_SOURCES",
    "compute_geometric_ranges",
    "compute_georeferencing",
    "slice_by_pings",
    "slice_by_samples",
    "slice_by_time",
    "slice_by_bbox",
    "slice_by_ground_range",
    "run_pipeline",
    "run_correction_steps",
    "apply_correction_steps",
    "VALID_PROCESS_LEVELS",
    "VALID_STEPS",
]