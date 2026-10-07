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
- :func:`along_track_distance_m`, :func:`fill_nav_stalls` — cumulative
  along-track distance from ping lat/lon, and nav-stall repair.
- :func:`apply_tvg` — apply time-varying gain (spreading + absorption)
  to backscatter. Converts raw linear amplitude to dB.
- :func:`apply_tvg_to_amplitude` — same TVG correction as
  :func:`apply_tvg`, but on a plain ``(..., n_samples)`` array rather
  than a full ``xr.Dataset`` — for data that isn't a survey channel
  (e.g. a target catalog's own decoded crop window).
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

from sidescanpy.processing.farrange import (
    crop_far_range_noise,
    detect_far_range_cutoff_sample,
)
from sidescanpy.processing.geometric import (
    compute_geometric_ranges,
)
from sidescanpy.processing.georeferencing import compute_georeferencing
from sidescanpy.processing.interpolate import (
    DEFAULT_FIELD_SOURCES,
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
)
from sidescanpy.processing.navigation import (
    along_track_distance_m,
    fill_nav_stalls,
)
from sidescanpy.processing.pipeline import (
    VALID_PROCESS_LEVELS,
    VALID_STEPS,
    apply_correction_steps,
    run_correction_steps,
    run_pipeline,
)
from sidescanpy.processing.radiometric import (
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
    apply_tvg_to_amplitude,
    tvg_gain_db,
)
from sidescanpy.processing.slicing import (
    slice_by_bbox,
    slice_by_ground_range,
    slice_by_pings,
    slice_by_samples,
    slice_by_time,
)

__all__ = [
    "along_track_distance_m",
    "fill_nav_stalls",
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "apply_tvg",
    "apply_tvg_to_amplitude",
    "tvg_gain_db",
    "TVGParams",
    "absorption_for_frequency",
    "DEFAULT_FIELD_SOURCES",
    "compute_geometric_ranges",
    "compute_georeferencing",
    "detect_far_range_cutoff_sample",
    "crop_far_range_noise",
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
