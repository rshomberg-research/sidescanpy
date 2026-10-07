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
- :func:`correct`, :func:`correct_survey`, :func:`correct_file` — the one
  correction API: an ordered list of built-in or caller-supplied steps,
  each skippable, idempotent and recorded in ``processing_history``
  (see :mod:`sidescanpy.processing.correct`).
- :func:`invert`, :func:`uncorrected_backscatter` — undo recorded steps
  from their history records.
- :func:`processing_level` — derived from ``processing_history``.

Conventions
-----------
- Corrections are applied in place to ``backscatter_intensity``; no raw
  copy is kept (a legacy ``backscatter_raw`` is dropped on rewrite).
- Every correction step appends one ``processing_history`` record with
  the parameters it used; a step already recorded is not re-applied.

See ``docs/processing.md`` for full reference.

2026 rshomberg@uri.edu
"""

from sidescanpy.core.history import (
    BUILTIN_STEPS,
    STEP_CROP_FARRANGE,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_INTERPOLATE_AUX,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    processing_level,
)
from sidescanpy.processing.correct import (
    DEFAULT_STEPS,
    Step,
    correct,
    correct_file,
    correct_survey,
    estimate_correct_disk_bytes,
)
from sidescanpy.processing.farrange import (
    crop_far_range,
    crop_far_range_noise,
    detect_far_range_cutoff_sample,
)
from sidescanpy.processing.geometric import (
    compute_geometric_ranges,
)
from sidescanpy.processing.georeferencing import compute_georeferencing
from sidescanpy.processing.interpolate import (
    DEFAULT_FIELD_SOURCES,
    interpolate_aux_onto_dataset,
    interpolate_aux_onto_pings,
    interpolate_onto_channel,
)
from sidescanpy.processing.inversion import (
    INVERTIBLE_STEPS,
    invert,
    uncorrected_backscatter,
)
from sidescanpy.processing.navigation import (
    along_track_distance_m,
    fill_nav_stalls,
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
    slice_by_range,
    slice_by_samples,
    slice_by_time,
    slice_survey,
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
    "slice_by_range",
    "slice_survey",
    "slice_by_time",
    "slice_by_bbox",
    "slice_by_ground_range",
    "correct",
    "correct_survey",
    "correct_file",
    "estimate_correct_disk_bytes",
    "Step",
    "DEFAULT_STEPS",
    "BUILTIN_STEPS",
    "STEP_CROP_FARRANGE",
    "STEP_INTERPOLATE_AUX",
    "STEP_TVG",
    "STEP_GEOMETRIC",
    "STEP_RANGE_CORRECTION",
    "STEP_GEOREF",
    "invert",
    "uncorrected_backscatter",
    "INVERTIBLE_STEPS",
    "processing_level",
    "crop_far_range",
    "interpolate_aux_onto_dataset",
]
