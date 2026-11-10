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
- :func:`held_value_fixes` — reduce a sample-and-hold aux record (the
  latest fix repeated at every ping or packet) to its real fixes; the
  ``interpolate_aux`` step does this by default.
- :func:`aux_update_events` — change and update-event statistics of a
  sample-and-hold record, for choosing ``groups`` and ``merge_window_s``.
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
- :func:`francois_garrison_absorption_db_per_m` / :func:`absorption_for_frequency`
  — seawater absorption (Francois-Garrison)
  defaults for typical coastal seawater.
- :func:`correct`, :func:`correct_survey`, :func:`correct_file` — the one
  correction API: an ordered list of built-in or caller-supplied steps,
  each skippable, idempotent and recorded in ``processing_history``
  (see :mod:`sidescanpy.processing.correct`).
- :func:`invert`, :func:`uncorrected_backscatter` — undo recorded steps
  from their history records.
- :func:`processing_level` — derived from ``processing_history``.
- Bottom / nadir handling (:mod:`~sidescanpy.processing.bottom`):
  :func:`detect_first_bottom_return`, :func:`altitude_from_bottom_track`
  (the ``"bottom_track"`` step), :func:`detect_nadir_noise_extent`,
  :func:`blank_nadir_noise`, :func:`mask_nadir_gap`,
  :func:`uniform_exclusion_boundary` and helpers.
- :func:`apply_layback` — move ship positions astern to the towfish (the
  ``"layback"`` step, :mod:`~sidescanpy.processing.layback`).
- :func:`reject_range_outliers` — per-range-bin MAD outlier rejection.
- :func:`mask_bright_ping_rows` — mask whole outlier-bright ping rows.
- Segments (:mod:`~sidescanpy.processing.segments`): :func:`find_segments`,
  :func:`split_by_segments`, :class:`Segment` and the individual detectors.
- :func:`backscatter_vs_range` / :func:`backscatter_vs_sample` — binned
  backscatter profiles; :func:`median_level_profile` — median level per
  range bin in dB, for the TVG model fit.
- :func:`build_survey_track` / :class:`SurveyTrack` — nadir track plus
  strided swath-edge positions across a multi-file survey.

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
    STEP_BOTTOM_TRACK,
    STEP_CROP_FARRANGE,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_INTERPOLATE_AUX,
    STEP_LAYBACK,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    processing_level,
)
from sidescanpy.processing.aux_fixes import (
    POSITION_GROUP,
    AuxUpdateStats,
    aux_update_events,
    held_value_fixes,
)
from sidescanpy.processing.bottom import (
    altitude_from_bottom_track,
    altitude_sample_index,
    blank_nadir_noise,
    detect_first_bottom_return,
    detect_nadir_noise_extent,
    mask_before_sample,
    mask_nadir_gap,
    reject_late_boundaries,
    smooth_boundary_across_pings,
    uniform_exclusion_boundary,
)
from sidescanpy.processing.correct import (
    DEFAULT_STEPS,
    Step,
    correct,
    correct_file,
    correct_survey,
    correction_steps,
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
    LEGACY_RAW_VAR,
    UNCORRECTED_VAR,
    invert,
    uncorrected_backscatter,
    with_uncorrected_var,
)
from sidescanpy.processing.layback import apply_layback
from sidescanpy.processing.navigation import (
    along_track_distance_m,
    course_over_ground,
    fill_nav_stalls,
    smooth_navigation,
)
from sidescanpy.processing.outliers import mask_bright_ping_rows, reject_range_outliers
from sidescanpy.processing.profiles import (
    backscatter_vs_range,
    backscatter_vs_sample,
    median_level_profile,
)
from sidescanpy.processing.radiometric import (
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
    apply_tvg_to_amplitude,
    center_frequency_hz,
    francois_garrison_absorption_db_per_m,
    tvg_gain_db,
)
from sidescanpy.processing.segments import (
    Segment,
    detect_gaps,
    detect_garbage_runs,
    detect_setting_changes,
    detect_unstable_spans,
    find_segments,
    score_stability,
    split_by_segments,
    write_segments,
)
from sidescanpy.processing.slicing import (
    ping_windows,
    slice_by_bbox,
    slice_by_ground_range,
    slice_by_pings,
    slice_by_range,
    slice_by_samples,
    slice_by_time,
    slice_survey,
)
from sidescanpy.processing.track import SurveyTrack, build_survey_track

__all__ = [
    "SurveyTrack",
    "build_survey_track",
    "along_track_distance_m",
    "fill_nav_stalls",
    "course_over_ground",
    "smooth_navigation",
    "interpolate_aux_onto_pings",
    "interpolate_onto_channel",
    "held_value_fixes",
    "aux_update_events",
    "AuxUpdateStats",
    "POSITION_GROUP",
    "apply_tvg",
    "apply_tvg_to_amplitude",
    "tvg_gain_db",
    "TVGParams",
    "absorption_for_frequency",
    "center_frequency_hz",
    "francois_garrison_absorption_db_per_m",
    "DEFAULT_FIELD_SOURCES",
    "compute_geometric_ranges",
    "compute_georeferencing",
    "apply_layback",
    "detect_far_range_cutoff_sample",
    "crop_far_range_noise",
    "ping_windows",
    "slice_by_pings",
    "slice_by_samples",
    "slice_by_range",
    "slice_survey",
    "slice_by_time",
    "slice_by_bbox",
    "slice_by_ground_range",
    "correct",
    "correct_survey",
    "correction_steps",
    "correct_file",
    "estimate_correct_disk_bytes",
    "Step",
    "DEFAULT_STEPS",
    "BUILTIN_STEPS",
    "STEP_CROP_FARRANGE",
    "STEP_INTERPOLATE_AUX",
    "STEP_BOTTOM_TRACK",
    "STEP_LAYBACK",
    "STEP_TVG",
    "STEP_GEOMETRIC",
    "STEP_RANGE_CORRECTION",
    "STEP_GEOREF",
    "invert",
    "uncorrected_backscatter",
    "with_uncorrected_var",
    "LEGACY_RAW_VAR",
    "UNCORRECTED_VAR",
    "INVERTIBLE_STEPS",
    "processing_level",
    "crop_far_range",
    "interpolate_aux_onto_dataset",
    # bottom / nadir
    "detect_first_bottom_return",
    "altitude_from_bottom_track",
    "smooth_boundary_across_pings",
    "detect_nadir_noise_extent",
    "blank_nadir_noise",
    "mask_before_sample",
    "altitude_sample_index",
    "uniform_exclusion_boundary",
    "mask_nadir_gap",
    "reject_late_boundaries",
    # outliers
    "reject_range_outliers",
    "mask_bright_ping_rows",
    # segments
    "Segment",
    "detect_gaps",
    "detect_setting_changes",
    "detect_garbage_runs",
    "score_stability",
    "detect_unstable_spans",
    "find_segments",
    "split_by_segments",
    "write_segments",
    # profiles
    "backscatter_vs_range",
    "backscatter_vs_sample",
    "median_level_profile",
]
