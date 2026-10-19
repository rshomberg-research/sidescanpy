"""
Processing history: the record of which correction steps were applied.

``ds.attrs["processing_history"]`` is a JSON-encoded list of step records,
appended to by every step that changes a frequency channel dataset::

    {
        "step": "tvg",                      # canonical step name
        "params": {...},                    # actual params used, defaults included
        "function": "sidescanpy.processing.radiometric.apply_tvg",
        "version": "0.5.0",                 # sidescanpy version that applied it
        "timestamp_utc": "2026-10-07T12:00:00+00:00",
    }

The history is what makes corrections idempotent (a recorded step is not
re-applied) and reversible (radiometric steps are undone from their recorded
params). The processing level is derived from it (:func:`processing_level`),
not stored as a separate attribute. The history records what the library did;
it does not describe the input data (units, hardware TVG state) -- that is the
caller's knowledge.

Geometric and georef steps only *add* variables, so for them a present,
not-all-NaN output variable is what counts as "done" (:func:`has_step`).

Legacy files: step names written by older versions (``tvg_correction``,
``compute_geometric_ranges``, ...) are mapped to the canonical names by
:func:`canonical_step`, and a legacy ``processing_level`` attribute is used
only as a fallback when the history records no recognised step.

2026 rshomberg@uri.edu
"""

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)

STEP_CROP_FARRANGE = "crop_farrange"
STEP_INTERPOLATE_AUX = "interpolate_aux"
STEP_BOTTOM_TRACK = "bottom_track"
STEP_TVG = "tvg"
STEP_GEOMETRIC = "geometric"
STEP_RANGE_CORRECTION = "range_correction"
STEP_GEOREF = "georef"

BUILTIN_STEPS: tuple[str, ...] = (
    STEP_CROP_FARRANGE,
    STEP_INTERPOLATE_AUX,
    STEP_BOTTOM_TRACK,
    STEP_TVG,
    STEP_GEOMETRIC,
    STEP_RANGE_CORRECTION,
    STEP_GEOREF,
)
"""Built-in correction steps, in their canonical (dependency) order."""

RADIOMETRIC_STEPS: tuple[str, ...] = (STEP_TVG, STEP_RANGE_CORRECTION)
"""Steps that change ``backscatter_intensity`` values and can be inverted."""

STEP_OUTPUT_VARS: dict[str, tuple[str, ...]] = {
    STEP_GEOMETRIC: (
        "slant_range_m",
        "ground_range_m",
        "across_track_m",
        "along_track_m",
        "vertical_offset_m",
        "depth_per_sample_m",
    ),
    STEP_GEOREF: ("latitude_per_sample", "longitude_per_sample"),
}
"""Variables added by the geometric/georef steps. The first one listed is the
variable checked by :func:`has_step`."""

LEGACY_STEP_NAMES: dict[str, str] = {
    "crop_far_range_noise": STEP_CROP_FARRANGE,
    "interpolate_aux_onto_pings": STEP_INTERPOLATE_AUX,
    "tvg_correction": STEP_TVG,
    "compute_geometric_ranges": STEP_GEOMETRIC,
    "empirical_range_correction": STEP_RANGE_CORRECTION,
    "compute_georeferencing": STEP_GEOREF,
}
"""Step names written to ``processing_history`` by older versions."""

LEVEL_STEPS: tuple[str, ...] = tuple(
    s for s in BUILTIN_STEPS if s not in (STEP_CROP_FARRANGE, STEP_INTERPOLATE_AUX, STEP_BOTTOM_TRACK)
)
"""Built-in steps that define a processing level. Cropping, aux
interpolation and bottom tracking prepare data but do not correct it, so
they are not levels."""

PROCESSING_LEVELS: tuple[str, ...] = ("raw", *LEVEL_STEPS)
"""Values :func:`processing_level` can return, least to most processed."""

_LEGACY_LEVEL_ATTR = "processing_level"
_LEGACY_LEVELS: dict[str, str] = {"raw": "raw", "radiometric": STEP_TVG}


def canonical_step(name: str) -> str:
    """Map a recorded step name (possibly a legacy one) to its canonical name."""
    return LEGACY_STEP_NAMES.get(name, name)


def read_history(ds: xr.Dataset) -> list[dict[str, Any]]:
    """Parsed ``processing_history`` records (raw, names not canonicalised).

    A missing attribute is an empty history. Malformed JSON logs a WARNING and
    is treated as empty.
    """
    raw = ds.attrs.get("processing_history", "[]")
    try:
        history = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("processing_history is not valid JSON; treating it as empty")
        return []
    if not isinstance(history, list):
        logger.warning("processing_history is not a JSON list; treating it as empty")
        return []
    return history


def applied_steps(ds: xr.Dataset) -> list[str]:
    """Canonical names of every recorded step, oldest first."""
    return [canonical_step(str(r.get("step", ""))) for r in read_history(ds)]


def _output_var_done(ds: xr.Dataset, step: str) -> bool:
    var = STEP_OUTPUT_VARS[step][0]
    if var not in ds.variables:
        return False
    if step in applied_steps(ds):
        return True  # recorded and present: skip the full-array NaN scan
    return not bool(ds[var].isnull().all())


def has_step(ds: xr.Dataset, step: str) -> bool:
    """True if ``step`` has been applied to ``ds``.

    Geometric/georef: their output variable is present and not all NaN (a
    variable dropped since is "not done"; when the step is also recorded in
    the history the NaN scan is skipped).
    Every other step (built-in or custom): a record with that name (legacy
    names included) is in the history. A legacy ``processing_level ==
    "radiometric"`` attribute counts as ``tvg`` when no history record says so.
    """
    step = canonical_step(step)
    if step in STEP_OUTPUT_VARS:
        return _output_var_done(ds, step)
    if step in applied_steps(ds):
        return True
    if step == STEP_TVG:
        return _LEGACY_LEVELS.get(ds.attrs.get(_LEGACY_LEVEL_ATTR, "raw")) == STEP_TVG
    return False


def processing_level(ds: xr.Dataset) -> str:
    """The most advanced correction step applied to ``ds`` (canonical order), or ``"raw"``.

    Derived from :func:`has_step` for each of :data:`LEVEL_STEPS`; custom
    steps, cropping and aux interpolation do not change the level. One of :data:`PROCESSING_LEVELS`.
    """
    level = "raw"
    for step in LEVEL_STEPS:
        if has_step(ds, step):
            level = step
    return level


def append_history(
    ds: xr.Dataset,
    step: str,
    params: Mapping[str, Any],
    *,
    function: str,
    version: str,
) -> xr.Dataset:
    """Return a shallow copy of ``ds`` with one step record appended.

    Also drops the legacy ``processing_level`` attribute: once the library
    has written history, the level is derived from it.

    ``params`` must be JSON-serialisable; non-serialisable values are stored
    via ``str()``.
    """
    history = read_history(ds)
    history.append({
        "step": step,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "params": json.loads(json.dumps(dict(params), default=str)),
        "function": function,
        "version": version,
    })
    out = ds.copy(deep=False)
    out.attrs = {k: v for k, v in ds.attrs.items() if k != _LEGACY_LEVEL_ATTR}
    out.attrs["processing_history"] = json.dumps(history)
    return out


def remove_history_records(ds: xr.Dataset, indices: set[int] | list[int]) -> xr.Dataset:
    """Return a shallow copy of ``ds`` without the history records at ``indices``.

    Also drops the legacy ``processing_level`` attribute (see :func:`append_history`).
    """
    drop = set(indices)
    history = [r for i, r in enumerate(read_history(ds)) if i not in drop]
    out = ds.copy(deep=False)
    out.attrs = {k: v for k, v in ds.attrs.items() if k != _LEGACY_LEVEL_ATTR}
    out.attrs["processing_history"] = json.dumps(history)
    return out
