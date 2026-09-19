"""
Pipeline orchestration: running processing steps to a target correction level.

Two entry points:

- :func:`run_pipeline` — the original 4-rung ladder
  (``none -> radiometric -> geometric -> georef``), where each level
  implies every step before it. Kept for callers that only need "how far
  through the pipeline" (e.g. a single ``--process`` CLI flag).
- :func:`run_correction_steps` — a lower-level primitive taking an
  explicit *set* of steps, so callers can request combinations the
  ladder can't express, such as geometric ranges without TVG (verified
  as a supported combination in ``docs/planning/workflow-decisions.md``'s
  "Verified combinations" table). Also accepts an optional
  ``step_params`` dict for callers that need to override a step
  function's own defaults (e.g. custom ``TVGParams`` for the ``"tvg"``
  step) without bypassing this pipeline machinery.

``run_pipeline`` is implemented on top of ``run_correction_steps`` — the
ladder is just a fixed mapping from level name to step set.

2026 rshomberg@uri.edu
"""

import json
import logging

import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.georeferencing import compute_georeferencing
from sidescanpy.processing.interpolate import interpolate_aux_onto_pings
from sidescanpy.processing.radiometric import apply_tvg
from sidescanpy.processing.range_correction import STEP_NAME as _RANGE_CORRECTION_STEP_NAME
from sidescanpy.processing.range_correction import apply_empirical_range_correction

logger = logging.getLogger(__name__)


# ============================================================
# STEPS
# ============================================================

STEP_TVG = "tvg"
STEP_GEOMETRIC = "geometric"
STEP_RANGE_CORRECTION = "range_correction"
STEP_GEOREF = "georef"

# Fixed application order — georeferencing needs across_track_m/
# along_track_m from the geometric step; TVG has no dependency on
# either but is applied first by convention (matches the pre-existing
# ladder's step order). range_correction needs ground_range_m from the
# geometric step, so it's ordered right after it; it has no dependency
# on georef either way. range_correction is deliberately **not** part of
# the default ladder below (_LADDER_STEPS) — it's an optional,
# empirically-calibrated step (see processing/range_correction.py), not
# a universal physics-based correction like the other three. Existing
# callers using run_pipeline()'s ladder see no behavior change; opt in
# via run_correction_steps()/apply_correction_steps() with an explicit
# step set that includes "range_correction".
VALID_STEPS = (STEP_TVG, STEP_GEOMETRIC, STEP_RANGE_CORRECTION, STEP_GEOREF)

_STEP_FUNCTIONS = {
    STEP_TVG: apply_tvg,
    STEP_GEOMETRIC: compute_geometric_ranges,
    STEP_RANGE_CORRECTION: apply_empirical_range_correction,
    STEP_GEOREF: compute_georeferencing,
}


def step_already_done(ds: xr.Dataset, step: str) -> bool:
    """True if ``step`` has already been applied to a single frequency
    channel dataset ``ds``.

    Per-dataset predicate for any of ``VALID_STEPS`` — the one place
    this logic lives; ``sidescanpy.cli.correct``'s own per-*survey*
    checks (looping over every channel) and
    ``sidescanpy.processing.chunked``'s per-*chunk* checks both call
    this rather than each keeping their own copy.

    Raises
    ------
    ValueError
        If ``step`` isn't one of ``VALID_STEPS``.
    """
    if step == STEP_TVG:
        return ds.attrs.get("processing_level") != "raw"
    if step == STEP_GEOMETRIC:
        return "ground_range_m" in ds.data_vars
    if step == STEP_GEOREF:
        return "latitude_per_sample" in ds.data_vars
    if step == STEP_RANGE_CORRECTION:
        history = json.loads(ds.attrs.get("processing_history", "[]"))
        return any(h.get("step") == _RANGE_CORRECTION_STEP_NAME for h in history)
    raise ValueError(f"Unknown correction step: {step!r}. Valid steps: {VALID_STEPS}")


def apply_correction_steps(
    ds: xr.Dataset,
    steps: frozenset[str] | set[str] | tuple[str, ...],
    *,
    step_params: dict[str, dict] | None = None,
) -> xr.Dataset:
    """Apply an explicit set of correction steps to one channel dataset.

    Lower-level than :func:`run_correction_steps` — operates directly on
    a single already-aux-interpolated ``xr.Dataset`` (one frequency
    channel), without calling ``interpolate_aux_onto_pings`` (a
    whole-survey operation, assumed to have already happened). Useful
    for segment-at-a-time processing, where aux interpolation happens
    once up front and correction steps are re-applied per segment.

    Parameters
    ----------
    ds : xr.Dataset
        Single frequency channel dataset.
    steps : set of str
        Any subset of ``{"tvg", "geometric", "range_correction", "georef"}``,
        applied in the fixed order ``VALID_STEPS``. Empty set is a no-op.
        ``"range_correction"`` requires ``"geometric"`` to have run first
        (needs ``ground_range_m``) — including it without ``"geometric"``
        raises inside ``apply_empirical_range_correction`` itself.
    step_params : dict[str, dict], optional
        Per-step keyword overrides, keyed by step name and unpacked as
        ``**kwargs`` into that step's function — e.g.
        ``step_params={"tvg": {"params": TVGParams(absorption_db_per_m=0.0)}}``
        calls ``apply_tvg(ds, params=TVGParams(absorption_db_per_m=0.0))``
        instead of ``apply_tvg(ds)``. A general dict rather than a
        TVG-specific kwarg since ``_STEP_FUNCTIONS`` already holds
        callables with differing optional second arguments (``apply_tvg``'s
        ``params``, ``apply_empirical_range_correction``'s ``table``) — one
        mechanism covers all of them, not just TVG, and needs no signature
        change the next time another step grows an override. An entry for
        a step not present in ``steps`` is silently unused, not an error.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    ValueError
        If ``steps`` contains anything outside ``VALID_STEPS``.
    """
    steps = frozenset(steps)
    invalid = steps - set(VALID_STEPS)
    if invalid:
        raise ValueError(
            f"Unknown correction step(s): {sorted(invalid)}. "
            f"Valid steps: {VALID_STEPS}"
        )

    step_params = step_params or {}
    for step in [s for s in VALID_STEPS if s in steps]:
        ds = _STEP_FUNCTIONS[step](ds, **step_params.get(step, {}))
    return ds


def run_correction_steps(
    survey: SideScanSurvey,
    steps: frozenset[str] | set[str] | tuple[str, ...],
    *,
    step_params: dict[str, dict] | None = None,
) -> SideScanSurvey:
    """Run an explicit set of correction steps on every frequency channel.

    Unlike :func:`run_pipeline`, ``steps`` is not a ladder position —
    any subset of ``{"tvg", "geometric", "range_correction", "georef"}``
    is valid, including combinations the ladder can't express (e.g.
    ``{"geometric"}`` without ``"tvg"``: geometric ranges over raw,
    non-TVG-corrected backscatter).

    Parameters
    ----------
    survey : SideScanSurvey
    steps : set of str
        Any subset of ``{"tvg", "geometric", "range_correction", "georef"}``.
        Empty set means no processing (equivalent to raw).
    step_params : dict[str, dict], optional
        Per-step keyword overrides — see :func:`apply_correction_steps`.
        Applied identically to every frequency channel in the survey
        (a term-selection or calibration choice is survey-wide, not
        per-channel).

    Returns
    -------
    SideScanSurvey
        Same survey object, mutated in place (frequency channels
        replaced with processed datasets) and returned for chaining.

    Raises
    ------
    ValueError
        If ``steps`` contains anything outside ``VALID_STEPS``.
    """
    steps = frozenset(steps)
    invalid = steps - set(VALID_STEPS)
    if invalid:
        raise ValueError(
            f"Unknown correction step(s): {sorted(invalid)}. "
            f"Valid steps: {VALID_STEPS}"
        )

    if not steps:
        return survey

    survey = interpolate_aux_onto_pings(survey)

    for fk, channel in list(survey.freq_channels.items()):
        ds = apply_correction_steps(channel.dataset, steps, step_params=step_params)
        survey.add_frequency(fk, ds)

    return survey


# ============================================================
# LADDER (backward-compatible 4-level target)
# ============================================================

VALID_PROCESS_LEVELS = ("none", "radiometric", "geometric", "georef")

_LADDER_STEPS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "radiometric": frozenset({STEP_TVG}),
    "geometric": frozenset({STEP_TVG, STEP_GEOMETRIC}),
    "georef": frozenset({STEP_TVG, STEP_GEOMETRIC, STEP_GEOREF}),
}


def run_pipeline(survey: SideScanSurvey, target: str) -> SideScanSurvey:
    """Run the processing pipeline through a target level.

    Levels in order: ``none -> radiometric -> geometric -> georef``.
    Each level implies every step before it — e.g. ``target="geometric"``
    always includes TVG. For a correction combination the ladder can't
    express (geometric without TVG, radiometric+geometric without
    georef), use :func:`run_correction_steps` directly.

    Parameters
    ----------
    survey : SideScanSurvey
    target : str
        One of ``VALID_PROCESS_LEVELS``.

    Returns
    -------
    SideScanSurvey
    """
    if target not in VALID_PROCESS_LEVELS:
        raise ValueError(
            f"target must be one of {VALID_PROCESS_LEVELS}, got {target!r}"
        )
    return run_correction_steps(survey, _LADDER_STEPS[target])


_GEOMETRY_VARS_TO_DROP = (
    "ground_range_m",
    "across_track_m",
    "along_track_m",
    "slant_range_m",
    "vertical_offset_m",
    "depth_per_sample_m",
    "latitude_per_sample",
    "longitude_per_sample",
)


def reconstruct_raw(ds: xr.Dataset) -> xr.Dataset:
    """Rebuild a ``processing_level == "raw"`` dataset from one that has
    already been through some or all of the correction pipeline.

    Relies on ``backscatter_raw`` being immutable through every
    correction step (``apply_tvg`` only ever reads/writes
    ``backscatter_intensity`` — see this module's own step-order
    comment above) — copies it back into ``backscatter_intensity`` and
    drops the downstream-computed geometry variables so a later
    ``compute_geometric_ranges`` call recomputes them fresh rather than
    silently reusing stale ones. Promoted here from a helper duplicated
    across two `project-a` debug scripts
    (``tvg_range_residual_diagnostic.py``, ``fit_tvg_hardware_curve.py``)
    plus this library's own investigation tooling
    (``sidescanpy.investigation``) — a third independent copy would have
    been one duplicate too many.

    Parameters
    ----------
    ds : xr.Dataset
        Single frequency channel dataset, any ``processing_level``.

    Returns
    -------
    xr.Dataset
        New dataset (input not modified) with ``processing_level =
        "raw"``, empty ``processing_history``, and no downstream
        geometry variables.
    """
    out = ds.copy(deep=False)
    out["backscatter_intensity"] = out["backscatter_raw"]
    drop_vars = [v for v in _GEOMETRY_VARS_TO_DROP if v in out]
    out = out.drop_vars(drop_vars)
    out.attrs = dict(out.attrs)
    out.attrs["processing_level"] = "raw"
    out.attrs["processing_history"] = "[]"
    return out
