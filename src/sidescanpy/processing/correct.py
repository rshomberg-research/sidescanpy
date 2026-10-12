"""
One correction API: apply an ordered list of steps, each skippable and
replaceable, recording every applied step in ``processing_history``.

- :func:`correct` -- one frequency channel dataset.
- :func:`correct_survey` -- every channel of a survey (supplies the survey's
  aux records to the ``"interpolate_aux"`` step).
- :func:`correct_file` -- read a ``.ss.h5`` file, correct it, write it
  atomically (in place by default), optionally in ping chunks to bound memory.
- :func:`correction_steps` -- a set of requested built-in steps in canonical
  order, with ``"interpolate_aux"`` added when a step needs it.

Steps
-----
A step is a built-in name, a :class:`Step` (custom name + function) or any
callable ``fn(ds, **params) -> ds``. Built-in steps, in canonical order:

``"crop_farrange"``
    :func:`~sidescanpy.processing.farrange.crop_far_range`. Params:
    ``cutoff_sample, n_bins, min_valid_fraction, rise_threshold_db, margin_bins``.
``"interpolate_aux"``
    :func:`~sidescanpy.processing.interpolate.interpolate_aux_onto_dataset`;
    needs the survey's aux records (``aux=`` / :func:`correct_survey`);
    :func:`correct` skips it when no ``aux`` is given.
    Params: ``fields, method, extrapolation, drop_invalid, held_values,
    max_hold_s, per_ping_fallback, groups, merge_window_s``.
``"tvg"``
    :func:`~sidescanpy.processing.radiometric.apply_tvg`. Params:
    ``spreading_loss_db, absorption_db_per_m, min_range_m, sound_velocity_ms``,
    plus ``temperature_c, salinity_psu, depth_m, ph`` (Francois-Garrison
    inputs, used only when ``absorption_db_per_m`` is None; model of Francois &
    Garrison, J. Acoust. Soc. Am. 72(3) 896-907 and 72(6) 1879-1890, 1982).
``"geometric"``
    :func:`~sidescanpy.processing.geometric.compute_geometric_ranges`.
    Params: ``expose_unmasked_z``.
``"range_correction"``
    :func:`~sidescanpy.processing.range_correction.apply_empirical_range_correction`
    (needs tvg + geometric). Params: ``table`` (required: a
    ``RangeCorrectionTable`` or the path of a saved one; no built-in default).
``"georef"``
    :func:`~sidescanpy.processing.georeferencing.compute_georeferencing`
    (needs geometric). Params: ``ellipsoid``.

Steps run in the order given; dependencies are the caller's to respect.
:data:`DEFAULT_STEPS` is interpolate_aux -> tvg -> geometric -> georef.

Conventions
-----------
- **Skip** a step by leaving it out of ``steps``. ``params`` may only name
  steps that are in ``steps`` (a typo raises ``ValueError``); a parameter
  the step does not accept raises ``TypeError``. Both are checked before
  anything runs.
- **Idempotent**: a step already applied (see
  :func:`sidescanpy.core.history.has_step`) is skipped with a DEBUG log, so
  re-running is a no-op. Custom steps are matched by name, so give each one
  a distinct name (lambdas are rejected for that reason).
- ``force=True`` re-applies already-applied steps. A radiometric step
  (``tvg``/``range_correction``) is first inverted -- together with every
  radiometric step applied after it, all of which must then be in
  ``steps`` -- so its new parameters replace, not stack on, the old ones.
- **History**: every applied step appends one record
  ``{"step", "params", "function", "version", "timestamp_utc"}`` with the
  parameters actually used (defaults resolved). Built-in steps write their
  own record; custom steps are recorded here with the bound arguments.
- No uncorrected copy is kept: :func:`sidescanpy.processing.inversion.invert`
  undoes recorded steps.

2026 rshomberg@uri.edu
"""

import contextlib
import inspect
import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import (
    BUILTIN_STEPS,
    RADIOMETRIC_STEPS,
    STEP_CROP_FARRANGE,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_INTERPOLATE_AUX,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    append_history,
    applied_steps,
    canonical_step,
    has_step,
)
from sidescanpy.core.schema import HDF5_CHUNK_PING
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.io.hdf5 import read_survey, write_survey
from sidescanpy.processing.farrange import crop_far_range, detect_far_range_cutoff_sample
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.georeferencing import compute_georeferencing
from sidescanpy.processing.interpolate import interpolate_aux_onto_dataset
from sidescanpy.processing.inversion import LEGACY_RAW_VAR, invert
from sidescanpy.processing.radiometric import TVGParams, apply_tvg, resolve_tvg_params
from sidescanpy.processing.range_correction import apply_empirical_range_correction
from sidescanpy.processing.sample_axis import (
    invalid_sound_vel_warning_suppressed,
    sample_interval_m,
)

logger = logging.getLogger(__name__)

DEFAULT_STEPS: tuple[str, ...] = (STEP_INTERPOLATE_AUX, STEP_TVG, STEP_GEOMETRIC, STEP_GEOREF)

_HDF5_ENGINE = "h5netcdf"


@dataclass(frozen=True)
class Step:
    """A caller-supplied correction step with an explicit history name.

    ``fn(ds, **params) -> xr.Dataset``. The name is what ``processing_history``
    records and what idempotency matches on; it must not be a built-in name.
    """

    name: str
    fn: Callable[..., xr.Dataset]


# ============================================================
# BUILT-IN STEP ADAPTERS (flat, JSON-friendly params)
# ============================================================

def _tvg_step(
    ds: xr.Dataset,
    *,
    spreading_loss_db: float = 40.0,
    absorption_db_per_m: float | None = None,
    min_range_m: float = 0.5,
    sound_velocity_ms: float | None = None,
    temperature_c: float | None = None,
    salinity_psu: float | None = None,
    depth_m: float | None = None,
    ph: float | None = None,
) -> xr.Dataset:
    return apply_tvg(ds, TVGParams(
        spreading_loss_db, absorption_db_per_m, min_range_m, sound_velocity_ms,
        temperature_c, salinity_psu, depth_m, ph,
    ))


def _interpolate_aux_step(
    ds: xr.Dataset,
    *,
    aux: SideScanSurvey,
    fields: dict[str, tuple[str, str]] | None = None,
    method: str = "linear",
    extrapolation: str = "clamp",
    drop_invalid: bool = True,
    held_values: str = "fixes",
    max_hold_s: float = 2.0,
    per_ping_fallback: bool = True,
    groups: list[list[str]] | None = None,
    merge_window_s: float | dict[str, float] | None = None,
) -> xr.Dataset:
    return interpolate_aux_onto_dataset(
        ds, aux, fields=fields, method=method, extrapolation=extrapolation, drop_invalid=drop_invalid,
        held_values=held_values, max_hold_s=max_hold_s, per_ping_fallback=per_ping_fallback,
        groups=groups, merge_window_s=merge_window_s,
    )


_BUILTIN_FUNCTIONS: dict[str, Callable[..., xr.Dataset]] = {
    STEP_CROP_FARRANGE: crop_far_range,
    STEP_INTERPOLATE_AUX: _interpolate_aux_step,
    STEP_TVG: _tvg_step,
    STEP_GEOMETRIC: compute_geometric_ranges,
    STEP_RANGE_CORRECTION: apply_empirical_range_correction,
    STEP_GEOREF: compute_georeferencing,
}
_INTERNAL_PARAMS = {STEP_INTERPOLATE_AUX: {"aux"}}


# ============================================================
# STEP RESOLUTION + PLANNING
# ============================================================

@dataclass
class _Resolved:
    name: str
    fn: Callable[..., xr.Dataset]
    builtin: bool
    params: dict[str, Any] = field(default_factory=dict)
    record_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Action:
    kind: str  # "invert" | "apply"
    step: _Resolved | None = None
    invert_steps: tuple[str, ...] = ()


def _step_name(step: str | Step | Callable) -> tuple[str, Callable[..., xr.Dataset], bool]:
    if isinstance(step, str):
        name = canonical_step(step)
        if name not in _BUILTIN_FUNCTIONS:
            raise ValueError(f"Unknown correction step {step!r}. Built-in steps: {BUILTIN_STEPS}")
        return name, _BUILTIN_FUNCTIONS[name], True
    if isinstance(step, Step):
        name, fn = step.name, step.fn
    elif callable(step):
        name, fn = getattr(step, "name", None) or getattr(step, "__qualname__", None), step
        if name and "<locals>." in name:
            name = name.rsplit("<locals>.", 1)[1]
        if not name or "<lambda>" in name:
            raise ValueError("custom steps need a distinct name: wrap a lambda as Step('name', fn)")
    else:
        raise TypeError(f"a step must be a built-in name, a Step or a callable, got {type(step).__name__}")
    if canonical_step(name) in BUILTIN_STEPS:
        raise ValueError(f"custom step name {name!r} collides with a built-in step")
    return name, fn, False


def _bind(name: str, fn: Callable, params: Mapping[str, Any], builtin: bool) -> dict[str, Any]:
    """Validate ``params`` against ``fn``'s signature; return the full bound arguments (defaults resolved)."""
    internal = _INTERNAL_PARAMS.get(name, set()) if builtin else set()
    bad_internal = internal & set(params)
    if bad_internal:
        raise TypeError(f"step {name!r}: {sorted(bad_internal)} is supplied by the caller API, not params")
    sig = inspect.signature(fn)
    first = next(iter(sig.parameters))
    try:
        bound = sig.bind(None, **{k: None for k in internal}, **params)
    except TypeError as exc:
        accepted = [p for p in sig.parameters if p != first and p not in internal]
        raise TypeError(f"step {name!r}: {exc}. Accepted params: {accepted}") from None
    bound.apply_defaults()
    out: dict[str, Any] = {}
    for key, value in bound.arguments.items():
        if key == first or key in internal:
            continue
        if sig.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[key] = value
    return out


def _resolve_steps(
    steps: Sequence[str | Step | Callable], params: Mapping[str, Mapping[str, Any] | None] | None,
) -> list[_Resolved]:
    if isinstance(steps, (str, Step)) or callable(steps):
        steps = [steps]
    resolved: list[_Resolved] = []
    for step in steps:
        name, fn, builtin = _step_name(step)
        if any(r.name == name for r in resolved):
            raise ValueError(f"step {name!r} is listed twice")
        resolved.append(_Resolved(name, fn, builtin))

    params = dict(params or {})
    names = {r.name for r in resolved}
    canon_params: dict[str, Any] = {}
    for key, value in params.items():
        name = canonical_step(key)
        if name not in names:
            raise ValueError(
                f"params given for {key!r}, which is not in steps {sorted(names)} "
                "(to skip a step, leave it out of steps)"
            )
        if not isinstance(value, Mapping):
            raise TypeError(f"params[{key!r}] must be a mapping of keyword arguments, got {type(value).__name__}")
        canon_params[name] = dict(value)
    for r in resolved:
        r.params = canon_params.get(r.name, {})
        r.record_params = _bind(r.name, r.fn, r.params, r.builtin)
        if r.builtin and r.name == STEP_RANGE_CORRECTION and r.params.get("table") is None:
            raise TypeError(
                "step 'range_correction' requires params['range_correction']['table'] "
                "(a RangeCorrectionTable or the path of a saved one); there is no built-in default table"
            )
    return resolved


def _merge_params(
    params: Mapping[str, Mapping[str, Any] | None] | None,
    override: Mapping[str, Mapping[str, Any] | None] | None,
) -> dict[str, dict[str, Any]] | None:
    """``params`` with ``override`` merged in, step by step (override wins per key)."""
    if not override:
        return None if params is None else {k: dict(v or {}) for k, v in params.items()}
    merged = {k: dict(v or {}) for k, v in (params or {}).items()}
    for step, step_params in override.items():
        merged.setdefault(step, {}).update(step_params or {})
    return merged


def _resolve_per_frequency(
    freq_keys: Iterable[str],
    steps: Sequence[str | Step | Callable],
    params: Mapping[str, Mapping[str, Any] | None] | None,
    frequency_params: Mapping[str, Mapping[str, Mapping[str, Any] | None]] | None,
) -> dict[str, list[_Resolved]]:
    """Resolved steps for each frequency: ``params`` plus that frequency's overrides.

    A ``frequency_params`` key that matches no frequency is logged (a project
    config may list frequencies some files lack) and otherwise ignored.
    """
    freq_keys = list(freq_keys)
    for fk in frequency_params or {}:
        if fk not in freq_keys:
            logger.warning("frequency_params names %r, which is not in this survey (%s); ignored", fk, ", ".join(freq_keys))
    return {fk: _resolve_steps(steps, _merge_params(params, (frequency_params or {}).get(fk))) for fk in freq_keys}


def _plan(ds: xr.Dataset, resolved: list[_Resolved], *, force: bool) -> list[_Action]:
    """Actions for one channel: optional inversion (force), then the steps not yet applied."""
    actions: list[_Action] = []
    requested = {r.name for r in resolved}
    if force:
        forced = [r.name for r in resolved if r.name in RADIOMETRIC_STEPS and has_step(ds, r.name)]
        if forced:
            # Undo the oldest forced radiometric record and everything radiometric after it.
            recorded = [s for s in applied_steps(ds) if s in RADIOMETRIC_STEPS]
            starts = [recorded.index(s) for s in forced if s in recorded]
            to_undo = tuple(dict.fromkeys(recorded[min(starts):])) if starts else tuple(forced)
            missing = sorted(set(to_undo) - requested)
            if missing:
                raise ValueError(
                    f"force re-applying {forced} first inverts {list(to_undo)}; "
                    f"also list {missing} in steps so it is re-applied"
                )
            actions.append(_Action("invert", invert_steps=to_undo))
    for r in resolved:
        if not force and has_step(ds, r.name):
            logger.debug("step %r already applied; skipping", r.name)
            continue
        actions.append(_Action("apply", step=r))
    return actions


def _execute(
    ds: xr.Dataset,
    actions: list[_Action],
    aux: SideScanSurvey | None,
    progress: Callable[[str], None] | None = None,
    label: str = "",
) -> xr.Dataset:
    for action in actions:
        if action.kind == "invert":
            logger.debug("force: inverting %s before re-applying", list(action.invert_steps))
            ds = invert(ds, action.invert_steps)
            continue
        r = action.step
        if progress is not None:
            progress(f"{label}{r.name}")
        logger.debug("applying step %r with params %s", r.name, r.record_params)
        if r.builtin:
            kwargs = dict(r.params)
            if r.name == STEP_INTERPOLATE_AUX:
                kwargs["aux"] = aux
            ds = r.fn(ds, **kwargs)
            continue
        out = r.fn(ds, **r.params)
        if not isinstance(out, xr.Dataset):
            raise TypeError(f"custom step {r.name!r} returned {type(out).__name__}, expected xr.Dataset")
        ds = append_history(
            out, r.name, r.record_params,
            function=f"{getattr(r.fn, '__module__', '?')}.{getattr(r.fn, '__qualname__', r.name)}",
            version=__version__,
        )
    return ds


def _check_aux(actions: list[_Action], aux: SideScanSurvey | None) -> None:
    if aux is None and any(a.kind == "apply" and a.step.name == STEP_INTERPOLATE_AUX for a in actions):
        raise ValueError(
            "step 'interpolate_aux' needs the survey's aux records: pass aux=survey "
            "(or use correct_survey / correct_file), or leave the step out"
        )


# ============================================================
# PUBLIC API
# ============================================================

_NEEDS_AUX = frozenset({STEP_TVG, STEP_GEOMETRIC, STEP_RANGE_CORRECTION, STEP_GEOREF})


def correction_steps(requested: Iterable[str]) -> list[str]:
    """Built-in step names in canonical order, adding ``"interpolate_aux"`` when needed.

    ``"interpolate_aux"`` is added whenever any of tvg / geometric /
    range_correction / georef is requested, since each reads per-ping
    navigation/attitude fields. Duplicates collapse.

    Raises
    ------
    ValueError
        A name that is not a built-in step.
    """
    requested = set(requested)
    unknown = requested - set(BUILTIN_STEPS)
    if unknown:
        raise ValueError(f"unknown correction step(s): {sorted(unknown)} (built-in: {list(BUILTIN_STEPS)})")
    if requested & _NEEDS_AUX:
        requested.add(STEP_INTERPOLATE_AUX)
    return [s for s in BUILTIN_STEPS if s in requested]


def correct(
    ds: xr.Dataset,
    steps: Sequence[str | Step | Callable] = DEFAULT_STEPS,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    aux: SideScanSurvey | None = None,
    force: bool = False,
) -> xr.Dataset:
    """Apply correction ``steps`` to one frequency channel dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Frequency channel dataset (not modified).
    steps : sequence of str / Step / callable, default :data:`DEFAULT_STEPS`
        Applied in order; see the module docstring for built-ins.
    params : mapping, optional
        ``{step_name: {param: value}}`` overrides for steps in ``steps``.
    aux : SideScanSurvey, optional
        Supplies navigation/attitude records to ``"interpolate_aux"``. When
        omitted, that step is skipped (DEBUG log) unless ``params`` gives it
        parameters, which is an error.
    force : bool, default False
        Re-apply steps that were already applied (see module docstring).

    Returns
    -------
    xr.Dataset
        Corrected dataset with one history record per applied step.

    Raises
    ------
    ValueError
        Unknown step name, params for a step not in ``steps``,
        ``params["interpolate_aux"]`` given without ``aux``, or an
        impossible ``force`` request.
    TypeError
        A parameter the step does not accept.
    """
    resolved = _resolve_steps(steps, params)
    if aux is None and any(r.name == STEP_INTERPOLATE_AUX for r in resolved):
        if any(canonical_step(k) == STEP_INTERPOLATE_AUX for k in (params or {})):
            raise ValueError(
                "params were given for step 'interpolate_aux' but no aux: pass aux=survey "
                "(or use correct_survey / correct_file), or leave the step out"
            )
        logger.debug("interpolate_aux skipped: no aux given")
        resolved = [r for r in resolved if r.name != STEP_INTERPOLATE_AUX]
    actions = _plan(ds, resolved, force=force)
    _check_aux(actions, aux)
    return _execute(ds, actions, aux)


def correct_survey(
    survey: SideScanSurvey,
    steps: Sequence[str | Step | Callable] = DEFAULT_STEPS,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    frequency_params: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SideScanSurvey:
    """:func:`correct` every frequency channel of ``survey``.

    The survey's own aux records feed ``"interpolate_aux"``. ``params`` apply
    to every channel; ``frequency_params`` (``{freq_key: {step: {param:
    value}}}``) overrides them for one frequency, e.g. a per-frequency
    ``absorption_db_per_m`` or range-correction table. Returns a new
    (shallow-copied) survey; ``survey`` is not modified. ``progress``, if
    given, is called with ``"<freq_key>: <step>"`` before each step.
    """
    resolved_by_fk = _resolve_per_frequency(survey.frequencies, steps, params, frequency_params)
    out = survey.copy()
    for freq_key, channel in survey.freq_channels.items():
        actions = _plan(channel.dataset, resolved_by_fk[freq_key], force=force)
        _check_aux(actions, survey)
        if actions:
            out.add_frequency(freq_key, _execute(channel.dataset, actions, survey, progress, f"{freq_key}: "))
    return out


# ============================================================
# FILE-LEVEL
# ============================================================

# Estimated growth of an .ss.h5 file per step, as a fraction of the
# converted file's size (gzip-4 + shuffle, float32 except per-sample lat/lon).
# Measured on a real JSF survey (converted file, no raw copy, 97 MB):
# +tvg 1.9x, +geometric 1.9x, +georef 4.4x -> tvg+geometric+georef = 9.2x the
# converted size (~4.6x a pre-change converted file that still held a raw
# copy). tvg swaps small-integer amplitudes (compress well) for dB floats;
# georef adds two float64 per-sample arrays.
_DISK_GROWTH = {
    STEP_CROP_FARRANGE: 0.0,
    STEP_INTERPOLATE_AUX: 0.0,
    STEP_TVG: 2.0,
    STEP_GEOMETRIC: 2.0,
    STEP_RANGE_CORRECTION: 0.0,
    STEP_GEOREF: 4.5,
}
_CUSTOM_STEP_GROWTH = 1.0
_DISK_SAFETY = 1.2
# Uncompressed bytes per (ping, channel, sample) cell in chunk temp files.
_CELL_BYTES = {STEP_TVG: 0, STEP_GEOMETRIC: 6 * 4, STEP_GEOREF: 2 * 8}


def estimate_correct_disk_bytes(
    path: Path | str,
    steps: Sequence[str | Step | Callable] = DEFAULT_STEPS,
    *,
    chunk_pings: int | None = None,
) -> int:
    """Free disk space :func:`correct_file` needs to correct ``path`` with ``steps``.

    Output size is estimated as ``size * (1 + sum(growth[step]))`` (growth
    per step in ``_DISK_GROWTH``, measured on real data: tvg + geometric +
    georef takes a converted file to ~9.2x), times a 1.2 safety factor. The
    original file stays until the atomic replace, so the full output size is
    needed. With ``chunk_pings`` the uncompressed per-chunk temp files are
    added (``n_cells * bytes per cell``). Deliberately conservative: it
    assumes every requested step will run.
    """
    names = [r.name for r in _resolve_steps(steps, None)]
    return _estimate_disk_bytes(Path(path), names, chunk_pings)


def _estimate_disk_bytes(path: Path, names: list[str], chunk_pings: int | None) -> int:
    growth = sum(_DISK_GROWTH.get(n, _CUSTOM_STEP_GROWTH) for n in names)
    required = path.stat().st_size * (1.0 + growth) * _DISK_SAFETY
    if chunk_pings is not None:
        survey = read_survey(path, chunks={"ping": HDF5_CHUNK_PING})
        added = sum(_CELL_BYTES.get(n, 0) for n in names)
        for channel in survey.freq_channels.values():
            ds = channel.dataset
            existing = sum(
                ds[v].dtype.itemsize for v in ds.data_vars if ds[v].dims == ("ping", "channel", "sample")
            )
            required += ds.sizes["ping"] * ds.sizes["channel"] * ds.sizes["sample"] * (existing + added)
    return int(required)


def _drop_legacy_raw(ds: xr.Dataset) -> xr.Dataset:
    if LEGACY_RAW_VAR in ds.variables:
        logger.debug("dropping legacy %s variable on rewrite", LEGACY_RAW_VAR)
        return ds.drop_vars(LEGACY_RAW_VAR)
    return ds


def _atomic_write(survey: SideScanSurvey, out_path: Path) -> None:
    tmp = out_path.with_name(out_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    write_survey(survey, tmp, overwrite=False)
    tmp.replace(out_path)


def _resolve_channel_params(ds: xr.Dataset, actions: list[_Action]) -> list[_Action]:
    """Fix channel-wide parameters once so every ping chunk uses the same values.

    ``tvg``: sound velocity (channel mean) and absorption default.
    ``crop_farrange``: the cutoff (detected on the whole channel; detection
    inverts radiometric steps, so it does not depend on step order).
    """
    out = []
    for action in actions:
        r = action.step
        if action.kind == "apply" and r.builtin and r.name == STEP_TVG:
            light = ds[["sound_vel_ms"]].load()
            resolved = resolve_tvg_params(light, TVGParams(**r.record_params))
            # A derived absorption is recomputed (identically) per chunk from the
            # now-fixed water conditions so those get recorded with it.
            fixed = {"sound_velocity_ms": resolved.sound_velocity_ms}
            if r.record_params.get("absorption_db_per_m") is None:
                fixed.update({k: getattr(resolved, k) for k in ("temperature_c", "salinity_psu", "depth_m", "ph")})
            else:
                fixed["absorption_db_per_m"] = resolved.absorption_db_per_m
            r = _Resolved(r.name, r.fn, True, {**r.params, **fixed}, r.record_params)
        elif action.kind == "apply" and r.builtin and r.name == STEP_CROP_FARRANGE \
                and r.record_params.get("cutoff_sample") is None:
            detect = {k: v for k, v in r.record_params.items() if k != "cutoff_sample"}
            cutoff = detect_far_range_cutoff_sample(ds, **detect)
            if cutoff is None:
                cutoff = ds.sizes["sample"]  # nothing to crop; stops per-chunk re-detection
            r = _Resolved(r.name, r.fn, True, {**r.params, "cutoff_sample": cutoff}, r.record_params)
        out.append(_Action(action.kind, r, action.invert_steps))
    return out


def correct_file(
    path: Path | str,
    steps: Sequence[str | Step | Callable] = DEFAULT_STEPS,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    frequency_params: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    out_path: Path | str | None = None,
    chunk_pings: int | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Correct a ``.ss.h5`` file and write it atomically.

    Parameters
    ----------
    path : path
        Input ``.ss.h5`` file.
    steps, params, force
        As for :func:`correct`; applied to every frequency channel, with the
        file's aux records feeding ``"interpolate_aux"``.
    frequency_params : mapping, optional
        ``{freq_key: {step: {param: value}}}`` overrides of ``params`` for one
        frequency (see :func:`correct_survey`).
    out_path : path, optional
        Output file. Default: ``path`` (in place, via a temp file + rename, so
        a failure never corrupts the original).
    chunk_pings : int, optional
        Process each channel in chunks of this many pings, bounding peak
        memory to roughly one chunk's worth of derived arrays. Results are
        identical to the unchunked path: channel-wide parameters (TVG sound
        velocity, far-range cutoff) are resolved once per channel first.
        Custom steps must be ping-independent to be used this way.
    progress : callable, optional
        Called with a short status string per step / chunk / write.

    Returns
    -------
    Path
        The written file (``path`` unchanged and not rewritten when every
        requested step was already applied and ``out_path`` is not given).

    Notes
    -----
    A legacy ``backscatter_raw`` variable is dropped when the file is
    rewritten. Free space is checked first (:func:`estimate_correct_disk_bytes`);
    ``OSError`` if it is insufficient.
    """
    path = Path(path)
    out = Path(out_path) if out_path is not None else path
    if chunk_pings is not None and chunk_pings <= 0:
        raise ValueError("chunk_pings must be a positive integer")
    resolved = _resolve_steps(steps, params)  # validates steps/params up front

    required = _estimate_disk_bytes(path, [r.name for r in resolved], chunk_pings)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    logger.debug("disk check %s: need ~%d B, free %d B", path.name, required, free)
    if free < required:
        raise OSError(f"insufficient free space to correct {path.name}: need ~{required} B, have {free} B")

    start = time.perf_counter()
    if chunk_pings is None:
        written = _correct_file_eager(path, out, steps, params, frequency_params, force, progress)
    else:
        written = _correct_file_chunked(path, out, steps, params, frequency_params, force, progress, chunk_pings)
    logger.debug("correct_file %s: %s in %.1fs", path.name, "written" if written else "unchanged",
                 time.perf_counter() - start)
    return out if written else path


def _correct_file_eager(path, out, steps, params, frequency_params, force, progress) -> bool:
    survey = read_survey(path)
    resolved_by_fk = _resolve_per_frequency(survey.frequencies, steps, params, frequency_params)
    plans = {fk: _plan(ch.dataset, resolved_by_fk[fk], force=force) for fk, ch in survey.freq_channels.items()}
    if not any(plans.values()) and out == path:
        logger.debug("%s: every requested step already applied; not rewriting", path.name)
        return False
    new = survey.copy()
    for freq_key, channel in survey.freq_channels.items():
        _check_aux(plans[freq_key], survey)
        ds = _execute(channel.dataset, plans[freq_key], survey, progress, f"{freq_key}: ")
        new.add_frequency(freq_key, _drop_legacy_raw(ds))
    if progress is not None:
        progress(f"writing {out.name}")
    _atomic_write(new, out)
    return True


class _DropRepeats(logging.Filter):
    """Drop log records whose message was already emitted while installed."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message in self._seen:
            return False
        self._seen.add(message)
        return True


@contextlib.contextmanager
def _once_per_channel(*logger_names: str) -> Iterator[None]:
    """Show each distinct warning from ``logger_names`` once, not once per ping chunk."""
    loggers = [logging.getLogger(n) for n in logger_names]
    flt = _DropRepeats()
    for lg in loggers:
        lg.addFilter(flt)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(flt)


def _correct_file_chunked(path, out, steps, params, frequency_params, force, progress, chunk_pings) -> bool:
    survey = read_survey(path, chunks={"ping": HDF5_CHUNK_PING})
    resolved_by_fk = _resolve_per_frequency(survey.frequencies, steps, params, frequency_params)
    plans = {fk: _plan(ch.dataset, resolved_by_fk[fk], force=force) for fk, ch in survey.freq_channels.items()}
    if not any(plans.values()) and out == path:
        logger.debug("%s: every requested step already applied; not rewriting", path.name)
        return False
    tmp_dir = Path(tempfile.mkdtemp(prefix=".sidescanpy_chunked_", dir=out.parent))
    try:
        new = survey.copy()
        for freq_key in survey.frequencies:
            ds = survey.get_frequency(freq_key).dataset
            actions = plans[freq_key]
            if not actions:
                new.add_frequency(freq_key, _drop_legacy_raw(ds))
                continue
            _check_aux(actions, survey)
            actions = _resolve_channel_params(ds, actions)
            if any(a.kind == "apply" and a.step.name == STEP_GEOMETRIC for a in actions):
                # One warning per channel (not per chunk) about invalid sound velocities.
                sample_interval_m(ds[["sound_vel_ms"]].load())
            n_pings = ds.sizes["ping"]
            chunk_paths = []
            # interpolate_aux warns about empty/NaN aux fields once per chunk otherwise.
            with _once_per_channel("sidescanpy.processing.interpolate"):
                for lo in range(0, n_pings, chunk_pings):
                    hi = min(lo + chunk_pings, n_pings)
                    if progress is not None:
                        progress(f"{freq_key}: pings {lo}-{hi} of {n_pings}")
                    with invalid_sound_vel_warning_suppressed():
                        chunk = _execute(ds.isel(ping=slice(lo, hi)).load(), actions, survey)
                    chunk_path = tmp_dir / f"{freq_key}_{lo:012d}.nc"
                    _drop_legacy_raw(chunk).to_netcdf(chunk_path, engine=_HDF5_ENGINE)
                    chunk_paths.append(chunk_path)
            # Lazily reassembled in chunk order (nested concat along ping).
            combined = xr.open_mfdataset(
                chunk_paths, engine=_HDF5_ENGINE, concat_dim="ping", combine="nested",
                data_vars="all", coords="minimal", compat="override",
                chunks={"ping": chunk_pings},
            )
            new.add_frequency(freq_key, combined)
        if progress is not None:
            progress(f"writing {out.name}")
        # The default threaded scheduler holds many computed chunks while
        # waiting on the HDF5 write lock; single-threaded keeps peak memory
        # proportional to one chunk.
        with dask.config.set(scheduler="single-threaded"):
            _atomic_write(new, out)
        for channel in new.freq_channels.values():
            channel.dataset.close()
        for channel in survey.freq_channels.values():
            channel.dataset.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True

