"""
Interpolate auxiliary records onto per-ping timestamps.

The converter populates per-ping fields (latitude, heading_deg, etc.) on each
SideScanFrequencyChannel from the embedded values in SonarDataMessage. For
many quantities those embedded values are stale — they're the most-recent
auxiliary reading at the moment the ping was recorded, *not* interpolated
to the ping's exact timestamp.

This module provides higher-precision per-ping values by interpolating
each frequency channel's ping times against the survey's auxiliary records.

Aux records are themselves often sample-and-hold (each entry repeats the
latest sensor value until the sensor updates). By default
(``held_values="fixes"``) each record is first reduced to its real fixes
with :func:`~sidescanpy.processing.aux_fixes.held_value_fixes`, so the
interpolated track is smooth instead of stair-stepped. When a field has no
usable aux source, the channel's own per-ping values (also typically held)
are reduced the same way and interpolated (``per_ping_fallback``).

Two layers:

- ``interpolate_aux_onto_pings(survey, ...)``  — top-level entrypoint.
  Iterates the default field-to-aux mapping (or a user override) and
  applies interpolation per channel.

- ``interpolate_onto_channel(channel, source, field_map, ...)``  — lower
  level. Takes any xr.Dataset with a ``time`` coordinate as the source,
  enabling external data (e.g. a separately recorded GPS log) to drive
  per-ping field updates.

2026 rshomberg@uri.edu
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_INTERPOLATE_AUX, append_history
from sidescanpy.core.survey import SideScanFrequencyChannel, SideScanSurvey
from sidescanpy.processing.aux_fixes import MergeWindow, _check_hold_and_merge, held_value_fixes

logger = logging.getLogger(__name__)


# ============================================================
# DEFAULTS
# ============================================================

# Field source spec: per-ping field name -> (aux_record_name, var_in_aux)
#
# TODO: promote to a dataclass (e.g. FieldSource(aux=..., var=...,
# sensor_type=...)) when we need per-field overrides like sensor_type or
# per-field method. Tuples are sufficient until then.

DEFAULT_FIELD_SOURCES: dict[str, tuple[str, str]] = {
    "latitude":    ("navigation", "latitude"),
    "longitude":   ("navigation", "longitude"),
    "heading_deg": ("attitude",   "heading_deg"),
    "roll_deg":    ("attitude",   "roll_deg"),
    "pitch_deg":   ("attitude",   "pitch_deg"),
    "heave_m":     ("attitude",   "heave_m"),
    # altitude_m, depth_m, sound_vel_ms, water_temp_C are "live sensors" on
    # SonarDataMessage — direct sensor readings, not mirrors of an aux
    # stream. Not interpolated by default. Override `fields=` to opt in.
}


# Per-ping fields holding compass angles in degrees (bearing clockwise from
# north, stored in [0, 360)). These wrap at 360, so interpolating them
# linearly across north (359 -> 1) would pass through ~180. They are
# unwrapped, interpolated, and wrapped back to [0, 360). Roll, pitch and
# the other fields are plain linear.
CIRCULAR_FIELDS: frozenset[str] = frozenset({"heading_deg", "course_deg"})

_VALID_AUX_NAMES = {"navigation", "attitude", "dvl", "sensor"}
_VALID_EXTRAPOLATION = ("clamp", "linear", "nan")
_VALID_HELD_VALUES = ("fixes", "raw")


# ============================================================
# PUBLIC: TOP-LEVEL ENTRYPOINT
# ============================================================

def interpolate_aux_onto_pings(
    survey: SideScanSurvey,
    *,
    fields: dict[str, tuple[str, str]] | None = None,
    method: str = "linear",
    extrapolation: Literal["clamp", "linear", "nan"] = "clamp",
    drop_invalid: bool = True,
    force: bool = False,
    in_place: bool = False,
    held_values: Literal["fixes", "raw"] = "fixes",
    max_hold_s: float = 2.0,
    per_ping_fallback: bool = True,
    groups: Sequence[Sequence[str]] | None = None,
    merge_window_s: MergeWindow = None,
) -> SideScanSurvey:
    """
    Update per-ping fields on every frequency channel by interpolating
    the survey's auxiliary records onto each ping's timestamp.

    Parameters
    ----------
    survey : SideScanSurvey
        Survey to operate on.
    fields : dict, optional
        Mapping ``{per_ping_field: (aux_name, aux_var)}``. Overrides the
        default in ``DEFAULT_FIELD_SOURCES``. Use this to interpolate only a
        subset of fields, or to point a field at a non-default source.
    method : str, default "linear"
        Interpolation method passed to ``xarray.Dataset.interp`` —
        ``"linear"``, ``"nearest"``, ``"cubic"``, etc.
    extrapolation : {"clamp", "linear", "nan"}, default "clamp"
        Behavior when a ping's timestamp falls outside the aux record's
        time range. ``"clamp"`` holds the boundary value; ``"linear"``
        extrapolates the slope; ``"nan"`` writes NaN.
    drop_invalid : bool, default True
        If True, drop rows where the aux source variable is NaN before
        interpolating. Avoids NaN propagation through the interpolated
        result. Set False to require the source be clean.
        With ``held_values="fixes"`` NaN entries are never fixes and are
        always dropped.
    force : bool, default False
        If True, attempt interpolation even when the aux record is empty
        or absent and no per-ping fallback applies: the per-ping field
        receives NaN. If False (default), the field is left unchanged and a
        warning is logged.
    in_place : bool, default False
        If True, mutate ``survey`` in place and return it. If False,
        return a new SideScanSurvey leaving the input untouched.
    held_values : {"fixes", "raw"}, default "fixes"
        ``"fixes"`` reduces each aux record to its real fixes first
        (:func:`~sidescanpy.processing.aux_fixes.held_value_fixes`): many
        recorders log the latest GPS or attitude value at every ping or
        packet, and interpolating between those repeated entries keeps the
        stair-steps (consecutive pings stack on one position, then jump).
        ``"raw"`` interpolates the entries as logged (the only behavior up
        to 0.7.2).
    max_hold_s : float, default 2.0
        With ``held_values="fixes"``: a value unchanged for longer than this
        is kept flat across the hold (a stop, or a sensor dropout) rather
        than ramped through it. See
        :func:`~sidescanpy.processing.aux_fixes.held_value_fixes`.
    per_ping_fallback : bool, default True
        With ``held_values="fixes"``: when a field's aux record is absent
        or empty, or its variable is missing or all NaN, reduce the
        channel's own per-ping values of that field (against the ping
        times) to fixes and interpolate those instead. Per-ping values
        that never repeat are returned unchanged. The fields filled this
        way are recorded in the history entry. No effect with
        ``held_values="raw"``.
    groups : sequence of sequence of str, optional
        With ``held_values="fixes"``: variables that update together, passed
        to :func:`~sidescanpy.processing.aux_fixes.held_value_fixes` for each
        record (by aux variable name; for the per-ping fallback, by per-ping
        field name). A group applies to a record only when all its
        variables are in it, so ``groups=[POSITION_GROUP]`` joins latitude
        and longitude in the navigation record and leaves attitude alone.
        Default: every variable is its own group.
    merge_window_s : float or mapping of str to float, optional
        With ``held_values="fixes"``: merge changes at most this many
        seconds apart into one update event, for recorders that log one
        sensor update over consecutive entries. A float applies to every
        group of every record; a mapping by aux variable name (for the
        per-ping fallback, by per-ping field name), such as
        ``{"latitude": 0.15, "longitude": 0.15}``, only to those variables'
        groups. See :func:`~sidescanpy.processing.aux_fixes.held_value_fixes`.
        Default ``None``: no merging.

    Returns
    -------
    SideScanSurvey
        Survey with updated per-ping fields. Each touched frequency
        channel gets a new processing_history entry.

    Raises
    ------
    ValueError
        If ``fields`` references an unknown aux record name.
        If ``extrapolation`` or ``held_values`` is not one of the supported
        values, or ``max_hold_s`` or ``merge_window_s`` is out of range.
    """
    _validate_modes(extrapolation, held_values, max_hold_s, merge_window_s)

    field_map = fields if fields is not None else dict(DEFAULT_FIELD_SOURCES)

    # Validate aux names up front so user gets a fast error
    for ping_field, (aux_name, _aux_var) in field_map.items():
        if aux_name not in _VALID_AUX_NAMES:
            raise ValueError(
                f"Unknown aux record '{aux_name}' for field '{ping_field}'. "
                f"Valid names: {sorted(_VALID_AUX_NAMES)}"
            )

    target = survey if in_place else survey.copy()

    for freq_key, channel in list(target.freq_channels.items()):
        new_ds = interpolate_aux_onto_dataset(
            channel.dataset,
            target,
            fields=field_map,
            method=method,
            extrapolation=extrapolation,
            drop_invalid=drop_invalid,
            force=force,
            held_values=held_values,
            max_hold_s=max_hold_s,
            per_ping_fallback=per_ping_fallback,
            groups=groups,
            merge_window_s=merge_window_s,
        )
        target._freq_channels[freq_key] = SideScanFrequencyChannel(new_ds, freq_key=freq_key)

    return target


def interpolate_aux_onto_dataset(
    ds: xr.Dataset,
    aux: SideScanSurvey,
    *,
    fields: dict[str, tuple[str, str]] | None = None,
    method: str = "linear",
    extrapolation: Literal["clamp", "linear", "nan"] = "clamp",
    drop_invalid: bool = True,
    force: bool = False,
    held_values: Literal["fixes", "raw"] = "fixes",
    max_hold_s: float = 2.0,
    per_ping_fallback: bool = True,
    groups: Sequence[Sequence[str]] | None = None,
    merge_window_s: MergeWindow = None,
) -> xr.Dataset:
    """One frequency channel dataset's per-ping fields, interpolated from
    ``aux``'s auxiliary records (the ``"interpolate_aux"`` step).

    Same parameters as :func:`interpolate_aux_onto_pings`, but for a single
    dataset (or a ping slice of one) with the survey supplying the aux
    records. Always appends an ``"interpolate_aux"`` history record (with
    the fields actually updated, possibly none, ``held_values``,
    ``max_hold_s``, ``per_ping_fallback``, ``groups``, ``merge_window_s``
    and the fields filled from per-ping values as
    ``per_ping_fallback_fields``) so the step is not re-run.

    For the per-ping fallback on a ping slice, the per-ping values come
    from the matching whole channel in ``aux`` (same ``freq_hz_start``,
    same times at the slice's ``ping`` coordinates) when there is one, so
    a chunked run gives the same result as an unchunked one.
    """
    _validate_modes(extrapolation, held_values, max_hold_s, merge_window_s)
    field_map = fields if fields is not None else dict(DEFAULT_FIELD_SOURCES)
    for ping_field, (aux_name, _aux_var) in field_map.items():
        if aux_name not in _VALID_AUX_NAMES:
            raise ValueError(
                f"Unknown aux record '{aux_name}' for field '{ping_field}'. "
                f"Valid names: {sorted(_VALID_AUX_NAMES)}"
            )
    new_ds, touched, fallback = _interpolate_dataset_from_survey(
        ds,
        survey=aux,
        field_map=field_map,
        method=method,
        extrapolation=extrapolation,
        drop_invalid=drop_invalid,
        force=force,
        held_values=held_values,
        max_hold_s=max_hold_s,
        per_ping_fallback=per_ping_fallback,
        groups=groups,
        merge_window_s=merge_window_s,
    )
    return append_history(
        new_ds,
        STEP_INTERPOLATE_AUX,
        {
            "fields": sorted(touched),
            "method": method,
            "extrapolation": extrapolation,
            "drop_invalid": drop_invalid,
            "held_values": held_values,
            "max_hold_s": float(max_hold_s),
            "per_ping_fallback": bool(per_ping_fallback),
            "per_ping_fallback_fields": sorted(fallback),
            "groups": None if groups is None else [list(g) for g in groups],
            "merge_window_s": _merge_window_param(merge_window_s),
        },
        function="sidescanpy.processing.interpolate.interpolate_aux_onto_dataset",
        version=__version__,
    )


# ============================================================
# PUBLIC: LOWER-LEVEL ENTRYPOINT
# ============================================================

def interpolate_onto_channel(
    channel: SideScanFrequencyChannel,
    source: xr.Dataset,
    *,
    field_map: dict[str, str],
    method: str = "linear",
    extrapolation: Literal["clamp", "linear", "nan"] = "clamp",
    drop_invalid: bool = True,
    record_history: bool = True,
) -> SideScanFrequencyChannel:
    """
    Interpolate variables from an arbitrary source dataset onto one
    frequency channel's ping times.

    The source entries are used as given. For a sample-and-hold source
    (the latest value repeated until the sensor updates), pass
    ``held_value_fixes(source)`` to interpolate between real fixes.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Target channel.
    source : xr.Dataset
        Source dataset. Must have a ``time`` coordinate (datetime64-typed)
        and a single non-time dimension over which the variables are
        defined. Variables named in ``field_map`` values must exist.
    field_map : dict
        ``{per_ping_field: source_var_name}``.
    method : str, default "linear"
        Interpolation method passed to ``xarray.Dataset.interp``.
    extrapolation : {"clamp", "linear", "nan"}, default "clamp"
    drop_invalid : bool, default True
        Drop NaN rows from each source variable before interpolating.
    record_history : bool, default True
        If True, append a processing_history entry to the resulting
        dataset. Set False when called as part of a higher-level pipeline
        that records history once.

    Returns
    -------
    SideScanFrequencyChannel
        New channel with updated per-ping fields. Original is not modified.
    """
    if extrapolation not in _VALID_EXTRAPOLATION:
        raise ValueError(
            f"extrapolation must be one of {_VALID_EXTRAPOLATION}, got {extrapolation!r}"
        )

    if "time" not in source.coords and "time" not in source.dims:
        raise KeyError("Source dataset must have a 'time' coordinate")

    for ping_field, src_var in field_map.items():
        if src_var not in source.data_vars:
            raise KeyError(
                f"Source dataset missing variable '{src_var}' "
                f"(requested for field '{ping_field}')"
            )

    ping_times = channel.dataset["time"].values
    new_ds = channel.dataset

    for ping_field, src_var in field_map.items():
        interpolated = _interpolate_one_var(
            source=source,
            var_name=src_var,
            target_times=ping_times,
            method=method,
            extrapolation=extrapolation,
            drop_invalid=drop_invalid,
            circular=ping_field in CIRCULAR_FIELDS,
        )
        new_ds = new_ds.assign({ping_field: ("ping", interpolated)})

    if record_history:
        new_ds = _append_history_entry(
            new_ds,
            fields=list(field_map.keys()),
            method=method,
            extrapolation=extrapolation,
            drop_invalid=drop_invalid,
        )

    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


# ============================================================
# INTERNALS
# ============================================================

def _validate_modes(
    extrapolation: str, held_values: str, max_hold_s: float, merge_window_s: MergeWindow = None,
) -> None:
    if extrapolation not in _VALID_EXTRAPOLATION:
        raise ValueError(
            f"extrapolation must be one of {_VALID_EXTRAPOLATION}, got {extrapolation!r}"
        )
    if held_values not in _VALID_HELD_VALUES:
        raise ValueError(f"held_values must be one of {_VALID_HELD_VALUES}, got {held_values!r}")
    _check_hold_and_merge(max_hold_s, merge_window_s)


def _merge_window_param(merge_window_s: MergeWindow) -> float | dict[str, float] | None:
    """``merge_window_s`` as recorded in history."""
    if merge_window_s is None:
        return None
    if isinstance(merge_window_s, Mapping):
        return {k: float(v) for k, v in merge_window_s.items()}
    return float(merge_window_s)


def _groups_in(record: xr.Dataset, groups: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]] | None:
    """The groups whose variables are all in ``record``."""
    if groups is None:
        return None
    kept = [tuple(g) for g in groups if all(name in record.data_vars for name in g)]
    if len(kept) < len(groups):
        logger.debug(
            "interpolate_aux: groups %s not (fully) in this record, ignored for it",
            [list(g) for g in groups if tuple(g) not in kept],
        )
    return kept


def _interpolate_dataset_from_survey(
    ds: xr.Dataset,
    *,
    survey: SideScanSurvey,
    field_map: dict[str, tuple[str, str]],
    method: str,
    extrapolation: str,
    drop_invalid: bool,
    force: bool,
    held_values: str = "raw",
    max_hold_s: float = 2.0,
    per_ping_fallback: bool = False,
    groups: Sequence[Sequence[str]] | None = None,
    merge_window_s: MergeWindow = None,
) -> tuple[xr.Dataset, list[str], list[str]]:
    """Interpolate one channel dataset against a survey's aux records.

    Returns the new dataset, the fields updated, and the fields of those
    filled from the channel's own per-ping values. A field without a usable
    aux source falls back to its per-ping values when enabled and
    available, else is skipped (NaN-filled with force=True if the record is
    empty).
    """
    ping_times = ds["time"].values
    new_ds = ds
    fields_touched: list[str] = []
    fixes = held_values == "fixes"
    use_fallback = fixes and per_ping_fallback
    pending_fallback: list[str] = []

    def can_fall_back(ping_field: str) -> bool:
        return (
            use_fallback
            and ping_field in ds.data_vars
            and bool(np.isfinite(np.asarray(ds[ping_field].values, dtype=np.float64)).any())
        )

    # Group field_map by aux_name so we apply interpolation per aux record.
    by_aux: dict[str, dict[str, str]] = {}
    for ping_field, (aux_name, aux_var) in field_map.items():
        by_aux.setdefault(aux_name, {})[ping_field] = aux_var

    for aux_name, sub_map in by_aux.items():
        aux_ds = _get_aux_record(survey, aux_name)

        is_empty = aux_ds is None or aux_ds.sizes.get("obs", 0) == 0

        if is_empty:
            fb = [f for f in sub_map if can_fall_back(f)]
            pending_fallback.extend(fb)
            rest = [f for f in sub_map if f not in fb]
            if rest and not force:
                logger.warning(f"Skipping {sorted(rest)}: aux record '{aux_name}' is empty")
            elif rest:
                # Force NaN-fill since no source data exists
                for ping_field in rest:
                    nan_arr = np.full(ds.sizes["ping"], np.nan, dtype=np.float64)
                    new_ds = new_ds.assign({ping_field: ("ping", nan_arr)})
                    fields_touched.append(ping_field)
            continue

        source = held_value_fixes(
            aux_ds, groups=_groups_in(aux_ds, groups), max_hold_s=max_hold_s, merge_window_s=merge_window_s,
        ) if fixes else aux_ds

        # Standard path: interpolate each var in sub_map from the source
        for ping_field, aux_var in sub_map.items():
            if aux_var not in aux_ds.data_vars:
                if can_fall_back(ping_field):
                    pending_fallback.append(ping_field)
                    continue
                logger.warning(
                    f"Skipping '{ping_field}': "
                    f"variable '{aux_var}' not found in '{aux_name}' aux record"
                )
                continue

            # All-NaN source -> per-ping fallback, else preserve original
            source_values = aux_ds[aux_var].values
            if np.all(np.isnan(source_values)):
                if can_fall_back(ping_field):
                    pending_fallback.append(ping_field)
                    continue
                logger.warning(
                    f"Skipping '{ping_field}': '{aux_name}.{aux_var}' source "
                    f"is all NaN. Preserving original per-ping value."
                )
                continue

            interpolated = _interpolate_one_var(
                source=source,
                var_name=aux_var,
                target_times=ping_times,
                method=method,
                extrapolation=extrapolation,
                # Reduced records are NaN wherever a variable has no fix.
                drop_invalid=drop_invalid or fixes,
                circular=ping_field in CIRCULAR_FIELDS,
            )

            new_ds = new_ds.assign({ping_field: ("ping", interpolated)})
            fields_touched.append(ping_field)

    if pending_fallback:
        logger.debug(
            "interpolate_aux: no usable aux source for %s; interpolating the "
            "channel's own per-ping values reduced to held-value fixes",
            sorted(pending_fallback),
        )
        whole = _whole_channel(ds, survey)
        per_ping = xr.Dataset(
            {f: (("obs",), np.asarray(whole[f].values, dtype=np.float64)) for f in pending_fallback},
            coords={"time": (("obs",), np.asarray(whole["time"].values))},
        )
        source = held_value_fixes(
            per_ping, groups=_groups_in(per_ping, groups), max_hold_s=max_hold_s, merge_window_s=merge_window_s,
        )
        for ping_field in pending_fallback:
            interpolated = _interpolate_one_var(
                source=source,
                var_name=ping_field,
                target_times=ping_times,
                method=method,
                extrapolation=extrapolation,
                drop_invalid=True,
                circular=ping_field in CIRCULAR_FIELDS,
            )
            new_ds = new_ds.assign({ping_field: ("ping", interpolated)})
            fields_touched.append(ping_field)

    return new_ds, fields_touched, pending_fallback


def _whole_channel(ds: xr.Dataset, survey: SideScanSurvey) -> xr.Dataset:
    """The survey channel ``ds`` is (a ping slice of), else ``ds`` itself.

    Matched by ``freq_hz_start`` and identical times at ``ds``'s ``ping``
    coordinates, so the per-ping fallback sees the whole channel when the
    step runs chunk by chunk.
    """
    if "ping" not in ds.coords:
        return ds
    for channel in survey.freq_channels.values():
        whole = channel.dataset
        if whole is ds:
            return ds
        if whole.attrs.get("freq_hz_start") != ds.attrs.get("freq_hz_start") or "ping" not in whole.coords:
            continue
        if whole.sizes["ping"] < ds.sizes["ping"]:
            continue
        try:
            times = whole["time"].sel(ping=ds["ping"].values).values
        except (KeyError, IndexError):
            continue
        if np.array_equal(times, ds["time"].values):
            return whole
    return ds


def _get_aux_record(survey: SideScanSurvey, aux_name: str) -> xr.Dataset | None:
    """Resolve an aux record name to its dataset on the survey.

    Returns None when the aux record is not set or doesn't apply (e.g.
    'sensor' generic name in the high-level path; if needed, callers
    should target a specific sensor type via the lower-level API).
    """
    if aux_name == "navigation":
        return survey.navigation
    if aux_name == "attitude":
        return survey.attitude
    if aux_name == "dvl":
        return survey.dvl
    if aux_name == "sensor":
        # Ambiguous in the high-level entrypoint; require the user to
        # disambiguate via the low-level interpolate_onto_channel.
        return None
    return None


def _interpolate_one_var(
    *,
    source: xr.Dataset,
    var_name: str,
    target_times: np.ndarray,
    method: str,
    extrapolation: str,
    drop_invalid: bool,
    circular: bool = False,
) -> np.ndarray:
    """Interpolate one variable from `source` onto `target_times`.

    With ``circular=True`` the values are compass angles in degrees: the
    de-duplicated, time-sorted source is unwrapped (period 360),
    interpolated (extrapolation applies to the unwrapped series), and the
    result is wrapped back to [0, 360).

    Returns a 1D numpy array aligned with target_times.
    """
    da = source[var_name]
    time_coord = source["time"]

    # Drop NaN rows from the source variable before interpolating
    if drop_invalid:
        finite_mask = np.isfinite(da.values)
        if finite_mask.sum() == 0:
            # All source values are NaN; nothing to interpolate
            return np.full(len(target_times), np.nan, dtype=np.float64)

        if not finite_mask.all():
            obs_dim = [d for d in da.dims if d != "time"]
            obs_dim = obs_dim[0] if obs_dim else da.dims[0]
            da = da.isel({obs_dim: finite_mask})
            time_coord = time_coord.isel({obs_dim: finite_mask})

    # xarray.interp wants 'time' as a coordinate of the source array.
    # Build a clean DataArray with 'time' as the only dim.
    source_da = xr.DataArray(
        da.values,
        coords={"time": time_coord.values},
        dims=["time"],
    )

    # Drop duplicate timestamps (xarray.interp requires monotonic coord).
    # Keep first occurrence.
    _, unique_idx = np.unique(source_da["time"].values, return_index=True)
    if len(unique_idx) < source_da.sizes["time"]:
        source_da = source_da.isel(time=np.sort(unique_idx))

    n_source = source_da.sizes["time"]
    if n_source == 0:
        return np.full(len(target_times), np.nan, dtype=np.float64)

    if circular:
        vals = source_da.values.astype(np.float64)
        if np.isfinite(vals).all():
            vals = np.unwrap(vals, period=360.0)
        else:
            # drop_invalid=False with NaNs present: unwrap finite runs only
            fin = np.isfinite(vals)
            vals[fin] = np.unwrap(vals[fin], period=360.0)
            logger.debug("Circular field has NaN source values; unwrapped finite ones")
        source_da = source_da.copy(data=vals)

    # Single-point degenerate case — return constant
    if n_source == 1:
        const = float(source_da.values[0])
        if circular:
            const = const % 360.0
        return np.full(len(target_times), const, dtype=np.float64)

    # Build interp kwargs based on extrapolation policy
    if extrapolation == "linear":
        kwargs = {"fill_value": "extrapolate"}
    elif extrapolation == "clamp":
        first_val = float(source_da.values[0])
        last_val  = float(source_da.values[-1])
        kwargs = {"fill_value": (first_val, last_val), "bounds_error": False}
    else:  # "nan"
        kwargs = {"fill_value": np.nan, "bounds_error": False}

    interpolated = source_da.interp(
        time=target_times,
        method=method,
        kwargs=kwargs,
    )

    out = interpolated.values.astype(np.float64)
    if circular:
        out = np.mod(out, 360.0)
    return out


def _append_history_entry(
    ds: xr.Dataset,
    *,
    fields: list[str],
    method: str,
    extrapolation: str,
    drop_invalid: bool,
) -> xr.Dataset:
    """Append an ``"interpolate_onto_channel"`` processing_history record."""
    return append_history(
        ds,
        "interpolate_onto_channel",
        {
            "fields": fields,
            "method": method,
            "extrapolation": extrapolation,
            "drop_invalid": drop_invalid,
        },
        function="sidescanpy.processing.interpolate.interpolate_onto_channel",
        version=__version__,
    )
