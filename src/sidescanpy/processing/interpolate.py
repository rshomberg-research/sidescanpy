"""
Interpolate auxiliary records onto per-ping timestamps.

The converter populates per-ping fields (latitude, heading_deg, etc.) on each
SideScanFrequencyChannel from the embedded values in SonarDataMessage. For
many quantities those embedded values are stale — they're the most-recent
auxiliary reading at the moment the ping was recorded, *not* interpolated
to the ping's exact timestamp.

This module provides higher-precision per-ping values by interpolating
each frequency channel's ping times against the survey's auxiliary records.

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

import json
import logging
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import xarray as xr

from sidescanpy import __version__
from sidescanpy.core.survey import SideScanFrequencyChannel, SideScanSurvey

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


_VALID_AUX_NAMES = {"navigation", "attitude", "dvl", "sensor"}
_VALID_EXTRAPOLATION = ("clamp", "linear", "nan")


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
    force : bool, default False
        If True, attempt interpolation even when the aux record is empty
        or absent. Per-ping field receives NaN. If False (default), the
        field is left unchanged and a warning is logged.
    in_place : bool, default False
        If True, mutate ``survey`` in place and return it. If False,
        return a new SideScanSurvey leaving the input untouched.

    Returns
    -------
    SideScanSurvey
        Survey with updated per-ping fields. Each touched frequency
        channel gets a new processing_history entry.

    Raises
    ------
    ValueError
        If ``fields`` references an unknown aux record name.
        If ``extrapolation`` is not one of the supported values.
    """
    if extrapolation not in _VALID_EXTRAPOLATION:
        raise ValueError(
            f"extrapolation must be one of {_VALID_EXTRAPOLATION}, got {extrapolation!r}"
        )

    field_map = fields if fields is not None else dict(DEFAULT_FIELD_SOURCES)

    # Validate aux names up front so user gets a fast error
    for ping_field, (aux_name, _aux_var) in field_map.items():
        if aux_name not in _VALID_AUX_NAMES:
            raise ValueError(
                f"Unknown aux record '{aux_name}' for field '{ping_field}'. "
                f"Valid names: {sorted(_VALID_AUX_NAMES)}"
            )

    target = survey if in_place else _shallow_copy_survey(survey)

    for freq_key, channel in list(target.freq_channels.items()):
        new_channel = _interpolate_channel_from_survey(
            channel,
            survey=target,
            field_map=field_map,
            method=method,
            extrapolation=extrapolation,
            drop_invalid=drop_invalid,
            force=force,
        )
        target._freq_channels[freq_key] = new_channel

    return target


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

def _interpolate_channel_from_survey(
    channel: SideScanFrequencyChannel,
    *,
    survey: SideScanSurvey,
    field_map: dict[str, tuple[str, str]],
    method: str,
    extrapolation: str,
    drop_invalid: bool,
    force: bool,
) -> SideScanFrequencyChannel:
    """Interpolate one channel against a survey's aux records.

    Skips fields whose aux record is empty/absent unless force=True.
    """
    ping_times = channel.dataset["time"].values
    new_ds = channel.dataset
    fields_touched: list[str] = []

    # Group field_map by aux_name so we apply interpolation per aux record.
    by_aux: dict[str, dict[str, str]] = {}
    for ping_field, (aux_name, aux_var) in field_map.items():
        by_aux.setdefault(aux_name, {})[ping_field] = aux_var

    for aux_name, sub_map in by_aux.items():
        aux_ds = _get_aux_record(survey, aux_name)

        is_empty = aux_ds is None or aux_ds.sizes.get("obs", 0) == 0

        if is_empty and not force:
            logger.warning(
                f"Skipping {sorted(sub_map.keys())}: "
                f"aux record '{aux_name}' is empty"
            )
            continue

        if is_empty and force:
            # Force NaN-fill since no source data exists
            for ping_field in sub_map.keys():
                nan_arr = np.full(channel.n_pings, np.nan, dtype=np.float64)
                new_ds = new_ds.assign({ping_field: ("ping", nan_arr)})
                fields_touched.append(ping_field)
            continue

        # Standard path: interpolate each var in sub_map from aux_ds
        for ping_field, aux_var in sub_map.items():
            if aux_var not in aux_ds.data_vars:
                logger.warning(
                    f"Skipping '{ping_field}': "
                    f"variable '{aux_var}' not found in '{aux_name}' aux record"
                )
                continue

            # NEW: check if source variable is all NaN → preserve original
            source_values = aux_ds[aux_var].values
            if np.all(np.isnan(source_values)):
                logger.warning(
                    f"Skipping '{ping_field}': '{aux_name}.{aux_var}' source "
                    f"is all NaN. Preserving original per-ping value."
                )
                continue

            interpolated = _interpolate_one_var(
                source=aux_ds,
                var_name=aux_var,
                target_times=ping_times,
                method=method,
                extrapolation=extrapolation,
                drop_invalid=drop_invalid,
            )

            new_ds = new_ds.assign({ping_field: ("ping", interpolated)})
            fields_touched.append(ping_field)

    if fields_touched:
        new_ds = _append_history_entry(
            new_ds,
            fields=sorted(fields_touched),
            method=method,
            extrapolation=extrapolation,
            drop_invalid=drop_invalid,
        )

    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


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
) -> np.ndarray:
    """Interpolate one variable from `source` onto `target_times`.

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

    # Single-point degenerate case — return constant
    if n_source == 1:
        return np.full(len(target_times),
                       float(source_da.values[0]),
                       dtype=np.float64)

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

    return interpolated.values.astype(np.float64)


def _append_history_entry(
    ds: xr.Dataset,
    *,
    fields: list[str],
    method: str,
    extrapolation: str,
    drop_invalid: bool,
) -> xr.Dataset:
    """Append a processing_history entry recording the interpolation step.

    Returns a new dataset (xarray attrs are mutable; we copy to avoid
    side effects on the caller's dataset).
    """
    new_ds = ds.copy(deep=False)
    new_ds.attrs = dict(ds.attrs)

    history_str = new_ds.attrs.get("processing_history", "[]")
    try:
        history = json.loads(history_str)
        if not isinstance(history, list):
            history = []
    except (TypeError, ValueError):
        history = []

    entry = {
        "step":         "interpolate_aux_onto_pings",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "fields":        fields,
            "method":        method,
            "extrapolation": extrapolation,
            "drop_invalid":  drop_invalid,
        },
        "function": "sidescanpy.processing.interpolate.interpolate_aux_onto_pings",
        "version":  __version__,
    }
    history.append(entry)
    new_ds.attrs["processing_history"] = json.dumps(history)
    return new_ds


def _shallow_copy_survey(survey: SideScanSurvey) -> SideScanSurvey:
    """Make a shallow copy of a SideScanSurvey: same aux/sensor refs (they
    aren't modified), but a new dict for _freq_channels so we can replace
    individual channels without touching the input.
    """
    new = SideScanSurvey(
        source_files=list(survey.source_files),
        source_format=survey.source_format,
        survey_name=survey.survey_name,
    )
    # Copy refs — channels will be replaced one at a time below
    new._freq_channels = dict(survey._freq_channels)
    new._navigation = survey._navigation
    new._attitude   = survey._attitude
    new._dvl        = survey._dvl
    new._sensors    = dict(survey._sensors)
    return new