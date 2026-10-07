"""
Undo recorded corrections, driven by ``processing_history``.

No uncorrected copy of the backscatter is stored: radiometric steps are
inverted exactly from the parameters recorded in their history records, and
geometric/georef steps are undone by dropping the variables they added.

- :func:`uncorrected_backscatter` -- the pre-radiometric signal as a
  DataArray (nothing else touched). Works lazily on dask-backed data.
- :func:`invert` -- a dataset with the chosen steps undone and their history
  records removed.

Not invertible: ``crop_farrange`` (samples are gone), ``interpolate_aux``
(the converter's embedded per-ping values were overwritten) and caller-supplied
steps (unknown effect).

Legacy files: TVG records written by older versions (``tvg_correction``) carry
the same parameters and invert the same way. A legacy file whose only TVG
evidence is the ``processing_level == "radiometric"`` attribute falls back to
its ``backscatter_raw`` variable.

2026 rshomberg@uri.edu
"""

import logging
from collections.abc import Iterable

import xarray as xr

from sidescanpy.core.history import (
    BUILTIN_STEPS,
    RADIOMETRIC_STEPS,
    STEP_GEOMETRIC,
    STEP_GEOREF,
    STEP_OUTPUT_VARS,
    STEP_RANGE_CORRECTION,
    STEP_TVG,
    canonical_step,
    has_step,
    read_history,
    remove_history_records,
)
from sidescanpy.core.units import FLOOR_DB, db_to_linear
from sidescanpy.processing.radiometric import tvg_gain_for_dataset, tvg_params_from_record
from sidescanpy.processing.range_correction import RangeCorrectionTable, range_correction_db

logger = logging.getLogger(__name__)

INVERTIBLE_STEPS: tuple[str, ...] = (STEP_TVG, STEP_RANGE_CORRECTION, STEP_GEOMETRIC, STEP_GEOREF)

_FLOOR_TOLERANCE_DB = 1e-2
"""Inverted dB values within this of the -120 dB floor map back to exact 0
amplitude (the forward step clipped exact zeros to the floor)."""


def _radiometric_records(ds: xr.Dataset) -> list[tuple[int, str, dict]]:
    return [
        (i, canonical_step(str(r.get("step", ""))), r)
        for i, r in enumerate(read_history(ds))
        if canonical_step(str(r.get("step", ""))) in RADIOMETRIC_STEPS
    ]


def _warn_custom_after(ds: xr.Dataset, first_index: int) -> None:
    custom = [
        str(r.get("step"))
        for i, r in enumerate(read_history(ds))
        if i > first_index and canonical_step(str(r.get("step", ""))) not in BUILTIN_STEPS
    ]
    if custom:
        logger.warning(
            "custom step(s) %s were applied after radiometric corrections; inversion "
            "cannot undo them, so the uncorrected backscatter may still include their effect",
            custom,
        )


def _undo_tvg(ds: xr.Dataset, db: xr.DataArray, record: dict) -> xr.DataArray:
    params = tvg_params_from_record(record.get("params", {}))
    gain = xr.DataArray(tvg_gain_for_dataset(ds, params), dims=["sample"])
    uncorrected_db = db - gain.astype(db.dtype)
    linear = db_to_linear(uncorrected_db)
    return xr.where(uncorrected_db <= FLOOR_DB + _FLOOR_TOLERANCE_DB, 0.0, linear).astype(db.dtype)


def _undo_range_correction(ds: xr.Dataset, db: xr.DataArray, record: dict) -> xr.DataArray:
    if "ground_range_m" not in ds:
        raise ValueError("cannot invert range_correction: ground_range_m is not in the dataset")
    params = record.get("params", {})
    if "range_m" in params and "correction_db" in params:
        range_m, correction_db = params["range_m"], params["correction_db"]
    else:
        # Legacy record: no curve stored; it was always the bundled default table.
        logger.debug("legacy range_correction record without its curve; using the bundled default table")
        table = RangeCorrectionTable.load_default()
        matched = params.get("freq_hz_matched") or table.matched_frequency(ds.attrs["freq_hz_start"])
        if matched is None:
            raise ValueError("cannot invert legacy range_correction record: no matching default curve")
        curve = table._curves[table.matched_frequency(float(matched))]
        range_m, correction_db = curve.range_m, curve.correction_db
    return (db - range_correction_db(ds, range_m, correction_db)).astype(db.dtype)


_UNDO = {STEP_TVG: _undo_tvg, STEP_RANGE_CORRECTION: _undo_range_correction}


def _undo_radiometric(ds: xr.Dataset, records: list[tuple[int, str, dict]]) -> xr.DataArray:
    """Undo ``records`` (radiometric, oldest first) on ``backscatter_intensity``, newest first."""
    values = ds["backscatter_intensity"]
    for _i, step, record in reversed(records):
        values = _UNDO[step](ds, values, record)
    return values


def _legacy_raw(ds: xr.Dataset) -> xr.DataArray | None:
    """``backscatter_raw`` for a legacy file whose TVG is known only from its attribute."""
    if has_step(ds, STEP_TVG) and not _radiometric_records(ds):
        if "backscatter_raw" not in ds:
            raise ValueError(
                "legacy dataset is marked processing_level='radiometric' but has no TVG "
                "history record and no backscatter_raw; cannot recover the uncorrected signal"
            )
        logger.debug("legacy radiometric dataset without TVG record; using backscatter_raw")
        return ds["backscatter_raw"]
    return None


def uncorrected_backscatter(ds: xr.Dataset) -> xr.DataArray:
    """``backscatter_intensity`` with every recorded radiometric step undone.

    Equals ``backscatter_intensity`` when no radiometric step was applied.
    Exact-zero input amplitude (clipped to -120 dB by TVG) comes back as 0.
    Lazy for dask-backed input. Does not modify ``ds``.
    """
    legacy = _legacy_raw(ds)
    if legacy is not None:
        return legacy
    records = _radiometric_records(ds)
    if not records:
        return ds["backscatter_intensity"]
    _warn_custom_after(ds, records[0][0])
    return _undo_radiometric(ds, records)


def invert(ds: xr.Dataset, steps: Iterable[str] | None = None) -> xr.Dataset:
    """Undo recorded correction steps.

    Parameters
    ----------
    ds : xr.Dataset
        Frequency channel dataset.
    steps : iterable of str, optional
        Steps to undo, any of :data:`INVERTIBLE_STEPS`. Default: every one
        that has been applied. Radiometric steps are undone newest first and
        must include every radiometric step recorded after the oldest one
        requested (``tvg`` cannot be undone underneath a later
        ``range_correction``). Undoing ``geometric`` also undoes ``georef``
        (it is derived from the geometric variables) and requires
        ``range_correction`` to be undone too if it was applied (it needs
        ``ground_range_m``). Steps that were never applied are ignored.

    Returns
    -------
    xr.Dataset
        New dataset: ``backscatter_intensity`` restored, geometric/georef
        output variables dropped, and the undone steps' history records
        removed. Records of non-invertible steps (``crop_farrange``,
        ``interpolate_aux``, custom steps) are kept.

    Raises
    ------
    ValueError
        For a non-invertible or unknown step name, or a request that would
        leave a dependent step behind (see above).
    """
    requested = set(INVERTIBLE_STEPS) if steps is None else {canonical_step(s) for s in steps}
    unknown = requested - set(INVERTIBLE_STEPS)
    if unknown:
        raise ValueError(f"cannot invert {sorted(unknown)}; invertible steps: {INVERTIBLE_STEPS}")
    if STEP_GEOMETRIC in requested:
        requested.add(STEP_GEOREF)

    out = ds
    remove: set[int] = set()

    legacy = _legacy_raw(ds) if STEP_TVG in requested else None
    records = _radiometric_records(ds)
    if legacy is not None:
        logger.debug("invert: restoring backscatter_intensity from legacy backscatter_raw")
        out = out.assign(backscatter_intensity=legacy)
    elif records:
        wanted = [k for k, (_i, step, _r) in enumerate(records) if step in requested]
        if wanted:
            to_undo = records[wanted[0]:]
            left_behind = sorted({step for _i, step, _r in to_undo if step not in requested})
            if left_behind:
                raise ValueError(
                    f"cannot invert {sorted(requested & set(RADIOMETRIC_STEPS))} without also inverting "
                    f"{left_behind}, applied after it"
                )
            _warn_custom_after(ds, to_undo[0][0])
            logger.debug("invert: undoing %s", [step for _i, step, _r in reversed(to_undo)])
            out = out.assign(backscatter_intensity=_undo_radiometric(ds, to_undo))
            remove |= {i for i, _s, _r in to_undo}

    remaining_rc = any(step == STEP_RANGE_CORRECTION for i, step, _r in records if i not in remove)
    if STEP_GEOMETRIC in requested and remaining_rc:
        raise ValueError("cannot invert geometric while range_correction (which needs ground_range_m) stays applied")

    for step in (STEP_GEOMETRIC, STEP_GEOREF):
        if step not in requested:
            continue
        drop = [v for v in STEP_OUTPUT_VARS[step] if v in out.variables]
        if drop:
            logger.debug("invert: dropping %s variables %s", step, drop)
            out = out.drop_vars(drop)
        remove |= {i for i, r in enumerate(read_history(ds)) if canonical_step(str(r.get("step", ""))) == step}

    if remove or legacy is not None:
        out = remove_history_records(out, remove)
    return out
