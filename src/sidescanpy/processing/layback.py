"""
Layback: move per-ping positions from the ship to the towfish.

A towed sonar sits behind (and below) the navigation antenna. When the
recorded per-ping ``latitude`` / ``longitude`` are the ship's, the
``"layback"`` step moves each ping astern along its heading by the
tow-point offset plus the layback, and to starboard by the tow-point
offset, before ``geometric`` and ``georef`` place the samples:

    aft       = tow_point_aft_m + layback_m
    starboard = tow_point_starboard_m

This is a fixed offset astern (the first-order model MB-System applies
from recorded layback and lever arms). The towfish is assumed directly
astern of the tow point on the ping's ``heading_deg``; course made good,
cable catenary and tractrix models of a fish lagging in turns are not
modelled.

Never double-applied: positions move only when the dataset's
``position_source`` (set by the reader, see :mod:`sidescanpy.core.schema`)
is ``"ship"``. ``"towfish"`` (the acquisition software already laid it
back), ``"mixed"`` and ``"unknown"`` leave the dataset unchanged and log a
WARNING saying why; a caller who knows better passes ``position_source``.

Per-ping layback comes from ``layback_m`` (a scalar parameter overrides
the variable). Where it is NaN and ``use_cable_out`` is set, the
Pythagorean estimate ``sqrt(cable_out_m² − depth_m²)`` is used (tow-point
height ignored). Pings left without a layback or heading are not moved
and are counted in the history record.

The recorded positions are kept as ``latitude_before_layback`` /
``longitude_before_layback``, so :func:`sidescanpy.processing.inversion.invert`
can undo the step.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pyproj
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_GEOREF, STEP_LAYBACK, append_history, has_step
from sidescanpy.core.schema import (
    OPTIONAL_PER_PING_VARS,
    POSITION_SHIP,
    POSITION_SOURCE_ATTR,
    POSITION_SOURCE_REASON_ATTR,
    POSITION_SOURCES,
    POSITION_TOWFISH,
    POSITION_UNKNOWN,
)

logger = logging.getLogger(__name__)

LAT_BEFORE_VAR = "latitude_before_layback"
LON_BEFORE_VAR = "longitude_before_layback"
LAYBACK_MODEL = "fixed_offset_astern"


def _per_ping_layback(
    ds: xr.Dataset, layback_m: float | None, use_cable_out: bool,
) -> tuple[np.ndarray, int, int]:
    """Layback per ping, and how many pings took it from the record / from cable out."""
    n = ds.sizes["ping"]
    if layback_m is not None:
        return np.full(n, float(layback_m)), n, 0
    values = (
        ds["layback_m"].values.astype(np.float64) if "layback_m" in ds else np.full(n, np.nan)
    )
    n_recorded = int(np.isfinite(values).sum())
    n_cable = 0
    if use_cable_out and "cable_out_m" in ds:
        cable = ds["cable_out_m"].values.astype(np.float64)
        depth = ds["depth_m"].values.astype(np.float64)
        with np.errstate(invalid="ignore"):
            pythagorean = np.sqrt(cable ** 2 - depth ** 2)
        fill = ~np.isfinite(values) & np.isfinite(pythagorean) & (cable > 0)
        values = np.where(fill, pythagorean, values)
        n_cable = int(fill.sum())
    return values, n_recorded, n_cable


def apply_layback(
    ds: xr.Dataset,
    *,
    layback_m: float | None = None,
    tow_point_aft_m: float | None = None,
    tow_point_starboard_m: float | None = None,
    position_source: str | None = None,
    use_cable_out: bool = True,
    ellipsoid: str = "WGS84",
) -> xr.Dataset:
    """Move ship positions to the towfish (the ``"layback"`` step).

    Parameters
    ----------
    ds : xr.Dataset
        Frequency channel dataset.
    layback_m : float, optional
        Layback for every ping (m). Default: the ``layback_m`` variable.
    tow_point_aft_m, tow_point_starboard_m : float, optional
        Antenna-to-tow-point offset (m, aft and starboard positive).
        Default: the dataset attributes of the same name, else 0.
    position_source : str, optional
        What ``latitude`` / ``longitude`` are the position of, one of
        :data:`sidescanpy.core.schema.POSITION_SOURCES`. Default: the
        dataset's ``position_source`` attribute (unknown when absent).
        Positions move only for ``"ship"``.
    use_cable_out : bool, default True
        Where ``layback_m`` is NaN, estimate it as
        ``sqrt(cable_out_m² − depth_m²)``.
    ellipsoid : str, default "WGS84"
        Ellipsoid for the geodesic offsets.

    Returns
    -------
    xr.Dataset
        New dataset. When positions moved: ``latitude`` / ``longitude``
        at the towfish, the recorded ones in ``latitude_before_layback`` /
        ``longitude_before_layback``, ``position_source`` set to
        ``"towfish"`` and a ``"layback"`` history record. Otherwise the
        input, unchanged and without a record.

    Raises
    ------
    ValueError
        If the step was already applied, ``georef`` was applied before it
        (its per-sample positions would go stale; invert it first), or
        ``position_source`` is not a known value.
    """
    source = position_source or ds.attrs.get(POSITION_SOURCE_ATTR, POSITION_UNKNOWN)
    if source not in POSITION_SOURCES:
        raise ValueError(f"position_source {source!r} is not one of {POSITION_SOURCES}")
    if LAT_BEFORE_VAR in ds or has_step(ds, STEP_LAYBACK):
        raise ValueError("layback was already applied to this dataset")
    if has_step(ds, STEP_GEOREF):
        raise ValueError(
            "georef was applied before layback; invert georef first so the per-sample "
            "positions are recomputed from the laid-back ping positions"
        )
    if source != POSITION_SHIP:
        reason = ds.attrs.get(POSITION_SOURCE_REASON_ATTR, "no position_source recorded")
        logger.warning(
            "layback not applied: position_source is %r, not 'ship' (%s); pass "
            "position_source='ship' if the positions are known to be the ship's",
            source, reason,
        )
        return ds

    aft_lever = float(tow_point_aft_m if tow_point_aft_m is not None else ds.attrs.get("tow_point_aft_m", 0.0))
    stbd_lever = float(
        tow_point_starboard_m if tow_point_starboard_m is not None else ds.attrs.get("tow_point_starboard_m", 0.0)
    )
    layback, n_recorded, n_cable = _per_ping_layback(ds, layback_m, use_cable_out)

    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    heading = ds["heading_deg"].values.astype(np.float64)
    aft = aft_lever + layback
    movable = np.isfinite(aft) & np.isfinite(heading) & np.isfinite(lat) & np.isfinite(lon)
    n_unmoved = int((~movable).sum())
    if n_unmoved:
        logger.warning(
            "layback: %d of %d pings have no layback or heading and keep their ship position",
            n_unmoved, movable.size,
        )

    geod = pyproj.Geod(ellps=ellipsoid)
    new_lon, new_lat = lon.copy(), lat.copy()
    if movable.any():
        lo, la, _ = geod.fwd(lon[movable], lat[movable], heading[movable] + 180.0, aft[movable])
        if stbd_lever:
            lo, la, _ = geod.fwd(lo, la, heading[movable] + 90.0, np.full(lo.shape, stbd_lever))
        new_lon[movable], new_lat[movable] = lo, la

    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)
    out = out.assign(
        latitude=(("ping",), new_lat, ds["latitude"].attrs),
        longitude=(("ping",), new_lon, ds["longitude"].attrs),
        **{
            LAT_BEFORE_VAR: (("ping",), lat, {"units": "degrees_north", "long_name": OPTIONAL_PER_PING_VARS[LAT_BEFORE_VAR]}),
            LON_BEFORE_VAR: (("ping",), lon, {"units": "degrees_east", "long_name": OPTIONAL_PER_PING_VARS[LON_BEFORE_VAR]}),
        },
    )
    out.attrs[POSITION_SOURCE_ATTR] = POSITION_TOWFISH
    out.attrs[POSITION_SOURCE_REASON_ATTR] = "ship positions moved astern by the layback step"

    finite = layback[np.isfinite(layback)]
    out = append_history(
        out,
        STEP_LAYBACK,
        {
            "model": LAYBACK_MODEL,
            "direction": "astern along heading_deg",
            "layback_m": layback_m,
            "tow_point_aft_m": aft_lever,
            "tow_point_starboard_m": stbd_lever,
            "use_cable_out": use_cable_out,
            "position_source": source,
            "position_source_reason": ds.attrs.get(POSITION_SOURCE_REASON_ATTR),
            "n_pings_layback_recorded": n_recorded,
            "n_pings_layback_from_cable_out": n_cable,
            "n_pings_not_moved": n_unmoved,
            "layback_m_median": float(np.median(finite)) if finite.size else None,
            "ellipsoid": ellipsoid,
        },
        function="sidescanpy.processing.layback.apply_layback",
        version=__version__,
    )
    logger.debug(
        "apply_layback: moved %d of %d pings astern (median layback %s m, tow point aft %.2f stbd %.2f)",
        int(movable.sum()), movable.size, f"{np.median(finite):.2f}" if finite.size else "n/a", aft_lever, stbd_lever,
    )
    return out


def undo_layback(ds: xr.Dataset, record: dict) -> xr.Dataset:
    """Restore the recorded positions and position source; drop the kept copies.

    The history record is not removed here (:func:`~sidescanpy.processing.inversion.invert` does that).
    """
    if LAT_BEFORE_VAR not in ds or LON_BEFORE_VAR not in ds:
        raise ValueError(f"cannot invert layback: {LAT_BEFORE_VAR} / {LON_BEFORE_VAR} are not in the dataset")
    out = ds.assign(
        latitude=(("ping",), ds[LAT_BEFORE_VAR].values, ds["latitude"].attrs),
        longitude=(("ping",), ds[LON_BEFORE_VAR].values, ds["longitude"].attrs),
    ).drop_vars([LAT_BEFORE_VAR, LON_BEFORE_VAR])
    out.attrs = dict(out.attrs)
    params = record.get("params", {})
    out.attrs[POSITION_SOURCE_ATTR] = params.get("position_source", POSITION_SHIP)
    reason = params.get("position_source_reason")
    if reason is None:
        out.attrs.pop(POSITION_SOURCE_REASON_ATTR, None)
    else:
        out.attrs[POSITION_SOURCE_REASON_ATTR] = reason
    return out
