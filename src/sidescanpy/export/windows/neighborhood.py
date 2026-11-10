"""
Bounded, block-averaged ping neighborhoods for geometric/georef windows.

A geometric or georef window is resampled from the pings around its
center. :func:`load_geo_neighborhood` loads only those pings, only the
variables the resample reads, nadir-masks them at native resolution, and
block-averages them down to about :data:`SWATH_OVERSAMPLE` source cells
per output pixel per axis -- loading a full-swath neighborhood of every
per-sample array at native resolution costs gigabytes per window on a
real survey.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.windows.levels import (
    mask_dropouts,
    mask_nadir_gap_with_uncorrected,
    nominal_sample_interval_m,
    with_db_var,
)
from sidescanpy.processing.navigation import along_track_distance_m

logger = logging.getLogger(__name__)

SWATH_OVERSAMPLE = 2.0
"""Source cells kept per output pixel, per axis, after block averaging."""

SWATH_LOAD_BLOCK_PINGS = 2048
"""Pings loaded and masked per piece in :func:`load_swath_neighborhood`."""

SWATH_LOAD_VARS = (
    "backscatter_intensity", "ground_range_m", "across_track_m",
    "latitude_per_sample", "longitude_per_sample", "latitude", "longitude", "sound_vel_ms",
)
"""Variables the geometric/georef resample and the nadir mask read."""

_SPACING_PROBE_PINGS = 500


def neighborhood_ping_spacing_m(ds_full: xr.Dataset, lo: int, hi: int) -> float:
    """Mean along-track meters per ping over pings ``[lo, hi)``.

    Navigation stalls (repeated fixes) are filled before measuring. ``0.0``
    for fewer than two pings.
    """
    lat_lon = ds_full[["latitude", "longitude"]].isel(ping=slice(lo, hi)).load()
    along = along_track_distance_m(lat_lon, fill_nav_stalls=True)
    return abs(float(along[-1] - along[0])) / max(along.size - 1, 1) if along.size > 1 else 0.0


def across_values(ds: xr.Dataset, across_axis: str) -> xr.DataArray:
    """Signed across-track coordinate (port negative) for a geometric window.

    ``"ground"``: ``across_track_m`` (ground-projected). ``"slant"``:
    ``slant_range_m`` with the same sign convention -- still meters, so a
    window covers the same real extent on both axes, but the water column
    is kept (``slant_range_m`` is finite everywhere; the nadir mask still
    blanks the backscatter there).

    Raises
    ------
    ValueError
        Unknown ``across_axis``.
    KeyError
        ``"slant"`` without ``slant_range_m`` in ``ds``.
    """
    if across_axis == "ground":
        return ds["across_track_m"]
    if across_axis != "slant":
        raise ValueError(f"unknown across_axis: {across_axis!r}")
    if "slant_range_m" not in ds:
        raise KeyError("slant_range_m")
    sign = xr.DataArray([-1.0, 1.0], dims="channel")
    return ds["slant_range_m"] * sign


def geo_decimation(
    ds_full: xr.Dataset, ping_idx: int, style: str, *, size_px: int, jitter_px: int, resolution_m: float | None,
    across_axis: str = "ground", swath_fill: bool = False, margin_m: float = 0.0,
) -> tuple[int, int, int]:
    """Neighborhood half-span and block-average factors for one style.

    The half-span (pings) covers the output window in meters --
    ``size_px * resolution_m`` along-track plus jitter and ``margin_m`` --
    with slack for turns and edge-shifting. The ``(ping, sample)`` factors
    keep about :data:`SWATH_OVERSAMPLE` source cells per output pixel.

    Parameters
    ----------
    ds_full : xr.Dataset
        The whole (lazy) frequency channel.
    ping_idx : int
        Window center ping.
    style : str
        ``"chip"``, ``"single-swath"`` or ``"dual-swath"``.
    size_px, jitter_px : int
        Output size and jitter of the window.
    resolution_m : float or None
        Output meters per pixel; ``None`` estimates the value
        :func:`~sidescanpy.export.windows.geo.extract_geo_style_window` will
        choose (chip: ``max(meters_per_sample, ping_spacing)``; swath
        styles: swath width / ``size_px``, floored at the ping spacing
        unless ``swath_fill``).
    across_axis : str, default "ground"
        See :func:`across_values`.
    swath_fill : bool, default False
        Estimate resolution the way ``swath_fill`` windows choose it.
    margin_m : float, default 0.0
        Extra along-track meters the neighborhood must cover beyond the
        window (background windows load enough to check nearby targets).

    Returns
    -------
    (half_span_pings, fp, fs)
    """
    n_pings = ds_full.sizes["ping"]
    mps = nominal_sample_interval_m(ds_full)
    probe = _SPACING_PROBE_PINGS
    ping_spacing_m = neighborhood_ping_spacing_m(ds_full, max(0, ping_idx - probe), min(n_pings, ping_idx + probe)) or mps
    if resolution_m is None:
        if style == "chip":
            resolution_m = max(mps, ping_spacing_m)
        else:
            across = across_values(ds_full.isel(ping=ping_idx), across_axis).values
            if style != "dual-swath":
                across = across[0]
            finite = across[np.isfinite(across)]
            swath_width_m = float(finite.max() - finite.min()) if finite.size else size_px * mps
            resolution_m = swath_width_m / size_px if swath_fill else max(swath_width_m / size_px, ping_spacing_m)
    half_extent_m = (size_px / 2.0 + jitter_px) * resolution_m + margin_m
    half_span = int(1.5 * half_extent_m / ping_spacing_m) + 50
    fs = max(1, int(resolution_m / (SWATH_OVERSAMPLE * mps)))
    fp = max(1, int(resolution_m / (SWATH_OVERSAMPLE * ping_spacing_m)))
    # Never average an axis down to (almost) nothing -- happens when the
    # other axis's spacing sets resolution_m (e.g. a swath narrower than
    # one along-track pixel).
    fs = min(fs, max(1, ds_full.sizes["sample"] // 8))
    fp = min(fp, max(1, (2 * half_span) // 8))
    return half_span, fp, fs


def load_swath_neighborhood(
    ds_full: xr.Dataset, lo: int, hi: int, fp: int, fs: int, *, with_db: bool, extra_vars: tuple[str, ...] = (),
    skip_vars: tuple[str, ...] = (),
) -> xr.Dataset:
    """Nadir- and dropout-masked pings ``[lo, hi)``, block-averaged by ``fp`` pings x ``fs`` samples.

    Loads only :data:`SWATH_LOAD_VARS` (plus ``extra_vars``, minus
    ``skip_vars``) in :data:`SWATH_LOAD_BLOCK_PINGS`-sized pieces, masking
    each piece at native resolution (nadir gap, and dropouts with
    :func:`~sidescanpy.export.windows.levels.mask_dropouts`) before
    averaging. ``sample_rate_hz`` is
    divided by ``fs`` so
    :func:`~sidescanpy.export.windows.levels.nominal_sample_interval_m`
    reports the averaged cell size.

    Parameters
    ----------
    ds_full : xr.Dataset
        The whole (lazy) frequency channel.
    lo, hi : int
        Ping range.
    fp, fs : int
        Ping and sample block-average factors (partial edge blocks are trimmed).
    with_db : bool
        Also add ``backscatter_db``.
    extra_vars, skip_vars : tuple of str
        Variables to add to / remove from :data:`SWATH_LOAD_VARS`.

    Returns
    -------
    xr.Dataset
    """
    keep = [v for v in (*SWATH_LOAD_VARS, *extra_vars) if v in ds_full and v not in skip_vars]
    step = fp * max(1, SWATH_LOAD_BLOCK_PINGS // fp)
    blocks = []
    for b0 in range(lo, hi, step):
        blk = mask_dropouts(mask_nadir_gap_with_uncorrected(ds_full[keep].isel(ping=slice(b0, min(hi, b0 + step))).load()))
        if with_db:
            blk = with_db_var(blk)
        if fp > 1 or fs > 1:
            blk = blk.coarsen(ping=fp, sample=fs, boundary="trim").mean(keep_attrs=True)
        blocks.append(blk)
    out = xr.concat(blocks, dim="ping") if len(blocks) > 1 else blocks[0]
    out.attrs = {**ds_full.attrs, "sample_rate_hz": float(ds_full.attrs["sample_rate_hz"]) / fs}
    return out


def decimate_target(target: Target, fs: int, fp: int) -> Target:
    """``target`` in a block-averaged neighborhood's ``(fp, fs)`` pixel units.

    Moves ``sample`` (integer division) and scales ``bbox`` (x by ``fs``,
    y by ``fp``).
    """
    t = target
    bbox = None if t.bbox is None else (t.bbox[0] / fs, t.bbox[1] / fp, t.bbox[2] / fs, t.bbox[3] / fp)
    return replace(t, sample=None if t.sample is None else t.sample // fs, bbox=bbox)


@dataclass
class GeoNeighborhood:
    """A loaded, block-averaged neighborhood for geometric/georef windows.

    Attributes
    ----------
    ds : xr.Dataset
        The averaged neighborhood (see :func:`load_swath_neighborhood`).
    target : Target or None
        The window's primary target in averaged units (``None`` for a
        window with no primary target).
    local_ping : int
        Center ping, indexed into ``ds``.
    mps : float
        Averaged meters per sample.
    items : dict
        ``{channel: [(target, ping), ...]}`` co-target candidates, in
        averaged units; ``ping`` maps file ping ``p`` to
        ``lo + (p - lo) // fp`` so that ``ping - lo`` indexes ``ds``.
    lo : int
        File ping index of the neighborhood's first ping.
    fp, fs : int
        Ping and sample averaging factors.
    """

    ds: xr.Dataset
    target: Target | None
    local_ping: int
    mps: float
    items: dict
    lo: int
    fp: int = 1
    fs: int = 1

    def to_averaged_sample(self, sample: int) -> int:
        """A native sample index in this neighborhood's averaged sample units."""
        return sample // self.fs


def load_geo_neighborhood(
    ds_full: xr.Dataset, target: Target | None, ping_idx: int, items_by_channel: dict, styles: list[str], *,
    size_px: int, jitter_px: int, resolution_m: float | None, with_db: bool, across_axis: str = "ground",
    per_sample_latlon: bool = True, swath_fill: bool = False, margin_m: float = 0.0,
) -> GeoNeighborhood:
    """Block-averaged neighborhood around ``ping_idx`` serving every style in ``styles``.

    Uses the widest half-span and finest averaging factors any of
    ``styles`` needs (see :func:`geo_decimation`), and translates ``target``
    and ``items_by_channel`` into the averaged units.

    Parameters
    ----------
    ds_full : xr.Dataset
        The whole (lazy) frequency channel.
    target : Target or None
        Primary target (``None`` for a background window).
    ping_idx : int
        Center ping (file index).
    items_by_channel : dict
        ``{channel: [(target, file_ping_idx), ...]}`` co-target candidates.
    styles : list of str
    size_px, jitter_px, resolution_m, across_axis, swath_fill, margin_m
        See :func:`geo_decimation`.
    with_db : bool
        Add ``backscatter_db``.
    per_sample_latlon : bool, default True
        Load ``latitude_per_sample``/``longitude_per_sample`` (only the
        georef level reads them).

    Returns
    -------
    GeoNeighborhood
        With ``across_axis="slant"``, ``ds["across_track_m"]`` is replaced
        by signed slant range (see :func:`across_values`), so every
        downstream geometric lookup works in slant meters.
    """
    n_pings = ds_full.sizes["ping"]
    plans = [
        geo_decimation(
            ds_full, ping_idx, st, size_px=size_px, jitter_px=jitter_px, resolution_m=resolution_m,
            across_axis=across_axis, swath_fill=swath_fill, margin_m=margin_m,
        )
        for st in styles
    ]
    half_span = max(p[0] for p in plans)
    fp = min(p[1] for p in plans)
    fs = min(p[2] for p in plans)
    lo = max(0, ping_idx - half_span)
    hi = min(n_pings, ping_idx + half_span)
    extra_vars = ("slant_range_m",) if across_axis == "slant" else ()
    skip_vars = () if per_sample_latlon else ("latitude_per_sample", "longitude_per_sample")
    ds = load_swath_neighborhood(ds_full, lo, hi, fp, fs, with_db=with_db, extra_vars=extra_vars, skip_vars=skip_vars)
    if across_axis == "slant":
        ds["across_track_m"] = across_values(ds, "slant")
    items = {
        ch: [(decimate_target(tt, fs, fp), lo + (p - lo) // fp) for tt, p in ch_its]
        for ch, ch_its in items_by_channel.items()
    }
    label = "background" if target is None else f"target {target.target_id}"
    logger.debug(f"{label} {styles}: {hi - lo} pings averaged {fp}x{fs} (ping x sample) -> {dict(ds.sizes)}")
    return GeoNeighborhood(
        ds=ds, target=None if target is None else decimate_target(target, fs, fp), local_ping=(ping_idx - lo) // fp,
        mps=nominal_sample_interval_m(ds), items=items, lo=lo, fp=fp, fs=fs,
    )
