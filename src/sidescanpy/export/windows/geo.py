"""
Real-meters export windows: the ``geometric`` and ``georef`` render levels.

:func:`extract_geo_style_window` resamples a loaded neighborhood (see
:mod:`sidescanpy.export.windows.neighborhood`) into a ``size_px`` square
grid at one ``resolution_m`` on both axes, for any style, and places the
window's targets in it. ``geometric`` grids are along/across-track meters;
``georef`` grids are UTM northing/easting.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import xarray as xr

from sidescanpy.core.target import Target
from sidescanpy.export.augment import jitter_latlon, jitter_offset_m
from sidescanpy.export.geometric_window import (
    extract_geometric_window,
    extract_geometric_window_binned,
    extract_geometric_window_griddata,
)
from sidescanpy.export.geometric_window import locate_point as locate_point_geometric
from sidescanpy.export.georef_window import extract_georef_window
from sidescanpy.export.georef_window import locate_point as locate_point_georef
from sidescanpy.export.orientation import orient_geo_window
from sidescanpy.export.windows.pixel import RenderedWindow, with_bbox_size_px
from sidescanpy.processing.navigation import along_track_distance_m

logger = logging.getLogger(__name__)

SWATH_FILL_PCT = 10.0
"""Percentile of per-ping reach that sets a ``swath_fill`` frame's edges.

The far edge is where at least ``100 - SWATH_FILL_PCT`` % of pings still
have data; the near (nadir-side, single swath only) edge is where at least
that share already do. Longer-reaching pings are cropped slightly instead
of every row carrying an empty margin."""

RESAMPLE_METHODS = ("kdtree", "binned", "griddata")
"""Geometric-level resamplers (see :func:`extract_geo_style_window`)."""


def swath_fill_extent(
    across: np.ndarray, values: np.ndarray, *, combine: bool, channel_idx: int | None,
) -> tuple[float, float] | None:
    """Across-track ``(lo, hi)`` meters the recorded data fills, for a swath frame.

    A sample counts only where both its coordinate and its backscatter are
    finite (in slant mode the coordinate is finite everywhere, including the
    masked nadir gap).

    Parameters
    ----------
    across, values : np.ndarray
        ``(ping, channel, sample)`` signed across-track meters and backscatter.
    combine : bool
        ``True`` (dual swath): port's far edge to starboard's far edge, not
        centered on nadir; a side with no data mirrors the other so nadir
        stays centered. ``False`` (single swath): ``channel_idx``'s near
        edge to its far edge.
    channel_idx : int or None
        Channel for ``combine=False``.

    Returns
    -------
    tuple of float or None
        ``None`` when there is no usable data.
    """
    valid = np.isfinite(across) & np.isfinite(values)
    mag = np.where(valid, np.abs(across), np.nan)

    def reach(ch):
        m = mag[:, ch, :]
        rows = np.isfinite(m).any(axis=1)
        if not rows.any():
            return None
        with np.errstate(all="ignore"):
            far = float(np.percentile(np.nanmax(m[rows], axis=1), SWATH_FILL_PCT))
            near = float(np.percentile(np.nanmin(m[rows], axis=1), 100.0 - SWATH_FILL_PCT))
        return near, far

    if combine:
        port, stbd = reach(0), reach(1)
        if port is None and stbd is None:
            return None
        port_far = port[1] if port else stbd[1]
        stbd_far = stbd[1] if stbd else port[1]
        return -port_far, stbd_far
    r = reach(channel_idx)
    if r is None:
        return None
    near, far = r
    if not far > near:
        return None
    return (-far, -near) if channel_idx == 0 else (near, far)


def geo_targets_within_window(
    candidates: list[tuple[Target, int]], masked: xr.Dataset, lo: int, along_dist: np.ndarray, size_px: int, meta: dict,
    locate_fn: Callable[[float, float, dict], tuple[float, float]], along_size_px: int | None = None, *,
    margin_px: float = 0.0,
) -> list[Target]:
    """Candidates whose real position falls inside a geometric window.

    The real-world counterpart of
    :func:`~sidescanpy.export.windows.targets_within_window`: each
    candidate's along-track distance (at its ping) and across-track meters
    (at its own channel and sample) are located in the window with
    ``locate_fn`` and kept if inside the grid. Candidates outside
    ``masked``'s pings or sample axis, or with no finite across-track
    value, are skipped.

    Parameters
    ----------
    candidates : list of (Target, int)
        Targets and their pings, where ``ping - lo`` indexes ``masked``.
    masked : xr.Dataset
        The neighborhood the window was resampled from.
    lo : int
        Ping offset of ``masked``.
    along_dist : np.ndarray
        Along-track distance per ``masked`` ping, as used for the window.
    size_px : int
        Grid column count (and row count unless ``along_size_px``).
    meta : dict
        The window extractor's ``meta``.
    locate_fn : callable
        ``(along_m, across_m, meta) -> (row, col)``, e.g.
        :func:`~sidescanpy.export.geometric_window.locate_point`.
    along_size_px : int, optional
        Grid row count when it differs from ``size_px``.
    margin_px : float, default 0.0
        Also keep candidates up to this many pixels outside the grid (for
        background-window exclusion).

    Returns
    -------
    list of Target
        Copies with window-local ``sample`` (column) and
        ``extra_attributes["window_row"]``, before orientation (see
        :func:`~sidescanpy.export.orientation.flip_geo_vertical`).
    """
    row_limit = along_size_px if along_size_px is not None else size_px
    included = []
    for candidate_t, ping_idx in candidates:
        local_ping = ping_idx - lo
        if not (0 <= local_ping < masked.sizes["ping"]):
            continue
        c_idx = candidate_t.channel if candidate_t.channel is not None else 0
        if candidate_t.sample is None or not (0 <= candidate_t.sample < masked.sizes["sample"]):
            continue
        try:
            t_across = float(masked["across_track_m"].isel(ping=local_ping, channel=c_idx, sample=candidate_t.sample).values)
        except (KeyError, IndexError):
            continue
        if not np.isfinite(t_across):
            continue
        t_along = float(along_dist[local_ping])
        row, col = locate_fn(t_along, t_across, meta)
        if -margin_px <= row < row_limit + margin_px and -margin_px <= col < size_px + margin_px:
            included.append(replace(candidate_t, sample=col, extra_attributes={**candidate_t.extra_attributes, "window_row": row}))
    return included


def georef_targets_within_window(
    candidates: list[tuple[Target, int]], masked: xr.Dataset, lo: int, meta: dict, size_px: int, *, margin_px: float = 0.0,
) -> list[Target]:
    """Candidates whose per-sample lat/lon falls inside a georef window.

    Like :func:`geo_targets_within_window`, but each candidate is located by
    ``latitude_per_sample``/``longitude_per_sample`` at its ping, channel
    and sample with :func:`~sidescanpy.export.georef_window.locate_point`.
    Candidates without a finite position are skipped.

    Returns
    -------
    list of Target
        Copies with window-local ``sample``/``extra_attributes["window_row"]``.
    """
    if "latitude_per_sample" not in masked or "longitude_per_sample" not in masked:
        return []
    included = []
    for candidate_t, ping_idx in candidates:
        local_ping = ping_idx - lo
        if not (0 <= local_ping < masked.sizes["ping"]):
            continue
        c_idx = candidate_t.channel if candidate_t.channel is not None else 0
        if candidate_t.sample is None or not (0 <= candidate_t.sample < masked.sizes["sample"]):
            continue
        lat = float(masked["latitude_per_sample"].isel(ping=local_ping, channel=c_idx, sample=candidate_t.sample).values)
        lon = float(masked["longitude_per_sample"].isel(ping=local_ping, channel=c_idx, sample=candidate_t.sample).values)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        row, col = locate_point_georef(lat, lon, meta)
        if -margin_px <= row < size_px + margin_px and -margin_px <= col < size_px + margin_px:
            included.append(replace(candidate_t, sample=col, extra_attributes={**candidate_t.extra_attributes, "window_row": row}))
    return included


def extract_geo_style_window(
    masked: xr.Dataset,
    *,
    center_sample: int,
    channel_idx: int,
    local_ping: int,
    style: str,
    level: str,
    size_px: int,
    jitter_px: int,
    rng: np.random.Generator,
    mps: float,
    items_by_channel: dict,
    lo: int,
    primary: Target | None = None,
    resolution_m_override: float | None = None,
    backscatter_var: str = "backscatter_intensity",
    resample_method: str = "griddata",
    swath_fill: bool = False,
) -> RenderedWindow | None:
    """``geometric`` or ``georef`` window for one style, with its targets.

    Resolution (unless ``resolution_m_override``): chip -- the native
    ``max(meters_per_sample, ping_spacing)``; swath styles -- the swath
    width over ``size_px``, floored at the ping spacing. With
    ``swath_fill`` (swath styles, no override) the frame instead spans the
    across-track extent the data fills (:func:`swath_fill_extent`) at
    square pixels, with no ping-spacing floor and not pinned to nadir.

    ``geometric`` windows include every co-located candidate from
    ``items_by_channel`` (both channels for dual swath) and are oriented
    with :func:`~sidescanpy.export.orientation.orient_geo_window`.
    ``georef`` windows carry only the primary target and are not
    reoriented (rows follow northing).

    ``resample_method`` selects the geometric resampler: ``"kdtree"``
    (average of measurements within a search radius; no fabricated values,
    some directional blur), ``"binned"`` (per-cell mean of the samples
    inside it; empty cells stay NaN) or ``"griddata"`` (scipy
    interpolation; no gaps inside coverage, but at a resolution finer than
    the ping spacing most cells are interpolated). ``georef`` always uses
    :func:`~sidescanpy.export.georef_window.extract_georef_window`.

    Parameters
    ----------
    masked : xr.Dataset
        Loaded neighborhood (see
        :class:`~sidescanpy.export.windows.neighborhood.GeoNeighborhood`).
    center_sample : int
        Window center on ``channel_idx``'s sample axis (``masked`` units).
    channel_idx : int
        Channel of the center sample (and of a single-channel window).
    local_ping : int
        Center ping, indexed into ``masked``.
    style, level : str
        See :mod:`sidescanpy.export.windows.levels`.
    size_px, jitter_px : int
        Output size and center jitter (pixels at ``resolution_m``).
    rng : np.random.Generator
    mps : float
        ``masked``'s meters per sample.
    items_by_channel : dict
        ``{channel: [(target, ping), ...]}``, ``ping - lo`` indexing ``masked``.
    lo : int
    primary : Target, optional
        The window's own target (positioned at ``center_sample``). ``None``
        for a background window: every candidate is searched and the
        result lists only candidates found inside.
    resolution_m_override : float, optional
    backscatter_var : str, default "backscatter_intensity"
        Variable resampled into the grid.
    resample_method : str, default "griddata"
    swath_fill : bool, default False

    Returns
    -------
    RenderedWindow or None
        ``None`` when the window can't be built: center sample outside
        ``masked``, geometry/georef variables missing, no finite geometry
        around the center, or (georef, or a background chip) a center
        sample with no position.
    """
    if center_sample is None or not (0 <= center_sample < masked.sizes["sample"]):
        return None
    try:
        along_dist = along_track_distance_m(masked, fill_nav_stalls=True)
        center_ping_along_m = float(along_dist[local_ping])
        center_ping_across_m = float(masked["across_track_m"].isel(ping=local_ping, channel=channel_idx, sample=center_sample).values)
    except KeyError:
        return None

    ping_span_m = float(along_dist[-1] - along_dist[0]) if along_dist.size > 1 else 0.0
    ping_spacing_m = abs(ping_span_m) / max(along_dist.size - 1, 1) if along_dist.size > 1 else mps
    along_diffs = np.abs(np.diff(along_dist)) if along_dist.size > 1 else np.array([])
    nonzero_along_diffs = along_diffs[along_diffs > 0]
    gap_p95_m = float(np.percentile(nonzero_along_diffs, 95)) if nonzero_along_diffs.size else 0.0

    combine = style == "dual-swath"
    if primary is None and style == "chip" and not np.isfinite(center_ping_across_m):
        return None  # a background chip centered in the water column has no across-track position
    if style == "chip":
        resolution_m = resolution_m_override if resolution_m_override is not None else max(mps, ping_spacing_m)
        center_across_m = center_ping_across_m + jitter_offset_m(jitter_px, resolution_m, rng)
    elif swath_fill and resolution_m_override is None and (
        extent := swath_fill_extent(masked["across_track_m"].values, masked[backscatter_var].values, combine=combine, channel_idx=channel_idx)
    ):
        lo_m, hi_m = extent
        resolution_m = max((hi_m - lo_m) / size_px, 1e-6)
        center_across_m = (lo_m + hi_m) / 2.0
    else:
        across_all = masked["across_track_m"].values if combine else masked["across_track_m"].isel(channel=channel_idx).values
        across_finite = across_all[np.isfinite(across_all)]
        swath_width_m = float(across_finite.max() - across_finite.min()) if across_finite.size else size_px * mps
        resolution_m = resolution_m_override if resolution_m_override is not None else max(swath_width_m / size_px, ping_spacing_m, 1e-6)
        center_across_m = 0.0 if combine else (float((across_finite.max() + across_finite.min()) / 2.0) if across_finite.size else 0.0)
    # Georef's isotropic KDTree search radius: at least the largest
    # typical gap between pings.
    geo_search_radius_m = max(1.5 * resolution_m, gap_p95_m)

    if level == "geometric":
        geom_channel_idx = None if combine else channel_idx
        center_along_m = center_ping_along_m + jitter_offset_m(jitter_px, resolution_m, rng)
        across_search_radius_m = max(1.5 * resolution_m, 1.5 * mps)
        common = dict(
            resolution_m=resolution_m, center_along_m=center_along_m, center_across_m=center_across_m,
            channel_idx=geom_channel_idx, avoid_edge_gaps=True, backscatter_var=backscatter_var,
        )
        # TODO(lib): three resamplers over one point cloud -- "kdtree" is
        # grids' radius-mean in a local along/across frame and "binned" is
        # scipy binned_statistic_2d; both could call one shared grids
        # aggregation (grids.resample._aggregate_cells on a frame-agnostic
        # grid) instead of separate implementations. Needs a parity test
        # per method against the current chip outputs.
        try:
            if resample_method == "kdtree":
                grid, meta = extract_geometric_window(
                    masked, size_px, along_search_radius_m=geo_search_radius_m, across_search_radius_m=across_search_radius_m, **common,
                )
            elif resample_method == "binned":
                grid, meta = extract_geometric_window_binned(masked, size_px, **common)
            else:
                grid, meta = extract_geometric_window_griddata(masked, size_px, **common)
        except ValueError:
            return None
        grid_shape = grid.shape
        all_items = [item for items in items_by_channel.values() for item in items] if combine else items_by_channel.get(channel_idx, [])
        positioned = []
        if primary is not None:
            row, col = locate_point_geometric(center_ping_along_m, center_ping_across_m, meta)
            positioned.append(replace(primary, sample=col, extra_attributes={**primary.extra_attributes, "window_row": row}))
            all_items = [item for item in all_items if item[0].target_id != primary.target_id]
        positioned += geo_targets_within_window(all_items, masked, lo, along_dist, size_px, meta, locate_point_geometric)
        grid, targets_out = orient_geo_window(grid, positioned, size_px, combine=combine, channel_idx=channel_idx)
        # t.bbox's (w, h) is in source sample(x) / ping(y) counts: to meters
        # via mps / ping_spacing_m, then to output pixels via resolution_m.
        targets_out = [with_bbox_size_px(ct, scale_x=mps / resolution_m, scale_y=ping_spacing_m / resolution_m) for ct in targets_out]
        targets_out = [replace(ct, extra_attributes={**ct.extra_attributes, "window_resolution_m": resolution_m}) for ct in targets_out]
        return RenderedWindow(image=grid, targets=targets_out, meta=meta, grid_shape=grid_shape, resolution_m=resolution_m)

    # georef
    try:
        center_lat = float(masked["latitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=center_sample).values)
        center_lon = float(masked["longitude_per_sample"].isel(ping=local_ping, channel=channel_idx, sample=center_sample).values)
    except KeyError:
        return None
    if not (np.isfinite(center_lat) and np.isfinite(center_lon)):
        return None  # center sample has no position (water column / nadir)
    north_jitter_m = jitter_offset_m(jitter_px, resolution_m, rng)
    east_jitter_m = jitter_offset_m(jitter_px, resolution_m, rng)
    jittered_lat, jittered_lon = jitter_latlon(center_lat, center_lon, north_jitter_m, east_jitter_m)
    georef_input = masked
    if not combine:
        excluded = 1 - channel_idx
        georef_input = masked.copy()
        georef_input[backscatter_var] = georef_input[backscatter_var].copy()
        georef_input[backscatter_var][{"channel": excluded}] = np.nan
    try:
        raster, meta = extract_georef_window(
            georef_input, size_px, resolution_m=resolution_m, center_lat=jittered_lat, center_lon=jittered_lon,
            max_search_radius_m=geo_search_radius_m, backscatter_var=backscatter_var,
        )
    except (ValueError, KeyError):
        return None
    grid = raster.dataset["backscatter"].values
    if grid.shape != (size_px, size_px):
        padded = np.full((size_px, size_px), np.nan)
        r = min(grid.shape[0], size_px)
        c = min(grid.shape[1], size_px)
        padded[:r, :c] = grid[:r, :c]
        grid = padded
    targets_out = []
    if primary is not None:
        row, col = locate_point_georef(center_lat, center_lon, meta)
        primary_local = replace(primary, sample=col, extra_attributes={**primary.extra_attributes, "window_row": row})
        primary_local = with_bbox_size_px(primary_local, scale_x=mps / resolution_m, scale_y=ping_spacing_m / resolution_m)
        primary_local = replace(primary_local, extra_attributes={**primary_local.extra_attributes, "window_resolution_m": resolution_m})
        targets_out.append(primary_local)
    return RenderedWindow(image=grid, targets=targets_out, meta=meta, grid_shape=grid.shape, resolution_m=resolution_m)
