"""
Geometric (along-track x across-track meters) windowed export.

Resamples a bounded region directly at an explicit ``(resolution_m,
size_px)``, centered on a given real-world point — a survey-local frame
(no CRS/UTM projection, unlike ``export.georef_window``), using the
per-sample ``across_track_m`` that
``processing/geometric.py::compute_geometric_ranges`` already attaches,
and the per-ping cumulative along-track distance from
``viz.waterfall.along_track_distance_m``.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from sidescanpy.viz.waterfall import along_track_distance_m


def extract_geometric_window(
    channel_ds: xr.Dataset,
    size_px: int,
    *,
    resolution_m: float,
    center_along_m: float,
    center_across_m: float,
    channel_idx: int | None = 0,
    backscatter_var: str = "backscatter_intensity",
) -> tuple[np.ndarray, dict]:
    """Resample a ``size_px x size_px`` square region, centered on
    ``(center_along_m, center_across_m)``, at exactly ``resolution_m``
    meters/pixel on both axes.

    Both axes always share the same ``resolution_m`` and ``size_px`` by
    construction — no derived-from-extent resolution, no separate resize
    step, so the result is never stretched relative to the real world
    (a previous version computed ``resolution_m`` from the *larger* of
    the window's own along/across extents but still forced *both* axes
    to ``size_px`` samples — the smaller-extent axis came out visibly
    stretched; fixed by making resolution and size the only inputs,
    extent purely a consequence of them).

    ``channel_idx=None`` pools samples from *both* channels into the
    same point cloud instead of one side only — unlike the raw-pixel
    ``export.windows`` primitives, this needs no explicit mirroring:
    ``across_track_m`` is already a real-world coordinate (port negative,
    starboard positive), so both sides combine correctly by construction,
    giving a single image spanning the full swath width with the nadir
    gap naturally in the middle (wherever ``across_track_m`` is NaN/near
    zero on both sides).

    Parameters
    ----------
    channel_ds : xr.Dataset
        Already ping-sliced to a neighborhood comfortably containing the
        requested ``size_px * resolution_m`` extent around the center —
        this function does not itself bound how much source data it
        searches; a much wider slice than needed just costs more KDTree
        build time for no benefit.
    size_px : int
        Output side length in pixels.
    resolution_m : float
        Meters per pixel, both axes. Pick small (near the channel's own
        sample interval) for a maximum-detail close-up; pick
        ``real_swath_width_m / size_px`` to fit an entire swath width
        (both channels) into the frame.
    center_along_m, center_across_m : float
        Real-world center of the requested window, in the same
        along-track-distance / ``across_track_m`` units the survey's own
        per-sample geometry uses (e.g. a target's own position, read
        directly from ``across_track_m`` at its ping/sample/channel).
    channel_idx : int or None, default 0
        Restrict to one physical channel (0=port, 1=starboard), or
        ``None`` to pool both.

    Returns
    -------
    (grid, meta)
        ``grid`` : ``(size_px, size_px)`` array, NaN where no sample
        fell within the search radius of a cell.
        ``meta`` : dict with ``center_row``/``center_col`` (the
        requested center's own pixel position in ``grid`` — always the
        exact middle, since the grid is built centered on it by
        construction) and the resolved ``along_min``/``across_min``
        bounds, for locating other real-world points in this same grid
        later.

    Raises
    ------
    ValueError
        If no finite ``across_track_m``/backscatter samples exist in
        ``channel_ds`` at all (e.g. entirely water column).
    """
    along = along_track_distance_m(channel_ds)  # (ping,)
    if channel_idx is None:
        across = channel_ds["across_track_m"].values  # (ping, channel, sample)
        values = channel_ds[backscatter_var].values
        along_grid = np.broadcast_to(along[:, None, None], across.shape)
    else:
        across = channel_ds["across_track_m"].isel(channel=channel_idx).values  # (ping, sample)
        values = channel_ds[backscatter_var].isel(channel=channel_idx).values
        along_grid = np.broadcast_to(along[:, None], across.shape)

    finite = np.isfinite(across) & np.isfinite(values)
    if not finite.any():
        raise ValueError("no finite across_track_m/backscatter samples in this window")

    along_valid = along_grid[finite]
    across_valid = across[finite]
    values_valid = values[finite]

    half_extent = size_px * resolution_m / 2.0
    along_min = center_along_m - half_extent
    across_min = center_across_m - half_extent

    tree = cKDTree(np.column_stack([along_valid, across_valid]))

    cell_centers = (np.arange(size_px) + 0.5) * resolution_m
    grid_along = along_min + cell_centers
    grid_across = across_min + cell_centers
    gg_along, gg_across = np.meshgrid(grid_along, grid_across, indexing="ij")
    query_points = np.column_stack([gg_along.ravel(), gg_across.ravel()])

    search_radius = 1.5 * resolution_m
    grid_values = np.full(query_points.shape[0], np.nan)
    for i, pt in enumerate(query_points):
        idx = tree.query_ball_point(pt, r=search_radius)
        if idx:
            grid_values[i] = np.mean(values_valid[idx])

    grid = grid_values.reshape(size_px, size_px)
    meta = {
        "center_row": (center_along_m - along_min) / resolution_m,
        "center_col": (center_across_m - across_min) / resolution_m,
        "along_min": along_min,
        "across_min": across_min,
        "resolution_m": resolution_m,
    }
    return grid, meta


def locate_point(along_m: float, across_m: float, meta: dict) -> tuple[float, float]:
    """Translate a real-world ``(along_m, across_m)`` point into pixel
    ``(row, col)`` coordinates within a grid produced by
    :func:`extract_geometric_window`, using that call's own returned
    ``meta`` — for marking a *different* point than the one the window
    was centered on (e.g. a second target sharing the same frame).
    """
    return (
        (along_m - meta["along_min"]) / meta["resolution_m"],
        (across_m - meta["across_min"]) / meta["resolution_m"],
    )
