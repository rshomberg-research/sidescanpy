"""
Geometric (along-track x across-track meters) windowed export.

Resamples a bounded region directly at an explicit ``(resolution_m,
size_px)``, centered on a given real-world point — a survey-local frame
(no CRS/UTM projection, unlike ``export.georef_window``), using the
per-sample ``across_track_m`` that
``processing/geometric.py::compute_geometric_ranges`` already attaches,
and the per-ping cumulative along-track distance from
``processing.navigation.along_track_distance_m``.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

from sidescanpy.processing.navigation import along_track_distance_m

logger = logging.getLogger(__name__)


def _clamp_center(center: float, half_extent: float, data_min: float, data_max: float) -> float:
    """Shift a window center so its full extent fits inside
    ``[data_min, data_max]`` when possible, instead of leaving genuine
    no-data gaps at one edge -- mirrors ``export.windows._place_window_1d``'s
    own non-padding placement for the raw-pixel case (``extract_chip``/
    ``extract_full_channel``). When the available extent is itself
    smaller than the requested window, centers on the available extent
    instead (still can't avoid *some* no-data margin, same as the
    pixel-space case falling back to the full array).

    Always returns a plain Python ``float`` regardless of ``data_min``/
    ``data_max``'s own dtype (real bug found downstream: ``along_valid``/
    ``across_valid`` are numpy float32 arrays, so an un-cast ``.min()``/
    ``.max()`` silently taints ``center_along_m``/``center_across_m`` --
    and everything computed from them (``meta``, target row/col) -- with
    ``np.float32``, which ``json.dumps`` (COCO export) can't serialize).
    """
    if data_max - data_min <= 2 * half_extent:
        return float((data_min + data_max) / 2.0)
    return float(min(max(center, data_min + half_extent), data_max - half_extent))


def extract_geometric_window(
    channel_ds: xr.Dataset,
    size_px: int,
    *,
    resolution_m: float,
    center_along_m: float,
    center_across_m: float,
    channel_idx: int | None = 0,
    backscatter_var: str = "backscatter_intensity",
    search_radius_m: float | None = None,
    along_search_radius_m: float | None = None,
    across_search_radius_m: float | None = None,
    avoid_edge_gaps: bool = False,
    along_resolution_m: float | None = None,
    across_resolution_m: float | None = None,
    along_size_px: int | None = None,
    across_size_px: int | None = None,
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
    extent purely a consequence of them). This is still the default —
    ``along_resolution_m``/``across_resolution_m``/``along_size_px``/
    ``across_size_px`` below are opt-in, and every existing caller that
    doesn't pass them gets byte-identical square-pixel output.

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
    search_radius_m : float, optional
        Isotropic KDTree query radius (same in both directions);
        defaults to ``1.5 * resolution_m``. A survey whose along-track
        position updates at a much lower rate than its ping rate (real
        case: a Klein SDF survey whose GPS fixes only every ~8 pings,
        jumping ~2m each time) has real gaps between along-track fixes
        wider than a resolution chosen from the *average* distance/ping
        can bridge at the default radius — the result grid shows
        periodic empty stripes at exactly the fix-to-fix interval,
        since most cells' nearest real point sits outside
        ``1.5 * resolution_m``. Passing a wider radius (e.g. the real
        fix-to-fix jump distance) bridges those gaps without also
        needing a coarser (more zoomed-out) ``resolution_m``. Ignored
        if either of the two params below is given.
    along_search_radius_m, across_search_radius_m : float, optional
        Independent along-track / across-track search radii — an
        ellipse instead of a circle, in real (along_m, across_m) space.
        Each axis is rescaled by its own radius before the ``cKDTree``
        is built/queried (then queried with a unit ball), which is
        exactly equivalent to an elliptical search with no custom
        distance metric needed. Default (``None``) for either falls
        back to ``search_radius_m`` (or its own default) — fully
        backward compatible with every existing caller.

        Real motivation (2026-09-28): a swath's along-track (ping
        spacing) and across-track (sample interval) native resolutions
        can differ by an order of magnitude or more. The single
        isotropic ``search_radius_m`` this function used to have had to
        be wide enough to bridge real along-track gaps — which meant
        *every* cell's average also pooled far more across-track real
        detail than the data's own resolution supports, regardless of
        how fine ``resolution_m`` was set. A caller requesting a finer
        ``resolution_m`` for more real detail got more pixels of an
        equally over-smoothed image, not more detail — confirmed
        directly on real data as the cause of a "chips are blurry"
        report. Passing a narrow ``across_search_radius_m`` (sized to
        the real sample interval) alongside a wide
        ``along_search_radius_m`` (still sized to bridge real ping
        gaps) fixes this: real across-track texture is preserved
        instead of averaged away, without reintroducing along-track
        gaps.
    avoid_edge_gaps : bool, default False
        When True and the requested center is close enough to the edge
        of this channel's own recorded extent that the window would
        otherwise include genuine no-data cells (e.g. a target near the
        max-range edge of the swath), shift the center inward just
        enough to keep the full window inside real data instead -- same
        tradeoff ``extract_chip`` already makes for raw-pixel windows
        near an array edge. The target itself is then no longer
        necessarily at the window's exact center; ``locate_point``
        against the *original* requested point still reports its true
        (now off-center) position correctly. Default ``False`` keeps
        the window exactly centered on the requested point regardless
        (this function's original, unconditional behavior) -- opt in
        per caller, since not every use of this function is a
        target-crop where "no data" is worse than "off-center".
    along_resolution_m, across_resolution_m : float, optional
        Independent along-track / across-track meters-per-pixel — real,
        non-square output pixels. Default (``None``) for either falls
        back to ``resolution_m`` — fully backward compatible, every
        existing caller gets exactly the prior square-pixel behavior.

        Real motivation (2026-09-29): even with anisotropic *search
        radii* (``along_search_radius_m``/``across_search_radius_m``
        above), a single shared ``resolution_m`` still forces the SAME
        pixel density on both axes. On a survey whose along-track
        native resolution is much coarser than across-track (ping
        spacing an order of magnitude or more wider than sample
        interval — one real survey measured at ~16:1), the
        along-track search radius floor (tied to real ping-to-ping
        gaps, independent of ``resolution_m`` — see
        ``along_search_radius_m`` above) ends up wide relative to a
        fine shared pixel spacing, visible in the output as elliptical
        averaging footprints — confirmed directly, real user report
        ("you can clearly see the ellipses which appear wide... that
        is not an issue w/ the radiometric"). No amount of tuning a
        single shared ``resolution_m``/search radius eliminates this,
        because a plain pixel-space resize (which is what "radiometric"
        looks like) never averages a real neighborhood at all, so it
        has no footprint to show — it isn't solving the same problem.
        Setting ``along_resolution_m`` coarser than
        ``across_resolution_m`` (close to the real ping spacing) lets
        each axis's pixel size match its OWN real resolving power,
        eliminating the visible-footprint artifact without averaging
        away real across-track detail that IS there to resolve.
    along_size_px, across_size_px : int, optional
        Independent output row / column counts. Default (``None``) for
        either falls back to ``size_px``. Combined with
        ``along_resolution_m``/``across_resolution_m``, this is what
        actually makes the output array non-square in shape, not just
        in real-world meters-per-pixel — e.g. ``along_size_px=20`` at
        ``along_resolution_m=0.4`` covers the same real 8m as
        ``across_size_px=320`` at ``across_resolution_m=0.025``, and
        the returned array is genuinely ``(20, 320)``, not square. A
        caller displaying this gets correct proportions for free from
        most plotting libraries' default aspect handling (e.g.
        matplotlib ``imshow``'s ``aspect="equal"``, driven by array
        shape alone, with no ``extent`` needed) — it is not a stretched
        square image, it is a genuinely shorter/narrower real array.

    Returns
    -------
    (grid, meta)
        ``grid`` : ``(along_size_px, across_size_px)`` array (both
        default to ``size_px``, reproducing the prior square shape
        unless overridden). NaN where no sample fell within the search
        radius of a cell.
        ``meta`` : dict with ``center_row``/``center_col`` (the
        requested center's own pixel position in ``grid`` — always the
        exact middle, since the grid is built centered on it by
        construction), the resolved ``along_min``/``across_min``
        bounds, ``along_resolution_m``/``across_resolution_m`` (the
        resolved per-axis values, always present even when isotropic),
        and ``resolution_m`` — a legacy key kept for backward
        compatibility, equal to ``along_resolution_m``; new code should
        read the per-axis keys instead, since this key is ambiguous
        once the two differ.

    Raises
    ------
    ValueError
        If no finite ``across_track_m``/backscatter samples exist in
        ``channel_ds`` at all (e.g. entirely water column).
    """
    along = along_track_distance_m(channel_ds, fill_nav_stalls=True)  # (ping,)
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

    # Resolved per-axis resolution/size -- default to the shared
    # resolution_m/size_px, reproducing the prior square-pixel behavior
    # exactly when neither is overridden.
    along_resolution_m = along_resolution_m if along_resolution_m is not None else resolution_m
    across_resolution_m = across_resolution_m if across_resolution_m is not None else resolution_m
    along_size_px = along_size_px if along_size_px is not None else size_px
    across_size_px = across_size_px if across_size_px is not None else size_px

    along_half_extent = along_size_px * along_resolution_m / 2.0
    across_half_extent = across_size_px * across_resolution_m / 2.0
    if avoid_edge_gaps:
        center_along_m = _clamp_center(center_along_m, along_half_extent, along_valid.min(), along_valid.max())
        center_across_m = _clamp_center(center_across_m, across_half_extent, across_valid.min(), across_valid.max())
    along_min = center_along_m - along_half_extent
    across_min = center_across_m - across_half_extent

    # Isotropic default radius still keys off the required, shared
    # resolution_m param -- unaffected by along_resolution_m/
    # across_resolution_m, so every existing caller's radius behavior
    # (which never passes those) is completely unchanged.
    default_radius = search_radius_m if search_radius_m is not None else 1.5 * resolution_m
    along_radius = along_search_radius_m if along_search_radius_m is not None else default_radius
    across_radius = across_search_radius_m if across_search_radius_m is not None else default_radius

    # Anisotropic search via axis rescaling: dividing each axis by its
    # own radius before building/querying the tree, then querying a
    # unit ball, is exactly equivalent to an ellipse with semi-axes
    # (along_radius, across_radius) in real space -- no custom distance
    # metric needed, cKDTree handles the rescaled Euclidean case
    # natively. Isotropic (along_radius == across_radius) reproduces
    # the prior behavior exactly.
    tree = cKDTree(np.column_stack([along_valid / along_radius, across_valid / across_radius]))

    along_cell_centers = (np.arange(along_size_px) + 0.5) * along_resolution_m
    across_cell_centers = (np.arange(across_size_px) + 0.5) * across_resolution_m
    grid_along = along_min + along_cell_centers
    grid_across = across_min + across_cell_centers
    gg_along, gg_across = np.meshgrid(grid_along, grid_across, indexing="ij")
    query_points = np.column_stack([gg_along.ravel() / along_radius, gg_across.ravel() / across_radius])

    # TODO(lib): the same per-cell query_ball_point loop grids.resample
    # replaced with cKDTree.sparse_distance_matrix + bincount
    # (grids.resample._aggregate_cells, MEAN); needs a parity test like
    # tests/unit/grids/test_resample_parity.py (allclose at float64
    # rounding: summation order differs).
    grid_values = np.full(query_points.shape[0], np.nan)
    for i, pt in enumerate(query_points):
        idx = tree.query_ball_point(pt, r=1.0)
        if idx:
            grid_values[i] = np.mean(values_valid[idx])

    grid = grid_values.reshape(along_size_px, across_size_px)
    meta = {
        "center_row": (center_along_m - along_min) / along_resolution_m,
        "center_col": (center_across_m - across_min) / across_resolution_m,
        "along_min": along_min,
        "across_min": across_min,
        "resolution_m": along_resolution_m,  # legacy key -- see docstring
        "along_resolution_m": along_resolution_m,
        "across_resolution_m": across_resolution_m,
    }
    return grid, meta


def extract_geometric_window_binned(
    channel_ds: xr.Dataset,
    size_px: int,
    *,
    resolution_m: float,
    center_along_m: float,
    center_across_m: float,
    channel_idx: int | None = 0,
    backscatter_var: str = "backscatter_intensity",
    avoid_edge_gaps: bool = False,
    along_resolution_m: float | None = None,
    across_resolution_m: float | None = None,
    along_size_px: int | None = None,
    across_size_px: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Same contract as :func:`extract_geometric_window` (same params,
    same ``(grid, meta)`` shape, same ``meta`` keys, drop-in
    interchangeable with :func:`locate_point`) but resamples via
    ``scipy.stats.binned_statistic_2d`` instead of a hand-written
    ``cKDTree.query_ball_point`` loop over every output cell.

    2026-09-29, real user request, following this file's own
    established-libraries-first convention (see ``sidescanpy/CLAUDE.md``'s
    "Dependencies" section): once per-axis pixel resolution existed
    (``along_resolution_m``/``across_resolution_m`` above), each output
    cell is really just "the mean of every real sample whose
    (along, across) position falls inside this cell's rectangle" — an
    ordinary 2D histogram-style binning, not a radius search. scipy
    already ships exactly that as a single vectorized call, so a custom
    per-cell loop wasn't buying anything.

    **Not a superset of the KDTree version — a genuinely different
    resampling philosophy, not just a faster implementation of the
    same one:**

    - No ``search_radius_m``/``along_search_radius_m``/
      ``across_search_radius_m`` — a point either falls inside a cell's
      exact rectangle or it doesn't, there is no overlap/gap-bridging
      radius to tune. A cell with zero real samples inside its own
      rectangle is NaN, even if a real sample sits just across the
      boundary in the neighboring cell. This means a coarse
      ``along_resolution_m`` chosen to bridge real ping-to-ping gaps
      (the KDTree version's ``along_search_radius_m`` role) still works
      here — a wide-enough bin naturally contains the next ping's
      point — but an ``along_resolution_m`` finer than the real
      ping-to-ping spacing will show real empty cells between pings
      instead of the KDTree version's radius-bridged fill.
    - Strictly faster on real data (one vectorized scipy call instead of
      one Python-level ``query_ball_point`` call per output pixel), but
      not chosen for speed — chosen because "one real neighborhood per
      pixel, no radius tuning" is a more direct match for what
      ``along_resolution_m``/``across_resolution_m`` were already
      designed to express once they exist per-axis. See
      ``export/geometric_window.py``'s module tests
      (``TestExtractGeometricWindowBinned``) for a real-data comparison
      against the KDTree version at matching resolution.

    Raises
    ------
    ValueError
        If no finite ``across_track_m``/backscatter samples exist in
        ``channel_ds`` at all (e.g. entirely water column).
    """
    along = along_track_distance_m(channel_ds, fill_nav_stalls=True)  # (ping,)
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

    along_resolution_m = along_resolution_m if along_resolution_m is not None else resolution_m
    across_resolution_m = across_resolution_m if across_resolution_m is not None else resolution_m
    along_size_px = along_size_px if along_size_px is not None else size_px
    across_size_px = across_size_px if across_size_px is not None else size_px

    along_half_extent = along_size_px * along_resolution_m / 2.0
    across_half_extent = across_size_px * across_resolution_m / 2.0
    if avoid_edge_gaps:
        center_along_m = _clamp_center(center_along_m, along_half_extent, along_valid.min(), along_valid.max())
        center_across_m = _clamp_center(center_across_m, across_half_extent, across_valid.min(), across_valid.max())
    along_min = center_along_m - along_half_extent
    across_min = center_across_m - across_half_extent
    along_max = along_min + along_size_px * along_resolution_m
    across_max = across_min + across_size_px * across_resolution_m

    along_edges = np.linspace(along_min, along_max, along_size_px + 1)
    across_edges = np.linspace(across_min, across_max, across_size_px + 1)

    # binned_statistic_2d returns NaN for an empty bin under "mean" --
    # matches extract_geometric_window's own no-sample-in-radius NaN
    # convention with no extra masking needed. statistic[i, j] pairs
    # with (along_edges[i:i+2], across_edges[j:j+2]) -- row=along,
    # col=across, same axis convention as the KDTree version's
    # meshgrid(..., indexing="ij").
    statistic, _, _, _ = binned_statistic_2d(
        along_valid,
        across_valid,
        values_valid,
        statistic="mean",
        bins=[along_edges, across_edges],
    )

    meta = {
        "center_row": (center_along_m - along_min) / along_resolution_m,
        "center_col": (center_across_m - across_min) / across_resolution_m,
        "along_min": along_min,
        "across_min": across_min,
        "resolution_m": along_resolution_m,  # legacy key -- see extract_geometric_window's docstring
        "along_resolution_m": along_resolution_m,
        "across_resolution_m": across_resolution_m,
    }
    return statistic, meta


def extract_geometric_window_griddata(
    channel_ds: xr.Dataset,
    size_px: int,
    *,
    resolution_m: float,
    center_along_m: float,
    center_across_m: float,
    channel_idx: int | None = 0,
    backscatter_var: str = "backscatter_intensity",
    avoid_edge_gaps: bool = False,
    along_resolution_m: float | None = None,
    across_resolution_m: float | None = None,
    along_size_px: int | None = None,
    across_size_px: int | None = None,
    interp_method: str = "linear",
    max_fill_distance_m: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Same contract as :func:`extract_geometric_window`/
    :func:`extract_geometric_window_binned` (same params, same
    ``(grid, meta)`` shape, same ``meta`` keys, drop-in interchangeable
    with :func:`locate_point`) but resamples via
    ``scipy.interpolate.griddata`` -- full scattered-data interpolation
    across the whole requested grid, not a radius search (KDTree) or
    strict rectangle membership (binned).

    2026-09-29, real user request/direction: an established-library
    resampling option with **no NaN gaps by construction** -- unlike
    :func:`extract_geometric_window_binned`, every output cell gets a
    value regardless of how sparse the real along-track sampling is
    relative to ``resolution_m``, because ``griddata`` interpolates
    (and, for the small edge fraction outside the real data's convex
    hull, nearest-neighbor-extrapolates -- see below) rather than
    reporting "no real sample here."

    **This is a genuinely different tradeoff than the other two
    functions, not a strict improvement — say so plainly:** at a
    resolution much finer than the real along-track sampling, the large
    majority of a griddata window's pixels are constructed by
    interpolation, not measured by the sensor. Real-data check (two
    real targets, ``resolution_m=0.03m``, ~60-66% of cells with no real
    sample at that resolution): griddata never shows a gap, but most of
    what fills those gaps is triangulated from real points several real
    ping-widths away, not a resolved local measurement. Compare
    :func:`extract_geometric_window` (zero fabrication, real radius
    averaging) and :func:`extract_geometric_window_binned` (zero
    fabrication, real gaps shown honestly).

    ``interp_method`` : ``"linear"`` (default), ``"cubic"``, or
    ``"nearest"`` -- passed straight through to ``scipy.interpolate.
    griddata``'s own ``method``. ``"linear"``/``"cubic"`` leave NaN
    outside the real data's convex hull (no point to triangulate
    against there); those cells are automatically filled with a second,
    ``method="nearest"`` pass so this function never returns a gap
    regardless of ``interp_method`` -- ``"nearest"`` itself is already
    gap-free after the first pass.
    ``max_fill_distance_m`` : cells farther than this from every real
    sample are NaN -- outside the recorded swath (past max range, the
    nadir gap, the excluded channel of a single-channel window), where
    either pass above would otherwise extend the nearest edge sample
    into smeared wedges/streaks of invented seafloor (seen on
    chips wider than the swath).
    ``None`` (default) derives it from the data: 1.5x the larger of
    the 95th-percentile along-track ping gap, the median across-track
    sample step, and the output resolution -- so gaps *between* real
    pings/samples still interpolate. ``np.inf`` restores the old
    never-NaN behavior.

    Raises
    ------
    ValueError
        If no finite ``across_track_m``/backscatter samples exist in
        ``channel_ds`` at all (e.g. entirely water column).
    """
    along = along_track_distance_m(channel_ds, fill_nav_stalls=True)  # (ping,)
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

    along_resolution_m = along_resolution_m if along_resolution_m is not None else resolution_m
    across_resolution_m = across_resolution_m if across_resolution_m is not None else resolution_m
    along_size_px = along_size_px if along_size_px is not None else size_px
    across_size_px = across_size_px if across_size_px is not None else size_px

    along_half_extent = along_size_px * along_resolution_m / 2.0
    across_half_extent = across_size_px * across_resolution_m / 2.0
    if avoid_edge_gaps:
        center_along_m = _clamp_center(center_along_m, along_half_extent, along_valid.min(), along_valid.max())
        center_across_m = _clamp_center(center_across_m, across_half_extent, across_valid.min(), across_valid.max())
    along_min = center_along_m - along_half_extent
    across_min = center_across_m - across_half_extent

    along_cell_centers = along_min + (np.arange(along_size_px) + 0.5) * along_resolution_m
    across_cell_centers = across_min + (np.arange(across_size_px) + 0.5) * across_resolution_m
    gg_along, gg_across = np.meshgrid(along_cell_centers, across_cell_centers, indexing="ij")
    query_points = np.column_stack([gg_along.ravel(), gg_across.ravel()])

    if max_fill_distance_m is None:
        along_gaps = np.abs(np.diff(along))
        along_gaps = along_gaps[along_gaps > 0]
        across_steps = np.abs(np.diff(across, axis=-1))
        across_steps = across_steps[np.isfinite(across_steps) & (across_steps > 0)]
        max_fill_distance_m = 1.5 * max(
            float(np.percentile(along_gaps, 95)) if along_gaps.size else 0.0,
            float(np.median(across_steps)) if across_steps.size else 0.0,
            along_resolution_m, across_resolution_m,
        )

    points = np.column_stack([along_valid, across_valid])
    if np.isfinite(max_fill_distance_m):
        # Triangulate only the points near this window, not the whole
        # loaded neighborhood (the full swath across-track, plus along-
        # track slack) -- Delaunay cost scales with point count, and a
        # tight chip can be a few percent of what's loaded (real case:
        # ~11s per dual-swath window without this).
        # A margin of 2x max_fill_distance_m keeps every point that can
        # affect a cell: cells with no point within max_fill_distance_m
        # are NaN below anyway, and interpolation triangles near the
        # window edge still get their neighbors.
        margin = 2.0 * max_fill_distance_m
        near = (
            (along_valid >= along_min - margin) & (along_valid <= along_min + along_size_px * along_resolution_m + margin)
            & (across_valid >= across_min - margin) & (across_valid <= across_min + across_size_px * across_resolution_m + margin)
        )
        points, values_valid = points[near], values_valid[near]
    if len(points) == 0:
        grid = np.full((along_size_px, across_size_px), np.nan)
        return grid, _griddata_meta(center_along_m, center_across_m, along_min, across_min, along_resolution_m, across_resolution_m)
    # TODO(lib): scipy.interpolate.LinearNDInterpolator/NearestNDInterpolator
    # built once per window would avoid griddata re-triangulating for the
    # NaN-fill pass below; parity test: identical grids (same Qhull
    # triangulation of the same points).
    if len(points) < 3:
        grid_values = griddata(points, values_valid, query_points, method="nearest")
    else:
        grid_values = griddata(points, values_valid, query_points, method=interp_method)
    nan_mask = np.isnan(grid_values)
    if nan_mask.any():
        # linear/cubic leave NaN outside the real data's convex hull --
        # fill with nearest-neighbor extrapolation so this function is
        # gap-free regardless of interp_method (its whole point, per the
        # real user request this exists to satisfy).
        grid_values[nan_mask] = griddata(points, values_valid, query_points[nan_mask], method="nearest")
    if np.isfinite(max_fill_distance_m):
        dist, _ = cKDTree(points).query(query_points, distance_upper_bound=max_fill_distance_m)
        grid_values[~np.isfinite(dist)] = np.nan

    grid = grid_values.reshape(along_size_px, across_size_px)
    return grid, _griddata_meta(center_along_m, center_across_m, along_min, across_min, along_resolution_m, across_resolution_m)


def _griddata_meta(center_along_m, center_across_m, along_min, across_min, along_resolution_m, across_resolution_m) -> dict:
    return {
        "center_row": (center_along_m - along_min) / along_resolution_m,
        "center_col": (center_across_m - across_min) / across_resolution_m,
        "along_min": along_min,
        "across_min": across_min,
        "resolution_m": along_resolution_m,  # legacy key -- see extract_geometric_window's docstring
        "along_resolution_m": along_resolution_m,
        "across_resolution_m": across_resolution_m,
    }


def locate_point(along_m: float, across_m: float, meta: dict) -> tuple[float, float]:
    """Translate a real-world ``(along_m, across_m)`` point into pixel
    ``(row, col)`` coordinates within a grid produced by
    :func:`extract_geometric_window`, using that call's own returned
    ``meta`` — for marking a *different* point than the one the window
    was centered on (e.g. a second target sharing the same frame).

    Reads ``meta``'s per-axis ``along_resolution_m``/
    ``across_resolution_m`` (not the legacy ``resolution_m`` key) —
    correct for both the isotropic/square case (where they're equal
    anyway) and a real anisotropic-pixel window.
    """
    return (
        (along_m - meta["along_min"]) / meta["along_resolution_m"],
        (across_m - meta["across_min"]) / meta["across_resolution_m"],
    )
