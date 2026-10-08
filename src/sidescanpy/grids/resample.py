"""
Resampling channel data to georeferenced grids.

Provides resample_channel_to_grid() to aggregate sidescan backscatter
from survey geometry to a regular grid using various aggregation methods,
and grid_channel() to do so on a UTM grid fit to the channel's own extent.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from sidescanpy.core.history import STEP_TVG, has_step
from sidescanpy.core.history import processing_level as derived_processing_level
from sidescanpy.grids.definition import (
    GridDefinition,
    bbox_from_channel_dataset,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.grids.raster import SideScanGriddedRaster

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)


def _validate_channel_data(channel: xr.Dataset, backscatter_var: str) -> None:
    """
    Validate that channel has the required 3D (ping, channel, sample)
    variables with matching shapes.

    Raises
    ------
    ValueError
        If required variables are missing, backscatter_var is not 3D, or
        latitude_per_sample / longitude_per_sample don't match its shape.
    """
    required_vars = {backscatter_var, "latitude_per_sample", "longitude_per_sample"}
    missing = required_vars - set(channel.data_vars)
    if missing:
        raise ValueError(
            f"channel missing required variables: {missing}. "
            f"Has: {set(channel.data_vars)}"
        )

    backscatter_shape = channel[backscatter_var].shape
    if len(backscatter_shape) != 3:
        raise ValueError(
            f"{backscatter_var} must be 3D (ping, channel, sample), got shape "
            f"{backscatter_shape} (ndim={len(backscatter_shape)}). Pass the "
            f"full channel dataset, not a pre-flattened 2D array."
        )

    lat_shape = channel["latitude_per_sample"].shape
    lon_shape = channel["longitude_per_sample"].shape
    if lat_shape != backscatter_shape or lon_shape != backscatter_shape:
        raise ValueError(
            f"{backscatter_var}, latitude_per_sample, and longitude_per_sample "
            f"must all share the same shape. Got {backscatter_var}="
            f"{backscatter_shape}, latitude_per_sample={lat_shape}, "
            f"longitude_per_sample={lon_shape}."
        )


class ResamplingMethod(str, Enum):  # noqa: UP042  (StrEnum changes str()/format() behavior)
    """Resampling aggregation method."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    NEAREST = "nearest"
    IDW = "idw"


# Upper bound on (cell, sample) pairs materialized per block; ~24 bytes per
# pair in sparse_distance_matrix's output plus a few same-length work
# arrays (~100 bytes/pair in all), times _MAX_WORKERS blocks in flight:
# roughly 0.8 GB peak at the defaults, independent of survey size.
_MAX_PAIRS_PER_BLOCK = 1_000_000
_MAX_WORKERS = 8


def _cell_blocks(pair_counts: np.ndarray, max_pairs: int):
    """Contiguous ``(start, stop)`` flat-cell ranges, each holding at most
    ``max_pairs`` (cell, sample) pairs (or a single cell, if one cell alone
    exceeds it). Trailing cells with no pairs are skipped."""
    cum = np.cumsum(pair_counts)
    total = int(cum[-1]) if cum.size else 0
    start = 0
    n = len(pair_counts)
    while start < n:
        base = int(cum[start - 1]) if start else 0
        if base >= total:
            break
        stop = int(np.searchsorted(cum, base + max_pairs, side="right"))
        stop = max(stop, start + 1)
        yield start, min(stop, n)
        start = stop


def _aggregate_cells(
    tree: cKDTree,
    sample_ys: np.ndarray,
    sample_xs: np.ndarray,
    sample_values: np.ndarray,
    grid_ys: np.ndarray,
    grid_xs: np.ndarray,
    search_radius: float,
    method: "ResamplingMethod",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate samples within ``search_radius`` of every cell center.

    Vectorized equivalent of one ``tree.query_ball_point(center, r)`` per
    cell (same inclusive ``distance <= r`` test): the (cell, sample) pairs
    come from ``cKDTree.sparse_distance_matrix`` between a tree of cell
    centers and the sample tree, in blocks of at most
    ``_MAX_PAIRS_PER_BLOCK`` pairs processed on a small thread pool
    (scipy's tree queries release the GIL), and each method is a grouped
    reduction over those pairs. Non-finite sample values are dropped
    before aggregating; a cell with no finite value stays NaN with a
    count of 0.

    MAX, MEDIAN and NEAREST reproduce a per-cell ``np.max`` /
    ``np.median`` / ``tree.query(k=1)`` exactly; MEAN and IDW sum in a
    different order than per-cell ``np.mean`` / ``np.sum``, so they agree
    to float64 rounding (~1e-15 relative), not always bit-for-bit.
    """
    n_ns, n_ew = grid_ys.shape
    centers = np.column_stack([grid_ys.ravel(), grid_xs.ravel()])
    n_cells = len(centers)
    n_samples = len(sample_values)

    count = np.zeros(n_cells, dtype=np.int64)
    grid = np.full(n_cells, np.nan)

    if method == ResamplingMethod.MEDIAN:
        # Global rank of every sample value: sorting pairs by
        # (cell, rank) == sorting each cell's values, with one integer key.
        rank = np.empty(n_samples, dtype=np.int64)
        rank[np.argsort(sample_values, kind="stable")] = np.arange(n_samples)

    def _block(start: int, stop: int) -> None:
        pairs = cKDTree(centers[start:stop]).sparse_distance_matrix(tree, search_radius, output_type="ndarray")
        cell = pairs["i"].astype(np.intp)
        smp = pairs["j"].astype(np.intp)
        del pairs
        vals = sample_values[smp]
        finite = np.isfinite(vals)
        cell, smp, vals = cell[finite], smp[finite], vals[finite]
        if cell.size == 0:
            return
        n_block = stop - start
        block_count = np.bincount(cell, minlength=n_block)
        count[start:stop] = block_count
        has = block_count > 0
        out = np.full(n_block, np.nan)

        if method == ResamplingMethod.MEAN:
            sums = np.bincount(cell, weights=vals, minlength=n_block)
            out[has] = sums[has] / block_count[has]
        elif method == ResamplingMethod.MAX:
            mx = np.full(n_block, -np.inf)
            np.maximum.at(mx, cell, vals)
            out[has] = mx[has]
        elif method == ResamplingMethod.MEDIAN:
            sorted_vals = vals[np.argsort(cell.astype(np.int64) * n_samples + rank[smp])]
            group_start = np.concatenate(([0], np.cumsum(block_count)[:-1]))
            gs, gc = group_start[has], block_count[has]
            lo = sorted_vals[gs + (gc - 1) // 2]
            hi = sorted_vals[gs + gc // 2]
            # np.median: middle element (odd) or mean of the two middle (even)
            out[has] = np.where(gc % 2 == 1, lo, (lo + hi) / 2.0)
        elif method == ResamplingMethod.IDW:
            dy = sample_ys[smp] - centers[start + cell, 0]
            dx = sample_xs[smp] - centers[start + cell, 1]
            distances = np.sqrt(dy**2 + dx**2)
            distances[distances == 0] = 1e-9
            weights = 1.0 / distances
            weights /= np.bincount(cell, weights=weights, minlength=n_block)[cell]
            out[has] = np.bincount(cell, weights=vals * weights, minlength=n_block)[has]
        elif method == ResamplingMethod.NEAREST:
            return  # only the counts are needed here; values come from tree.query below
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        grid[start:stop] = out

    # Pair count per cell (includes non-finite values); only used to size
    # the blocks, so memory stays bounded at any sample density.
    pair_counts = np.asarray(tree.query_ball_point(centers, r=search_radius, return_length=True, workers=-1))
    blocks = list(_cell_blocks(pair_counts, _MAX_PAIRS_PER_BLOCK))
    n_workers = min(_MAX_WORKERS, os.cpu_count() or 1, len(blocks))
    if n_workers <= 1:
        for start, stop in blocks:
            _block(start, stop)
    else:
        # Blocks write disjoint slices of count/grid, so no locking needed.
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for future in [pool.submit(_block, a, b) for a, b in blocks]:
                future.result()

    if method == ResamplingMethod.NEAREST:
        # Nearest sample over *all* positioned samples (not just finite
        # values), so a cell whose single nearest sample is NaN stays NaN
        # even though it still counts its in-radius finite contributors.
        filled_idx = np.flatnonzero(count > 0)
        if filled_idx.size:
            _, nearest = tree.query(centers[filled_idx], k=1, workers=-1)
            grid[filled_idx] = sample_values[nearest]
            grid[~np.isfinite(grid)] = np.nan

    return grid.reshape(n_ns, n_ew), count.reshape(n_ns, n_ew)


def resample_channel_to_grid(
    channel: xr.Dataset,
    grid_def: GridDefinition,
    method: ResamplingMethod = ResamplingMethod.MEAN,
    separate_sides: bool = False,
    max_search_radius_m: Optional[float] = None,
    backscatter_var: str = "backscatter_intensity",
) -> SideScanGriddedRaster:
    """
    Resample channel backscatter to a regular grid.

    Parameters
    ----------
    channel : xr.Dataset
        Channel dataset with a backscatter variable (see backscatter_var)
        and per-sample coordinates (latitude_per_sample, longitude_per_sample).
    grid_def : GridDefinition
        Target grid definition (bbox, resolution, CRS).
    method : ResamplingMethod, default "mean"
        Aggregation method: mean, median, max, nearest, or idw.
    separate_sides : bool, default False
        If True, output separate port and starboard bands (diagnostic mode).
        If False, merge port and starboard into single backscatter variable.
    max_search_radius_m : float, optional
        Maximum search radius in meters. If None, use 1.5x grid resolution.
    backscatter_var : str, default "backscatter_intensity"
        Name of the backscatter variable in channel to resample. Corrections
        are applied in place to "backscatter_intensity" (dB once TVG has
        run, linear amplitude before); for the uncorrected signal of a
        corrected channel, assign
        ``sidescanpy.processing.uncorrected_backscatter(channel)`` to a
        variable and grid that.

    Notes
    -----
    **This function does not filter out or otherwise special-case the
    TVG floor-clipped sample population** (see `apply_tvg()`'s
    "KNOWN ISSUE" docstring note). Those samples generally have finite
    georeferenced coordinates, so they pass the `np.isfinite(lat)/
    np.isfinite(lon)` filter here like any other sample and get
    included in each cell's aggregation. With `method=MEAN` (the
    default) and typical real-survey point densities (thousands of raw
    samples per cell at 1m resolution), that ~1% outlier population
    gets *diluted*, not removed — it's swamped by the majority-real
    samples in the same cell, not excluded from the average. This
    dilution is exactly why a gridded raster's effective dynamic range
    (and therefore its percentile-based color scale) can differ sharply
    from a per-sample waterfall's.

    Returns
    -------
    SideScanGriddedRaster
        Gridded backscatter raster with n_contributing_samples tracking.

    Raises
    ------
    ValueError
        If channel is missing required variables or grid bounds don't overlap.
    """
    # Validate input
    _validate_channel_data(channel, backscatter_var)

    # Channel label from the converters' own attribute (``freq_hz_start``
    # labels the channel even for a chirp -- see core/schema.py and the
    # io converters; it is also what ``freq_key`` is built from).
    frequency_hz = channel.attrs.get("freq_hz_start", 0)
    processing_level = derived_processing_level(channel)

    # Convert frequency to friendly name
    if frequency_hz > 0:
        freq_khz = frequency_hz / 1000
        channel_name = f"{freq_khz:.0f} kHz"
    else:
        channel_name = "unknown"

    # Extract data
    backscatter_data = channel[backscatter_var].values.astype(float)  # (ping, channel, sample)
    lat_per_sample = channel["latitude_per_sample"].values.astype(float)
    lon_per_sample = channel["longitude_per_sample"].values.astype(float)

    # Flatten to 1D for KDTree
    backscatter_1d = backscatter_data.ravel()
    lat_1d = lat_per_sample.ravel()
    lon_1d = lon_per_sample.ravel()

    # Filter out NaN coordinates
    valid_mask = np.isfinite(lat_1d) & np.isfinite(lon_1d)
    if not np.any(valid_mask):
        raise ValueError("No valid sample coordinates in channel")

    backscatter_valid = backscatter_1d[valid_mask]
    lat_valid = lat_1d[valid_mask]
    lon_valid = lon_1d[valid_mask]

    logger.debug(
        f"Resampling {np.sum(valid_mask):,} / {len(backscatter_1d):,} "
        f"valid samples using {method.value} aggregation"
    )

    # Sample coordinates are always WGS84 lat/lon (from compute_georeferencing).
    # Reproject into the grid's own CRS before building the KDTree, so search
    # radius and distances are computed in the grid's native units — degrees
    # for a geographic grid, meters for a projected one (e.g. UTM) — rather
    # than always assuming WGS84 degrees regardless of grid_def.crs.
    is_geographic = grid_def.crs.is_geographic
    if is_geographic:
        y_valid, x_valid = lat_valid, lon_valid
    else:
        transformer = Transformer.from_crs(_WGS84, grid_def.crs, always_xy=True)
        x_valid, y_valid = transformer.transform(lon_valid, lat_valid)

        # A finite WGS84 lat/lon can still project to a non-finite
        # projected coordinate -- e.g. within a couple degrees of a pole,
        # where UTM's own math blows up. Filtered here rather than
        # relying on the pre-transform isfinite check above, which can't
        # see this: cKDTree raises an opaque scipy ValueError on any
        # non-finite input, not a message pointing at the real cause.
        projected_finite = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.all(projected_finite):
            n_dropped = int(np.sum(~projected_finite))
            logger.warning(
                f"{n_dropped:,} sample(s) had a finite WGS84 position but a non-finite "
                f"{grid_def.crs} projection (e.g. too close to a pole) -- excluded from gridding"
            )
            backscatter_valid = backscatter_valid[projected_finite]
            x_valid = x_valid[projected_finite]
            y_valid = y_valid[projected_finite]

    if len(x_valid) == 0:
        raise ValueError("No valid sample coordinates in channel")

    # Build KDTree in the grid's native coordinate space
    sample_coords = np.column_stack([y_valid, x_valid])
    tree = cKDTree(sample_coords)

    # Get grid cell centers — same native space as above (lat/lon for a
    # geographic grid, northing/easting for a projected one; see
    # GridDefinition.get_grid_centers)
    grid_ys, grid_xs = grid_def.get_grid_centers()

    # Search radius in the grid's native units: degrees (~111 km per degree
    # at the equator) for a geographic grid, meters directly for a projected
    # one — resolution_m is already in meters, so no unit conversion needed
    # there.
    # Accuracy note: on a geographic grid the radius is isotropic in
    # *degrees*, so its east-west ground reach is only cos(lat) of the
    # north-south one (0.75x at 41 deg, 0.5x at 60 deg). Past ~70.5 deg
    # (1.5 * cos(lat) < 0.5 cells) cells can miss samples between them.
    # Projected (UTM) grids -- what the CLI uses -- are unaffected.
    # TODO(lib): grid in a local projected CRS (pyproj) and drop the
    # degree radius; needs a parity test against the UTM path.
    if is_geographic:
        if max_search_radius_m is None:
            search_radius = 1.5 * grid_def.resolution_m / 111000.0
        else:
            search_radius = max_search_radius_m / 111000.0
    else:
        search_radius = (
            1.5 * grid_def.resolution_m
            if max_search_radius_m is None
            else max_search_radius_m
        )

    backscatter_grid, n_contributing = _aggregate_cells(
        tree, y_valid, x_valid, backscatter_valid, grid_ys, grid_xs, search_radius, method,
    )

    logger.debug(
        f"Resampling complete: {np.sum(np.isfinite(backscatter_grid)):,} / "
        f"{backscatter_grid.size:,} cells filled"
    )

    # Units: TVG converts backscatter_intensity to dB in place. Other
    # variable names default to linear.
    units = "dB" if backscatter_var == "backscatter_intensity" and has_step(channel, STEP_TVG) else "linear"

    # Create raster
    raster = SideScanGriddedRaster(
        grid_def=grid_def,
        backscatter=backscatter_grid,
        channel_name=channel_name,
        processing_level=processing_level,
        units=units,
    )

    # Add n_contributing_samples to dataset
    raster.dataset["n_contributing_samples"] = (
        ["ns", "ew"],
        n_contributing,
    )
    raster.dataset["n_contributing_samples"].attrs["units"] = "count"
    raster.dataset["n_contributing_samples"].attrs["long_name"] = (
        "Number of samples contributing to grid cell"
    )

    # Add resampling metadata
    raster.dataset.attrs["resampling_method"] = method.value
    raster.dataset.attrs["frequency_hz"] = frequency_hz
    raster.dataset.attrs["backscatter_variable"] = backscatter_var

    return raster


def grid_channel(
    channel: xr.Dataset,
    resolution_m: float = 1.0,
    method: ResamplingMethod | str = ResamplingMethod.MEAN,
    *,
    backscatter_var: str = "backscatter_intensity",
    max_search_radius_m: Optional[float] = None,
) -> SideScanGriddedRaster:
    """Grid a georeferenced channel onto a UTM grid fit to its own extent.

    The grid covers :func:`~sidescanpy.grids.definition.bbox_from_channel_dataset`
    in the UTM zone of its center
    (:func:`~sidescanpy.grids.definition.utm_grid_from_wgs84_bbox`), and
    the channel is resampled onto it with :func:`resample_channel_to_grid`.

    Parameters
    ----------
    channel : xr.Dataset
        Channel with ``latitude_per_sample``/``longitude_per_sample``
        (georeferencing applied) and ``backscatter_var``.
    resolution_m : float, default 1.0
        Cell size in meters.
    method : ResamplingMethod or str, default "mean"
        Aggregation method (``"mean"``, ``"median"``, ``"max"``,
        ``"nearest"``, ``"idw"``).
    backscatter_var : str, default "backscatter_intensity"
        Variable to grid.
    max_search_radius_m : float, optional
        Passed to :func:`resample_channel_to_grid`.

    Returns
    -------
    SideScanGriddedRaster
    """
    grid_def = utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(channel), resolution_m=resolution_m)
    return resample_channel_to_grid(
        channel, grid_def, method=ResamplingMethod(method),
        max_search_radius_m=max_search_radius_m, backscatter_var=backscatter_var,
    )
