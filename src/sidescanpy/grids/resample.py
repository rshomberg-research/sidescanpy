"""
Resampling channel data to georeferenced grids.

Provides resample_channel_to_grid() to aggregate sidescan backscatter
from survey geometry to a regular grid using various aggregation methods.
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from sidescanpy.grids.definition import GridDefinition
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


class ResamplingMethod(str, Enum):
    """Resampling aggregation method."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    NEAREST = "nearest"
    IDW = "idw"


def resample_channel_to_grid(
    channel: xr.Dataset,
    grid_def: GridDefinition,
    method: ResamplingMethod = ResamplingMethod.MEAN,
    separate_sides: bool = False,
    max_search_radius_m: Optional[float] = None,
    backscatter_var: str = "backscatter_raw",
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
    backscatter_var : str, default "backscatter_raw"
        Name of the backscatter variable in channel to resample. Radiometric
        correction (processing/radiometric.py apply_tvg()) writes corrected
        values to "backscatter_intensity", leaving "backscatter_raw" as an
        untouched linear-amplitude reference — pass
        backscatter_var="backscatter_intensity" to grid the corrected data.

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

    # Extract frequency and metadata
    frequency_hz = channel.attrs.get("frequency_hz", 0)
    processing_level = channel.attrs.get("processing_level", "raw")
    
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

    logger.info(
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

    # Build KDTree in the grid's native coordinate space
    sample_coords = np.column_stack([y_valid, x_valid])
    tree = cKDTree(sample_coords)

    # Get grid cell centers — same native space as above (lat/lon for a
    # geographic grid, northing/easting for a projected one; see
    # GridDefinition.get_grid_centers)
    grid_ys, grid_xs = grid_def.get_grid_centers()
    num_cells_ns, num_cells_ew = grid_ys.shape

    # Search radius in the grid's native units: degrees (~111 km per degree
    # at the equator) for a geographic grid, meters directly for a projected
    # one — resolution_m is already in meters, so no unit conversion needed
    # there.
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

    # Initialize output arrays
    backscatter_grid = np.full((num_cells_ns, num_cells_ew), np.nan, dtype=float)
    n_contributing = np.zeros((num_cells_ns, num_cells_ew), dtype=int)

    # Resample each cell
    for i in range(num_cells_ns):
        for j in range(num_cells_ew):
            cell_y = grid_ys[i, j]
            cell_x = grid_xs[i, j]

            # Query tree for samples near this cell
            indices = tree.query_ball_point(
                [cell_y, cell_x],
                r=search_radius,
            )

            if not indices:
                continue

            # Get backscatter values for contributing samples
            contrib_backscatter = backscatter_valid[indices]
            
            # Filter out NaN
            valid_contrib = contrib_backscatter[np.isfinite(contrib_backscatter)]

            if len(valid_contrib) == 0:
                continue

            n_contributing[i, j] = len(valid_contrib)

            # Aggregate per method
            if method == ResamplingMethod.MEAN:
                backscatter_grid[i, j] = np.mean(valid_contrib)
            elif method == ResamplingMethod.MEDIAN:
                backscatter_grid[i, j] = np.median(valid_contrib)
            elif method == ResamplingMethod.MAX:
                backscatter_grid[i, j] = np.max(valid_contrib)
            elif method == ResamplingMethod.NEAREST:
                # Find nearest sample
                dists, nearest_idx = tree.query([cell_y, cell_x], k=1)
                nearest_backscatter = backscatter_valid[nearest_idx]
                if np.isfinite(nearest_backscatter):
                    backscatter_grid[i, j] = nearest_backscatter
            elif method == ResamplingMethod.IDW:
                # Inverse-distance weighting
                sample_ys = y_valid[indices]
                sample_xs = x_valid[indices]

                # Distances in the grid's native units (degrees or meters)
                dy = sample_ys - cell_y
                dx = sample_xs - cell_x
                distances = np.sqrt(dy**2 + dx**2)
                
                # Avoid division by zero
                distances[distances == 0] = 1e-9
                
                # Weight by 1/distance
                weights = 1.0 / distances
                weights /= np.sum(weights)
                
                backscatter_grid[i, j] = np.sum(valid_contrib * weights[:len(valid_contrib)])
            else:
                raise ValueError(f"Unknown resampling method: {method}")

    logger.info(
        f"Resampling complete: {np.sum(np.isfinite(backscatter_grid)):,} / "
        f"{backscatter_grid.size:,} cells filled"
    )

    # Units: apply_tvg() (processing/radiometric.py) writes TVG-corrected
    # values to backscatter_intensity in dB, leaving backscatter_raw as
    # untouched linear amplitude. Other variable names default to linear.
    units = "dB" if backscatter_var == "backscatter_intensity" else "linear"

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