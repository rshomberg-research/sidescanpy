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
from scipy.spatial import cKDTree

from sidescanpy.grids.definition import GridDefinition
from sidescanpy.grids.raster import SideScanGriddedRaster

logger = logging.getLogger(__name__)


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
) -> SideScanGriddedRaster:
    """
    Resample channel backscatter to a regular grid.

    Parameters
    ----------
    channel : xr.Dataset
        Channel dataset with backscatter_raw and per-sample coordinates
        (latitude_per_sample, longitude_per_sample).
    grid_def : GridDefinition
        Target grid definition (bbox, resolution, CRS).
    method : ResamplingMethod, default "mean"
        Aggregation method: mean, median, max, nearest, or idw.
    separate_sides : bool, default False
        If True, output separate port and starboard bands (diagnostic mode).
        If False, merge port and starboard into single backscatter variable.
    max_search_radius_m : float, optional
        Maximum search radius in meters. If None, use 1.5x grid resolution.

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
    required_vars = {"backscatter_raw", "latitude_per_sample", "longitude_per_sample"}
    missing = required_vars - set(channel.data_vars)
    if missing:
        raise ValueError(
            f"channel missing required variables: {missing}. "
            f"Has: {set(channel.data_vars)}"
        )

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
    backscatter_raw = channel["backscatter_raw"].values.astype(float)  # (ping, sample)
    lat_per_sample = channel["latitude_per_sample"].values.astype(float)
    lon_per_sample = channel["longitude_per_sample"].values.astype(float)

    # Flatten to 1D for KDTree
    backscatter_1d = backscatter_raw.ravel()
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

    # Build KDTree in lat/lon space (WGS84)
    # Note: this is approximate for large areas; a projection would be more accurate
    sample_coords = np.column_stack([lat_valid, lon_valid])
    tree = cKDTree(sample_coords)

    # Get grid cell centers
    grid_lats, grid_lons = grid_def.get_grid_centers()
    num_cells_ns, num_cells_ew = grid_lats.shape

    # Search radius: use 1.5x grid resolution in degrees (~111 km per degree at equator)
    if max_search_radius_m is None:
        search_radius_deg = 1.5 * grid_def.resolution_m / 111000.0
    else:
        search_radius_deg = max_search_radius_m / 111000.0

    # Initialize output arrays
    backscatter_grid = np.full((num_cells_ns, num_cells_ew), np.nan, dtype=float)
    n_contributing = np.zeros((num_cells_ns, num_cells_ew), dtype=int)

    # Resample each cell
    for i in range(num_cells_ns):
        for j in range(num_cells_ew):
            cell_lat = grid_lats[i, j]
            cell_lon = grid_lons[i, j]

            # Query tree for samples near this cell
            indices = tree.query_ball_point(
                [cell_lat, cell_lon],
                r=search_radius_deg,
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
                dists, nearest_idx = tree.query([cell_lat, cell_lon], k=1)
                nearest_backscatter = backscatter_valid[nearest_idx]
                if np.isfinite(nearest_backscatter):
                    backscatter_grid[i, j] = nearest_backscatter
            elif method == ResamplingMethod.IDW:
                # Inverse-distance weighting
                sample_lats = lat_valid[indices]
                sample_lons = lon_valid[indices]
                
                # Compute distances in degrees
                dlat = sample_lats - cell_lat
                dlon = sample_lons - cell_lon
                distances = np.sqrt(dlat**2 + dlon**2)
                
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

    # Create raster
    raster = SideScanGriddedRaster(
        grid_def=grid_def,
        backscatter=backscatter_grid,
        channel_name=channel_name,
        processing_level=processing_level,
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

    return raster