"""
Per-sample georeferencing: compute latitude_per_sample and
longitude_per_sample from per-ping nav, heading, and the platform-frame
offsets (across_track_m, along_track_m) attached by
compute_geometric_ranges.

Uses pyproj.Geod for geodetic forward calculations on a specified
ellipsoid (WGS84 by default). Vectorized over the entire (ping, channel,
sample) array via a single Geod.fwd call.

Bearing convention: compass heading (north = 0, increasing clockwise).
Per-sample bearing is heading + atan2(across_track, along_track).

NaN propagation:
    - Any NaN per-ping latitude, longitude, or heading → all samples
      for that ping are NaN.
    - Any NaN per-sample across_track or along_track → that sample is NaN.

This function does not modify processing_level. processing_history is
appended.

2026 rshomberg@uri.edu
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pyproj
import xarray as xr

from sidescanpy import __version__

logger = logging.getLogger(__name__)


_DEFAULT_ELLIPSOID = "WGS84"


def compute_georeferencing(
    ds: xr.Dataset,
    *,
    ellipsoid: str = _DEFAULT_ELLIPSOID,
) -> xr.Dataset:
    """
    Compute per-sample latitude and longitude.

    Parameters
    ----------
    ds : xr.Dataset
        Validated sidescan dataset with `across_track_m` and
        `along_track_m` already attached (from `compute_geometric_ranges`).
    ellipsoid : str, default "WGS84"
        Ellipsoid name passed to `pyproj.Geod`. Other common values:
        `"GRS80"`, `"clrk66"`, `"intl"`. Most users want WGS84.

    Returns
    -------
    xr.Dataset
        New dataset (input not modified) with two added per-sample data
        variables:

        - `latitude_per_sample` (float64, degrees_north)
        - `longitude_per_sample` (float64, degrees_east)

        `processing_level` is unchanged. `processing_history` is appended.

    Raises
    ------
    ValueError
        If `across_track_m` or `along_track_m` are not in the dataset
        (run `compute_geometric_ranges` first).
    """
    ds.sidescan.validate()

    required = {"across_track_m", "along_track_m"}
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError(
            f"compute_georeferencing requires {sorted(missing)} in the "
            f"dataset. Run compute_geometric_ranges first."
        )

    n_pings = ds.sizes["ping"]
    n_channels = ds.sizes["channel"]
    n_samples = ds.sizes["sample"]
    shape = (n_pings, n_channels, n_samples)

    # ----- per-ping inputs -----
    ping_lat = ds["latitude"].values.astype(np.float64)
    ping_lon = ds["longitude"].values.astype(np.float64)
    heading_deg = ds["heading_deg"].values.astype(np.float64)

    # ----- per-sample inputs -----
    across_track = ds["across_track_m"].values.astype(np.float64)
    along_track = ds["along_track_m"].values.astype(np.float64)

    # Distance from platform: sqrt(across² + along²)
    distance_m = np.sqrt(across_track ** 2 + along_track ** 2)

    # Bearing offset from heading: atan2(across, along) in degrees
    # In compass convention: along=positive forward, across=positive starboard
    # atan2(across, along) returns 0 when across=0 and along>0 (forward)
    # and +90 when along=0 and across>0 (starboard) — exactly compass offset
    bearing_offset_deg = np.degrees(np.arctan2(across_track, along_track))

    # Per-sample compass bearing
    heading_b = np.broadcast_to(heading_deg[:, None, None], shape)
    bearing_deg = heading_b + bearing_offset_deg

    # Per-sample starting position (broadcast per-ping nav)
    lat_b = np.broadcast_to(ping_lat[:, None, None], shape)
    lon_b = np.broadcast_to(ping_lon[:, None, None], shape)

    # Flatten for pyproj.Geod.fwd vectorized call
    geod = pyproj.Geod(ellps=ellipsoid)
    flat_lon = lon_b.ravel()
    flat_lat = lat_b.ravel()
    flat_bearing = bearing_deg.ravel()
    flat_distance = distance_m.ravel()

    # pyproj.Geod.fwd handles NaN by returning NaN
    out_lon, out_lat, _ = geod.fwd(
        flat_lon, flat_lat, flat_bearing, flat_distance,
    )

    lat_per_sample = out_lat.reshape(shape)
    lon_per_sample = out_lon.reshape(shape)

    # ----- assemble output -----
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)

    out = out.assign(
        latitude_per_sample=(("ping", "channel", "sample"), lat_per_sample),
        longitude_per_sample=(("ping", "channel", "sample"), lon_per_sample),
    )

    out["latitude_per_sample"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude of sample reflection point",
        "description": (
            f"Latitude of each sample's reflection point on the seafloor, "
            f"computed via geodetic forward calculation on the {ellipsoid} "
            f"ellipsoid."
        ),
    }
    out["longitude_per_sample"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude of sample reflection point",
        "description": (
            f"Longitude of each sample's reflection point on the seafloor, "
            f"computed via geodetic forward calculation on the {ellipsoid} "
            f"ellipsoid."
        ),
    }

    # ----- update processing history -----
    history_str = out.attrs.get("processing_history", "[]")
    try:
        history = json.loads(history_str)
        if not isinstance(history, list):
            history = []
    except (TypeError, ValueError):
        history = []

    history.append(
        {
            "step": "compute_georeferencing",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {
                "ellipsoid": ellipsoid,
            },
            "function": "sidescanpy.processing.georeferencing.compute_georeferencing",
            "version": __version__,
        }
    )
    out.attrs["processing_history"] = json.dumps(history)

    logger.info(
        "compute_georeferencing: added latitude_per_sample and "
        "longitude_per_sample for %d pings × %d channels × %d samples "
        "(ellipsoid=%s)",
        n_pings, n_channels, n_samples, ellipsoid,
    )

    return out