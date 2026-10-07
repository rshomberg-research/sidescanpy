"""
Navigation-derived ping geometry: along-track distance and nav-stall repair.

- :func:`along_track_distance_m` -- cumulative along-track distance (m)
  from per-ping latitude/longitude, optionally with short navigation
  stalls filled.
- :func:`fill_nav_stalls` -- replace short runs of stalled position
  steps with the local median ping step.

Pure array/dataset functions; no plotting.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def along_track_distance_m(ds: xr.Dataset, *, fill_nav_stalls: bool = False) -> np.ndarray:
    """Cumulative along-track distance (meters) from per-ping lat/lon.

    Public wrapper around the same computation used internally for
    ``y_axis="along_track"`` — exposed for callers (e.g. distance-based
    segmentation) that need the coordinate array without going through
    ``plot_waterfall``.

    ``fill_nav_stalls``: navigation can stall -- several consecutive
    pings keep (nearly) the same position while the vessel keeps moving,
    and it resumes without catching up. The raw distance is then flat
    across those pings, and geometric resampling stacks their seafloor
    into a single along-track line (seen on one real survey: ~7 pings
    at each checked target, 42ms apart with distinct data, advanced only
    ~0.07m in total instead of ~0.5m -- a hard seam through the target
    at fine resolution). With ``True``, each stalled step is replaced by
    the local median step (see ``fill_nav_stalls``). Off by default so
    gap/stationarity detection still sees the raw navigation.
    """
    along = _along_track_distance(ds)
    return _fill_nav_stalls_impl(along) if fill_nav_stalls else along


def fill_nav_stalls(
    along: np.ndarray, *, stall_fraction: float = 0.5, max_stall_pings: int = 20, window_pings: int = 25,
) -> np.ndarray:
    """Replace short navigation stalls with the local median ping step.

    A step is stalled when it is below ``stall_fraction`` of the median
    step within ``window_pings`` either side. Only runs of at most
    ``max_stall_pings`` stalled steps with moving steps on both sides
    are filled -- a longer run, or one at the window edge, is treated as
    the vessel really stopping and left as is.
    """
    if along.size < 3:
        return along
    steps = np.diff(along)
    n = steps.size
    local = np.array([np.median(steps[max(0, k - window_pings):k + window_pings + 1]) for k in range(n)])
    stalled = steps < stall_fraction * local
    out = steps.copy()
    k = 0
    while k < n:
        if not stalled[k]:
            k += 1
            continue
        end = k
        while end < n and stalled[end]:
            end += 1
        if 0 < k and end < n and end - k <= max_stall_pings:
            moving = steps[max(0, k - window_pings):min(n, end + window_pings)]
            moving = moving[moving >= stall_fraction * np.median(moving)]
            out[k:end] = np.median(moving) if moving.size else steps[k:end]
        k = end
    return np.concatenate([[along[0]], along[0] + np.cumsum(out)])


def _along_track_distance(ds: xr.Dataset) -> np.ndarray:
    """Compute cumulative along-track distance from per-ping lat/lon."""
    import pyproj

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    geod = pyproj.Geod(ellps="WGS84")

    if len(lats) < 2:
        return np.array([0.0])

    # Forward azimuths and distances between consecutive pings
    _, _, segment_distances = geod.inv(
        lons[:-1], lats[:-1], lons[1:], lats[1:]
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_distances)])
    return cumulative.astype(np.float64)


# ``along_track_distance_m`` has a ``fill_nav_stalls`` keyword that shadows the
# function inside its body, so it calls through this alias.
_fill_nav_stalls_impl = fill_nav_stalls
