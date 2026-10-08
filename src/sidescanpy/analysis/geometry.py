"""
Ping geometry measurements for one section of a survey.

:func:`measure_ping_geometry` measures median along-track ping spacing,
across-track swath width and range-axis sample interval — the native
resolutions to choose an export pixel size from.
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin
from sidescanpy.processing.navigation import along_track_distance_m

logger = logging.getLogger(__name__)


@dataclass
class PingGeometry(ResultMixin):
    """Native sampling of a section, in metres; ``None`` where not measurable."""

    ping_spacing_m_median: float | None = None
    swath_width_m: float | None = None
    sample_interval_m: float | None = None


def measure_ping_geometry(ds: xr.Dataset, *, low_pct: float = 1.0, high_pct: float = 99.0) -> PingGeometry:
    """Median ping spacing, swath width and sample interval of ``ds``.

    - ``ping_spacing_m_median``: median nonzero along-track step between
      consecutive pings (needs at least 2 pings).
    - ``swath_width_m``: ``high_pct`` minus ``low_pct`` percentile of
      ``across_track_m`` (robust to a few far-flung samples).
    - ``sample_interval_m``: median per-ping step of ``slant_range_m``
      on channel 0.
    """
    result = PingGeometry()
    n_pings = ds.sizes["ping"]
    if n_pings >= 2:
        dist = along_track_distance_m(ds)
        diffs = np.abs(np.diff(dist))
        nonzero = diffs[diffs > 0]
        if nonzero.size:
            result.ping_spacing_m_median = float(np.median(nonzero))
    if "across_track_m" in ds:
        across = np.asarray(ds["across_track_m"].values, dtype=np.float64)
        finite = across[np.isfinite(across)]
        if finite.size:
            lo, hi = np.percentile(finite, [low_pct, high_pct])
            result.swath_width_m = float(hi - lo)
    if "slant_range_m" in ds and ds.sizes.get("sample", 0) >= 2:
        slant = np.asarray(ds["slant_range_m"].isel(channel=0).values, dtype=np.float64)
        per_ping_interval = slant[:, 1] - slant[:, 0]
        finite = per_ping_interval[np.isfinite(per_ping_interval) & (per_ping_interval > 0)]
        if finite.size:
            result.sample_interval_m = float(np.median(finite))
    return result
