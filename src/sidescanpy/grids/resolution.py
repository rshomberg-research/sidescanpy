"""
Recommended grid / chip resolution from a section's native sampling.
"""

from __future__ import annotations

import logging

import xarray as xr

from sidescanpy.analysis.geometry import PingGeometry, measure_ping_geometry

logger = logging.getLogger(__name__)


def recommend_resolution(ds: xr.Dataset | PingGeometry) -> float | None:
    """Half the finer of the median ping spacing and the sample interval, in metres.

    Oversamples past the finest native axis (every real sample on that
    axis lands in its own cell) and leaves the coarser axis to be
    interpolated. When the sample interval is unavailable (no
    ``slant_range_m``) this is half the ping spacing.

    Parameters
    ----------
    ds : xr.Dataset or PingGeometry
        A georeferenced section (measured with
        :func:`sidescanpy.analysis.measure_ping_geometry`), or an already
        measured :class:`~sidescanpy.analysis.PingGeometry`.

    Returns
    -------
    float or None
        Resolution in metres, or None when ping spacing can't be measured
        (fewer than 2 pings, or no along-track navigation).
    """
    geometry = ds if isinstance(ds, PingGeometry) else measure_ping_geometry(ds)
    spacing = geometry.ping_spacing_m_median
    if spacing is None:
        logger.debug("recommend_resolution: no ping spacing measurable")
        return None
    interval = geometry.sample_interval_m
    finest = spacing if interval is None else min(spacing, interval)
    return finest / 2.0
