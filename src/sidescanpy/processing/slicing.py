"""
Slicing operations for SideScan frequency channels.

Every function takes an ``xr.Dataset`` (one frequency channel) and returns
a new ``xr.Dataset`` -- the sliced version of the input. Empty slices
(zero pings or zero samples) are valid and return an empty-but-
schema-valid dataset. ``slice_survey`` maps any of them over every
frequency of a ``SideScanSurvey``.

Slicing is structural: backscatter values, processing_level, and
processing_history are preserved. No values are modified.

Available slicing functions:

- ``slice_by_pings`` — by ping index range
- ``slice_by_samples`` — by absolute ``sample`` coordinate range
- ``slice_by_range`` — by slant range in metres
- ``slice_by_time`` — by ping timestamp range (supports str or datetime64)
- ``slice_by_bbox`` — by lat/lon bounding box (per-ping by default,
  per-sample with ``use_per_sample=True``)
- ``slice_by_ground_range`` — by maximum ground range (requires
  ``ground_range_m`` to be present, run ``compute_geometric_ranges``
  first)

2026 rshomberg@uri.edu
"""

import logging
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.processing.sample_axis import sample_index, sample_interval_m

logger = logging.getLogger(__name__)


# ============================================================
# SLICE BY PINGS
# ============================================================

def slice_by_pings(
    ds: xr.Dataset,
    *,
    start: int = 0,
    stop: int | None = None,
) -> xr.Dataset:
    """Return a new dataset with pings ``[start, stop)``.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset.
    start : int, default 0
        First ping index to include.
    stop : int, optional
        One past the last ping index to include. Defaults to ``n_pings``.

    Returns
    -------
    xr.Dataset
        New dataset with the selected ping range.
    """
    n_pings = ds.sizes["ping"]
    if stop is None:
        stop = n_pings
    stop = min(stop, n_pings)

    new_ds = ds.isel(ping=slice(start, stop))
    return new_ds


# ============================================================
# SLICE BY SAMPLES
# ============================================================

def slice_by_samples(
    ds: xr.Dataset,
    *,
    start: int = 0,
    stop: int | None = None,
) -> xr.Dataset:
    """Return a new dataset with samples whose index is in ``[start, stop)``.

    Selection is by the *absolute* integer ``sample`` coordinate (label
    based: 0 is the first recorded sample of the ping), so slicing an
    already-sliced dataset keeps meaning the same samples. Datasets with
    no ``sample`` coordinate fall back to positional selection.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset.
    start : int, default 0
        First absolute sample index to include.
    stop : int, optional
        One past the last absolute sample index. Defaults to unbounded.

    Returns
    -------
    xr.Dataset
        New dataset with the selected sample range.
    """
    if "sample" not in ds.coords:
        n_samples = ds.sizes["sample"]
        stop_pos = n_samples if stop is None else min(stop, n_samples)
        return ds.isel(sample=slice(start, stop_pos))
    idx = ds["sample"].values
    mask = idx >= start
    if stop is not None:
        mask &= idx < stop
    return ds.isel(sample=np.flatnonzero(mask))


# ============================================================
# SLICE BY RANGE
# ============================================================

def slice_by_range(
    ds: xr.Dataset,
    min_m: float | None = None,
    max_m: float | None = None,
) -> xr.Dataset:
    """Return a new dataset with samples whose slant range is in ``[min_m, max_m]``.

    Slant range is ``sample * c / (2 * fs)`` from the absolute ``sample``
    coordinate, using the mean per-ping sample interval (the same
    ``sound_vel_ms``/``sample_rate_hz`` logic as
    ``compute_geometric_ranges``). Because sound velocity varies slightly
    per ping, the cut is made at a single sample boundary for the whole
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset.
    min_m, max_m : float, optional
        Inclusive slant-range bounds in metres. ``None`` means unbounded.
    """
    interval = float(np.nanmean(sample_interval_m(ds))) if ds.sizes["ping"] else np.nan
    rng = sample_index(ds) * interval
    mask = np.ones(rng.shape, dtype=bool)
    if min_m is not None:
        mask &= rng >= min_m
    if max_m is not None:
        mask &= rng <= max_m
    return ds.isel(sample=np.flatnonzero(mask))


# ============================================================
# SLICE BY TIME
# ============================================================

DateTimeLike = Union[str, np.datetime64, None]


def _to_datetime64_ns(value: DateTimeLike) -> np.datetime64 | None:
    """Convert a string or datetime64 to datetime64[ns]. Pass through None."""
    if value is None:
        return None
    if isinstance(value, str):
        return np.datetime64(value, "ns")
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    raise TypeError(f"Expected str, np.datetime64, or None; got {type(value)}")


def slice_by_time(
    ds: xr.Dataset,
    *,
    start: DateTimeLike = None,
    stop: DateTimeLike = None,
) -> xr.Dataset:
    """Return a new dataset with pings whose timestamp is in ``[start, stop)``.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset.
    start : str, np.datetime64, or None
        Inclusive lower bound. Strings are parsed as ISO-format. ``None``
        means no lower bound.
    stop : str, np.datetime64, or None
        Exclusive upper bound. ``None`` means no upper bound.

    Returns
    -------
    xr.Dataset
        New dataset containing only pings whose ``time`` falls in the range.
    """
    start_ns = _to_datetime64_ns(start)
    stop_ns = _to_datetime64_ns(stop)

    times = ds["time"].values
    mask = np.ones(ds.sizes["ping"], dtype=bool)
    if start_ns is not None:
        mask &= times >= start_ns
    if stop_ns is not None:
        mask &= times < stop_ns

    selected = np.where(mask)[0]
    new_ds = ds.isel(ping=selected)
    return new_ds


# ============================================================
# SLICE BY BBOX
# ============================================================

def slice_by_bbox(
    ds: xr.Dataset,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    use_per_sample: bool = False,
) -> xr.Dataset:
    """Return a new dataset containing pings whose position is within the bbox.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset.
    lat_min, lat_max : float
        Latitude bounds (inclusive). degrees_north.
    lon_min, lon_max : float
        Longitude bounds (inclusive). degrees_east.
    use_per_sample : bool, default False
        If False, a ping is kept when its per-ping ``latitude`` and
        ``longitude`` are both within the bbox.

        If True, a ping is kept when *any* of its samples
        (``latitude_per_sample`` and ``longitude_per_sample``) is within
        the bbox. Requires ``compute_georeferencing`` to have been run.

    Returns
    -------
    xr.Dataset
        New dataset with pings within the bbox.
    """
    if use_per_sample:
        if "latitude_per_sample" not in ds.data_vars:
            raise ValueError(
                "use_per_sample=True requires 'latitude_per_sample' in the "
                "dataset. Run compute_georeferencing first."
            )
        if "longitude_per_sample" not in ds.data_vars:
            raise ValueError(
                "use_per_sample=True requires 'longitude_per_sample' in the "
                "dataset. Run compute_georeferencing first."
            )
        lat_arr = ds["latitude_per_sample"].values     # (ping, channel, sample)
        lon_arr = ds["longitude_per_sample"].values
        in_box = (
            (lat_arr >= lat_min) & (lat_arr <= lat_max) &
            (lon_arr >= lon_min) & (lon_arr <= lon_max)
        )
        mask = in_box.any(axis=(1, 2))   # any sample in any channel
    else:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )

    selected = np.where(mask)[0]
    new_ds = ds.isel(ping=selected)
    return new_ds


# ============================================================
# SLICE BY GROUND RANGE
# ============================================================

def slice_by_ground_range(
    ds: xr.Dataset,
    *,
    range_max_m: float,
) -> xr.Dataset:
    """Return a new dataset with samples whose ground_range_m is ≤ range_max_m.

    Trims the sample dimension to the maximum sample index where any
    ping/channel has ground_range_m ≤ range_max_m. Pings with shorter
    in-range coverage retain NaN beyond their individual cutoffs.

    Parameters
    ----------
    ds : xr.Dataset
        Input frequency-channel dataset. Must have ``ground_range_m`` (run
        ``compute_geometric_ranges`` first).
    range_max_m : float
        Maximum ground range to retain (meters).

    Returns
    -------
    xr.Dataset
        New dataset with truncated sample dim.
    """
    if "ground_range_m" not in ds.data_vars:
        raise ValueError(
            "slice_by_ground_range requires 'ground_range_m' in the dataset. "
            "Run compute_geometric_ranges first."
        )

    gr = ds["ground_range_m"].values   # (ping, channel, sample)
    in_range = gr <= range_max_m       # NaN values give False naturally

    # For each sample, is it in range for ANY ping/channel?
    valid_per_sample = in_range.any(axis=(0, 1))

    if not valid_per_sample.any():
        # No sample is within range — return empty sample dim
        new_ds = ds.isel(sample=slice(0, 0))
        return new_ds

    # Find the largest sample index that's still in range
    last_valid = int(np.where(valid_per_sample)[0].max()) + 1
    stop_abs = int(sample_index(ds)[last_valid - 1]) + 1
    return slice_by_samples(ds, start=0, stop=stop_abs)


# ============================================================
# SURVEY-LEVEL MAPPING
# ============================================================

def slice_survey(
    survey: SideScanSurvey,
    fn: Callable[..., xr.Dataset],
    **kwargs: Any,
) -> SideScanSurvey:
    """Apply a ``Dataset`` slicer to every frequency of ``survey``.

    Returns a new survey with the same metadata and each frequency
    replaced by ``fn(ds, **kwargs)``. Aux streams are not carried over.
    """
    out = SideScanSurvey(
        source_files=survey.source_files,
        source_format=survey.source_format,
        survey_name=survey.survey_name,
    )
    for fk in survey.frequencies:
        out.add_frequency(fk, fn(survey.get_frequency(fk).dataset, **kwargs))
    return out
