"""
Slicing operations for SideScan frequency channels.

Each function returns a new SideScanFrequencyChannel — sliced version
of the input. The schema is preserved; the result is validated. Empty
slices (zero pings or zero samples) are valid and return an empty-but-
schema-valid channel.

Slicing is structural: backscatter values, processing_level, and
processing_history are preserved. No values are modified.

Available slicing functions:

- ``slice_by_pings`` — by ping index range
- ``slice_by_samples`` — by sample index range
- ``slice_by_time`` — by ping timestamp range (supports str or datetime64)
- ``slice_by_bbox`` — by lat/lon bounding box (per-ping by default,
  per-sample with ``use_per_sample=True``)
- ``slice_by_ground_range`` — by maximum ground range (requires
  ``ground_range_m`` to be present, run ``compute_geometric_ranges``
  first)

2026 rshomberg@uri.edu
"""

import logging
from typing import Union

import numpy as np

from sidescanpy.core.survey import SideScanFrequencyChannel

logger = logging.getLogger(__name__)


# ============================================================
# SLICE BY PINGS
# ============================================================

def slice_by_pings(
    channel: SideScanFrequencyChannel,
    *,
    start: int = 0,
    stop: int | None = None,
) -> SideScanFrequencyChannel:
    """Return a new channel with pings ``[start, stop)``.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Input channel.
    start : int, default 0
        First ping index to include.
    stop : int, optional
        One past the last ping index to include. Defaults to ``n_pings``.

    Returns
    -------
    SideScanFrequencyChannel
        New channel with the selected ping range.
    """
    n_pings = channel.n_pings
    if stop is None:
        stop = n_pings
    stop = min(stop, n_pings)

    new_ds = channel.dataset.isel(ping=slice(start, stop))
    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


# ============================================================
# SLICE BY SAMPLES
# ============================================================

def slice_by_samples(
    channel: SideScanFrequencyChannel,
    *,
    start: int = 0,
    stop: int | None = None,
) -> SideScanFrequencyChannel:
    """Return a new channel with samples ``[start, stop)``.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Input channel.
    start : int, default 0
        First sample index to include.
    stop : int, optional
        One past the last sample index. Defaults to ``n_samples``.

    Returns
    -------
    SideScanFrequencyChannel
        New channel with the selected sample range.
    """
    n_samples = channel.n_samples
    if stop is None:
        stop = n_samples
    stop = min(stop, n_samples)

    new_ds = channel.dataset.isel(sample=slice(start, stop))
    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


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
    channel: SideScanFrequencyChannel,
    *,
    start: DateTimeLike = None,
    stop: DateTimeLike = None,
) -> SideScanFrequencyChannel:
    """Return a new channel with pings whose timestamp is in ``[start, stop)``.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Input channel.
    start : str, np.datetime64, or None
        Inclusive lower bound. Strings are parsed as ISO-format. ``None``
        means no lower bound.
    stop : str, np.datetime64, or None
        Exclusive upper bound. ``None`` means no upper bound.

    Returns
    -------
    SideScanFrequencyChannel
        New channel containing only pings whose ``time`` falls in the range.
    """
    start_ns = _to_datetime64_ns(start)
    stop_ns = _to_datetime64_ns(stop)

    times = channel.dataset["time"].values
    mask = np.ones(channel.n_pings, dtype=bool)
    if start_ns is not None:
        mask &= times >= start_ns
    if stop_ns is not None:
        mask &= times < stop_ns

    selected = np.where(mask)[0]
    new_ds = channel.dataset.isel(ping=selected)
    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


# ============================================================
# SLICE BY BBOX
# ============================================================

def slice_by_bbox(
    channel: SideScanFrequencyChannel,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    use_per_sample: bool = False,
) -> SideScanFrequencyChannel:
    """Return a new channel containing pings whose position is within the bbox.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Input channel.
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
    SideScanFrequencyChannel
        New channel with pings within the bbox.
    """
    ds = channel.dataset

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
    return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)


# ============================================================
# SLICE BY GROUND RANGE
# ============================================================

def slice_by_ground_range(
    channel: SideScanFrequencyChannel,
    *,
    range_max_m: float,
) -> SideScanFrequencyChannel:
    """Return a new channel with samples whose ground_range_m is ≤ range_max_m.

    Trims the sample dimension to the maximum sample index where any
    ping/channel has ground_range_m ≤ range_max_m. Pings with shorter
    in-range coverage retain NaN beyond their individual cutoffs.

    Parameters
    ----------
    channel : SideScanFrequencyChannel
        Input channel. Must have ``ground_range_m`` (run
        ``compute_geometric_ranges`` first).
    range_max_m : float
        Maximum ground range to retain (meters).

    Returns
    -------
    SideScanFrequencyChannel
        New channel with truncated sample dim.
    """
    ds = channel.dataset
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
        return SideScanFrequencyChannel(new_ds, freq_key=channel.freq_key)

    # Find the largest sample index that's still in range
    last_valid = int(np.where(valid_per_sample)[0].max()) + 1
    return slice_by_samples(channel, start=0, stop=last_valid)
