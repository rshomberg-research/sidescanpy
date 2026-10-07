"""
Small helpers shared by the range-binned analysis functions.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def resolve_channel_idx(ds: xr.Dataset, channel: int | str, *, caller: str) -> int:
    """``channel`` (int position, or a ``ds["channel_id"]`` value) → int index.

    Range-binned analyses restrict to one channel by masking samples,
    not by slicing ``ds``: the schema requires the ``channel`` dimension
    to stay size 2.
    """
    if isinstance(channel, str):
        channel_ids = np.asarray(ds["channel_id"].values)
        matches = np.flatnonzero(channel_ids == channel)
        if matches.size == 0:
            raise ValueError(f"{caller}: channel={channel!r} not found in channel_id={list(channel_ids)}")
        return int(matches[0])
    return int(channel)


def water_column_var(ds: xr.Dataset, range_var: str) -> str:
    """Variable whose NaN pattern marks the water column.

    Always ``ground_range_m`` when present (NaN where slant range is
    below altitude), whatever ``range_var`` the analysis bins by:
    ``slant_range_m`` is finite everywhere. Falls back to ``range_var``.
    """
    return "ground_range_m" if "ground_range_m" in ds else range_var


def channel_mask(ds: xr.Dataset, like: xr.DataArray, channel_idx: int) -> np.ndarray:
    """Boolean array shaped like ``like``: ``True`` on channel ``channel_idx`` only."""
    selector = xr.DataArray(np.arange(ds.sizes["channel"]) == channel_idx, dims=["channel"])
    return selector.broadcast_like(like).values
