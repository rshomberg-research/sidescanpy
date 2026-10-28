"""
Absolute sample coordinate and per-ping sample spacing helpers.

Converters write an integer ``sample`` coordinate counted in sample
intervals from the transmit, so slant range is ``sample x interval``. 0 is
normally the first recorded sample of the ping; a JSF starting depth
(window offset) moves the first recorded sample to that offset. Range math must use that *absolute* index,
not the local position along the (possibly sliced) sample axis, so that
slicing leading samples never changes the computed ranges.

2026 rshomberg@uri.edu
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_SOUND_VEL_MS = 1500.0
"""Fallback sound velocity (m/s) for NaN / non-positive per-ping values."""


_warn_invalid_sound_vel: ContextVar[bool] = ContextVar("warn_invalid_sound_vel", default=True)


@contextmanager
def invalid_sound_vel_warning_suppressed() -> Iterator[None]:
    """Silence the invalid-``sound_vel_ms`` warning inside the block.

    For callers that process one channel in pieces (ping chunks): they warn
    once for the whole channel themselves instead of once per piece.
    """
    token = _warn_invalid_sound_vel.set(False)
    try:
        yield
    finally:
        _warn_invalid_sound_vel.reset(token)


def sample_index(ds: xr.Dataset) -> np.ndarray:
    """Absolute sample index of every position along ``ds``'s sample axis.

    Returns the ``sample`` coordinate as float64 when present, else
    ``arange(n_samples)`` (assumes the axis starts at the first recorded
    sample).
    """
    if "sample" in ds.coords:
        return np.asarray(ds["sample"].values, dtype=np.float64)
    logger.debug("no sample coordinate; assuming local index from 0")
    return np.arange(ds.sizes["sample"], dtype=np.float64)


def sample_interval_m(ds: xr.Dataset) -> np.ndarray:
    """Per-ping slant-range spacing of one sample, ``c / (2 * fs)``, shape (ping,).

    NaN or non-positive ``sound_vel_ms`` falls back to
    :data:`DEFAULT_SOUND_VEL_MS` with a warning.
    """
    sound_vel = ds["sound_vel_ms"].values.astype(np.float64)
    invalid = np.isnan(sound_vel) | (sound_vel <= 0)
    if invalid.any():
        (logger.warning if _warn_invalid_sound_vel.get() else logger.debug)(
            "%d of %d pings have invalid sound_vel_ms (NaN or non-positive); "
            "falling back to %s m/s",
            int(invalid.sum()), sound_vel.size, DEFAULT_SOUND_VEL_MS,
        )
        sound_vel = np.where(invalid, DEFAULT_SOUND_VEL_MS, sound_vel)
    return sound_vel / (2.0 * float(ds.attrs["sample_rate_hz"]))
