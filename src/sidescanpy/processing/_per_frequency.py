"""
Let a one-channel correction step take a whole survey.

:func:`per_frequency` wraps a step function ``fn(ds, ...) -> ds`` so it
also accepts a :class:`~sidescanpy.core.survey.SideScanSurvey`: the step
then runs on every frequency channel with the same arguments and a new
(shallow-copied) survey is returned, leaving the input untouched. A
dataset argument is passed straight through.

2026 rshomberg@uri.edu
"""

import functools
import logging
from collections.abc import Callable

import xarray as xr

from sidescanpy.core.survey import SideScanSurvey

logger = logging.getLogger(__name__)


def per_frequency(fn: Callable[..., xr.Dataset]) -> Callable:
    """Make ``fn(ds, ...)`` accept a survey as its first argument too."""

    @functools.wraps(fn)
    def wrapper(ds, *args, **kwargs):
        if not isinstance(ds, SideScanSurvey):
            return fn(ds, *args, **kwargs)
        out = ds.copy()
        for freq_key, channel in ds.freq_channels.items():
            logger.debug("%s: %s", fn.__name__, freq_key)
            out.add_frequency(freq_key, fn(channel.dataset, *args, **kwargs))
        return out

    return wrapper
