"""
Single source for linear <-> dB conversion and the floor-clip artifact.

Convention: every conversion in sidescanpy is **amplitude** (20*log10),
because raw ``backscatter_intensity`` is linear amplitude. There is no
power (10*log10) conversion anywhere in the library.

Linear values below :data:`MIN_LINEAR` are clipped before the log so
exact zeros (blanking / ring-down) map to :data:`FLOOR_DB` instead of
``-inf``. That deterministic floor is an artifact, not a real return;
:func:`mask_floor_clip` / :func:`mask_floor_db` turn it into NaN.

2026 rshomberg@uri.edu
"""

import numpy as np
import xarray as xr

MIN_LINEAR = 1e-6
"""Smallest linear amplitude kept before taking the log."""

FLOOR_DB = -120.0
"""``20*log10(MIN_LINEAR)``: the dB value exact-zero amplitude converts to."""

FLOOR_CLIP_VALUES = (0.0, FLOOR_DB)
"""Deterministic floor-clip artifact values: exact-zero linear amplitude
(pre-dB) and the dB floor (post-dB)."""


def linear_to_db(x, floor_db: float = FLOOR_DB):
    """Convert linear amplitude to dB: ``20*log10(clip(x, min_linear))``.

    ``min_linear`` is the linear value of ``floor_db`` (``1e-6`` for the
    default -120 dB). Accepts scalars, numpy arrays and ``xr.DataArray``;
    dtype and NaN are preserved (NaN stays NaN).
    """
    min_linear = MIN_LINEAR if floor_db == FLOOR_DB else 10.0 ** (floor_db / 20.0)
    if isinstance(x, xr.DataArray):
        return 20.0 * np.log10(x.clip(min=min_linear))
    return 20.0 * np.log10(np.clip(x, min_linear, None))


def db_to_linear(db):
    """Convert dB to linear amplitude: ``10**(db/20)``."""
    return 10.0 ** (db / 20.0)


def mask_floor_clip(values: np.ndarray) -> np.ndarray:
    """Replace the exact floor-clip artifact values (0.0 and -120 dB) with NaN."""
    mask = np.isin(values, FLOOR_CLIP_VALUES)
    if not mask.any():
        return values
    return np.where(mask, np.nan, values)


def mask_floor_db(db: np.ndarray) -> np.ndarray:
    """Replace dB values at or below :data:`FLOOR_DB` with NaN."""
    mask = db <= FLOOR_DB
    if not mask.any():
        return db
    return np.where(mask, np.nan, db)
