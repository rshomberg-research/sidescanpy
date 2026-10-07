"""
Per-range-bin outlier rejection.

:func:`reject_range_outliers` drops samples far from their own range
bin's median (median/MAD test). Meant for calibration-style fits that
assume a flat, homogeneous bottom population (noise-floor and
range-decay estimates), not for display or export: it deliberately
discards genuine bright returns such as hard targets.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reject_range_outliers(
    values: np.ndarray,
    ranges: np.ndarray,
    *,
    bin_width_m: float = 5.0,
    mad_k: float = 3.5,
    min_bin_count: int = 5,
) -> np.ndarray:
    """Return a copy of ``values`` with per-range-bin outliers set to NaN.

    Samples are grouped into ``bin_width_m`` range bins; within each bin
    a sample is rejected when its modified z-score
    ``0.6745 * |value - median| / MAD`` exceeds ``mad_k``. Median/MAD
    rather than mean/std, so the right-skewed backscatter distribution
    doesn't hide the outliers it contains.

    Parameters
    ----------
    values, ranges : np.ndarray
        Same shape, any dims (flattened internally, reshaped on return).
    bin_width_m : float, default 5.0
        Range-bin width for the local median/MAD.
    mad_k : float, default 3.5
        Modified z-score threshold (Iglewicz & Hoaglin's "outlier"
        cutoff). Lower rejects more.
    min_bin_count : int, default 5
        Bins with fewer finite samples are left unfiltered.

    Returns
    -------
    np.ndarray
        float64 copy of ``values``, same shape, rejected samples NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    shape = values.shape
    v_flat = values.ravel().copy()
    r_flat = ranges.ravel()

    valid = np.isfinite(v_flat) & np.isfinite(r_flat)
    if not valid.any():
        return v_flat.reshape(shape)

    edges = np.arange(0.0, r_flat[valid].max() + bin_width_m, bin_width_m)
    bin_idx = np.digitize(r_flat, edges) - 1

    for i in range(len(edges) - 1):
        sel = valid & (bin_idx == i)
        if int(sel.sum()) < min_bin_count:
            continue
        v = v_flat[sel]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad == 0:
            continue  # no scale to reject against (e.g. many identical values)
        modified_z = 0.6745 * (v - med) / mad
        reject = np.abs(modified_z) > mad_k
        if reject.any():
            v_flat[np.flatnonzero(sel)[reject]] = np.nan

    return v_flat.reshape(shape)
