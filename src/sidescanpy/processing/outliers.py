"""
Median/MAD outlier rejection.

:func:`reject_range_outliers` drops samples far from their own range
bin's median. Meant for calibration-style fits that assume a flat,
homogeneous bottom population (noise-floor and range-decay estimates),
not for display or export: it deliberately discards genuine bright
returns such as hard targets.

:func:`mask_bright_ping_rows` drops whole pings whose overall level is an
outlier -- the full-width bright streak a per-ping automatic gain jump
leaves.
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


def mask_bright_ping_rows(array_2d: np.ndarray, mad_k: float = 3.5) -> np.ndarray:
    """NaN out whole ping rows whose typical level is an extreme outlier.

    Hardware with per-ping automatic gain control occasionally raises one
    ping's gain well above its neighbors, leaving a uniformly bright,
    full-width streak. The gain multiplies the whole ping, so clipping
    values only dims the streak; masking the row removes it. Each row's
    level is its median; a row is masked when its modified z-score
    ``0.6745 * |level - median| / MAD`` across rows exceeds ``mad_k``.

    Parameters
    ----------
    array_2d : np.ndarray
        ``(n_pings, n_samples)``.
    mad_k : float, default 3.5
        Modified z-score threshold (Iglewicz & Hoaglin's "outlier" cutoff).

    Returns
    -------
    np.ndarray
        ``array_2d`` itself when nothing is masked (or fewer than 5 rows
        have a finite level, or the MAD is 0), otherwise a copy with the
        outlier rows NaN.
    """
    row_level = np.nanmedian(array_2d, axis=1)
    finite = np.isfinite(row_level)
    if finite.sum() < 5:
        return array_2d
    med = np.nanmedian(row_level[finite])
    mad = np.nanmedian(np.abs(row_level[finite] - med))
    if mad == 0:
        return array_2d
    z = 0.6745 * (row_level - med) / mad
    bright_rows = np.abs(z) > mad_k
    if not bright_rows.any():
        return array_2d
    logger.debug(f"masking {int(bright_rows.sum())} of {row_level.size} outlier ping rows")
    out = array_2d.copy()
    out[bright_rows, :] = np.nan
    return out
