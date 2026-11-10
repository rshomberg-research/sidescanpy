"""
Range- and sample-binned backscatter profiles.

Binning primitives used by the TVG range-correction fit in
:mod:`sidescanpy.processing.range_correction` and by the TVG analysis in
:mod:`sidescanpy.analysis.tvg` (which re-exports both functions).

- :func:`backscatter_vs_range` — per-range-bin statistics.
- :func:`backscatter_vs_sample` — per-sample-index-bin statistics.
- :func:`median_level_profile` — median level of linear amplitude per
  range bin, in dB, with dither for integer samples (the profile the TVG
  model fit in :mod:`sidescanpy.analysis.tvg_fit` works on).

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.processing.sample_axis import sample_index

logger = logging.getLogger(__name__)


def _bin_stats(values: np.ndarray, ranges: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Shared binning/aggregation core for the two profile functions."""
    bin_idx = np.digitize(ranges, edges) - 1

    rows = []
    for i in range(len(edges) - 1):
        sel = bin_idx == i
        n = int(sel.sum())
        if n == 0:
            continue
        v = values[sel]
        rows.append(
            {
                "min": float(edges[i]),
                "max": float(edges[i + 1]),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "std": float(v.std()),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
                "count": n,
            }
        )
    return rows


def backscatter_vs_range(
    ds: xr.Dataset,
    value_var: str,
    range_var: str = "ground_range_m",
    bin_width_m: float = 5.0,
    exclude_below_db: float | None = None,
) -> pd.DataFrame:
    """
    Bin samples by range and return per-bin mean/median/std/p10/p90/count.

    Parameters
    ----------
    ds : xr.Dataset
        Any dataset with `value_var` and `range_var` (same shape,
        arbitrary dims — flattened internally). Does not need to be a
        full valid sidescan schema dataset.
    value_var : str
        Name of the backscatter (or other) variable to aggregate.
    range_var : str, default "ground_range_m"
    bin_width_m : float, default 5.0
    exclude_below_db : float, optional
        If given, samples with `value_var < exclude_below_db` are
        excluded before binning (e.g. Known Issue #5's floor-clipped
        samples). Default `None` (include everything) is deliberate:
        whether to exclude floor-clipped samples is one of the open
        questions this tool exists to help answer, not a decision to
        bake in silently.

    Returns
    -------
    pd.DataFrame
        Columns: range_min_m, range_max_m, mean, median, std, p10, p90,
        count. Empty (zero rows) if no samples survive filtering.
    """
    values = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
    ranges = np.asarray(ds[range_var].values, dtype=np.float64).ravel()

    valid = np.isfinite(values) & np.isfinite(ranges)
    if exclude_below_db is not None:
        valid &= values >= exclude_below_db

    values = values[valid]
    ranges = ranges[valid]

    columns = ["range_min_m", "range_max_m", "mean", "median", "std", "p10", "p90", "count"]
    if values.size == 0:
        return pd.DataFrame(columns=columns)

    edges = np.arange(0.0, ranges.max() + bin_width_m, bin_width_m)
    rows = _bin_stats(values, ranges, edges)
    df = pd.DataFrame(rows)
    return df.rename(columns={"min": "range_min_m", "max": "range_max_m"})[columns]


def median_level_profile(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    range_var: str = "slant_range_m",
    *,
    bin_width_m: float = 1.0,
    channel: int | None = None,
    min_range_m: float | None = None,
    exclude_water_column: bool = True,
    min_count: int = 50,
    dither: bool | str = "auto",
    seed: int = 0,
) -> pd.DataFrame:
    """Median level of linear amplitude per range bin, in dB re 1.

    Every bin from 0 to the largest range is returned, so the rows form a
    regular range axis; a bin with fewer than ``min_count`` samples has a
    NaN level. The median is taken on the linear values and converted with
    ``20 log10``; a non-positive median gives NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Channel dataset with ``value_var`` (linear amplitude) and
        ``range_var`` of the same shape, or broadcastable to it.
    value_var : str, default "backscatter_intensity"
    range_var : str, default "slant_range_m"
    bin_width_m : float, default 1.0
    channel : int, optional
        Use one side only (0 port, 1 starboard). ``None`` pools both.
    min_range_m : float, optional
        Drop samples at or below this range, e.g. the survey's median
        altitude (one cut for all pings keeps each bin's pings the same).
    exclude_water_column : bool, default True
        Drop samples where ``ground_range_m`` is NaN (slant range below
        the altitude). No-op when ``ground_range_m`` is missing.
    min_count : int, default 50
        Fewer samples than this in a bin give a NaN level.
    dither : bool or "auto", default "auto"
        Add a uniform offset in ``[-0.5, 0.5)`` to each sample before
        taking medians, which spreads integer samples over their unit
        interval so the medians do not stick to whole numbers. ``"auto"``
        dithers only when every finite sample is a whole number.
    seed : int, default 0
        Seed of the dither generator (the profile is reproducible).

    Returns
    -------
    pd.DataFrame
        Columns ``range_min_m``, ``range_max_m``, ``range_m`` (bin center),
        ``median`` (linear), ``level_db`` and ``count``.
        ``df.attrs["dithered"]`` records whether dither was applied.
    """
    values = np.asarray(ds[value_var].values, dtype=np.float64)
    ranges = np.broadcast_to(np.asarray(ds[range_var].values, dtype=np.float64), values.shape)
    keep = np.isfinite(values) & np.isfinite(ranges)
    if exclude_water_column and "ground_range_m" in ds:
        ground = np.broadcast_to(np.asarray(ds["ground_range_m"].values), values.shape)
        keep &= np.isfinite(ground)
    if min_range_m is not None:
        keep &= ranges > min_range_m
    if channel is not None:
        axis = ds[value_var].dims.index("channel")
        side = np.zeros(values.shape[axis], dtype=bool)
        side[int(channel)] = True
        shape = [1] * values.ndim
        shape[axis] = -1
        keep &= side.reshape(shape)

    finite = values[np.isfinite(values)]
    if dither == "auto":
        dither = bool(finite.size) and bool(np.all(finite == np.round(finite)))
    dither = bool(dither)
    if dither:
        values = values + np.random.default_rng(seed).uniform(-0.5, 0.5, values.shape)

    vals = values[keep]
    rng_m = ranges[keep]
    max_range = float(np.nanmax(ranges)) if np.isfinite(ranges).any() else 0.0
    n_bins = max(1, int(np.ceil(max_range / bin_width_m)) + 1)
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_width_m

    idx = np.floor(rng_m / bin_width_m).astype(np.int64)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    bounds = np.searchsorted(idx, np.arange(n_bins + 1))
    counts = np.diff(bounds)
    median = np.full(n_bins, np.nan)
    for i in np.flatnonzero(counts >= max(min_count, 1)):
        median[i] = np.median(vals[bounds[i]:bounds[i + 1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        level = 20.0 * np.log10(median)
    level[~np.isfinite(level)] = np.nan

    df = pd.DataFrame({
        "range_min_m": edges[:-1],
        "range_max_m": edges[1:],
        "range_m": 0.5 * (edges[:-1] + edges[1:]),
        "median": median,
        "level_db": level,
        "count": counts,
    })
    df.attrs["dithered"] = dither
    return df


def backscatter_vs_sample(
    ds: xr.Dataset,
    value_var: str,
    bin_width: int = 50,
    exclude_below_db: float | None = None,
) -> pd.DataFrame:
    """
    Bin samples by absolute sample index (not range) and return per-bin stats.

    Works before geometric ranges exist, and for artifacts fixed in
    sample position rather than range (e.g. a leading blanking run).

    Parameters
    ----------
    ds : xr.Dataset
        Must have a ``sample`` dimension and ``value_var`` defined over
        it (other dims are flattened).
    value_var : str
    bin_width : int, default 50
        Bin width in sample-index units.
    exclude_below_db : float, optional
        See :func:`backscatter_vs_range`.

    Returns
    -------
    pd.DataFrame
        Columns: sample_min, sample_max, mean, median, std, p10, p90,
        count.
    """
    values_da = ds[value_var]
    abs_idx = sample_index(ds)
    sample_idx_da = xr.DataArray(abs_idx, dims=["sample"])
    idx_b, values_b = xr.broadcast(sample_idx_da, values_da)

    idx_flat = idx_b.values.ravel().astype(np.float64)
    values_flat = values_b.values.ravel().astype(np.float64)

    valid = np.isfinite(values_flat) & np.isfinite(idx_flat)
    if exclude_below_db is not None:
        valid &= values_flat >= exclude_below_db

    values_flat = values_flat[valid]
    idx_flat = idx_flat[valid]

    columns = ["sample_min", "sample_max", "mean", "median", "std", "p10", "p90", "count"]
    if values_flat.size == 0:
        return pd.DataFrame(columns=columns)

    edges = np.arange(0.0, (abs_idx.max() if abs_idx.size else 0.0) + 1 + bin_width, bin_width)
    rows = _bin_stats(values_flat, idx_flat, edges)
    df = pd.DataFrame(rows)
    return df.rename(columns={"min": "sample_min", "max": "sample_max"})[columns]
