"""
Range-binned backscatter profiles.

Binning primitive used both by the TVG range-correction fit in
:mod:`sidescanpy.processing.range_correction` and by the diagnostics in
:mod:`sidescanpy.analysis.tvg_diagnostics` (which re-exports
:func:`mean_backscatter_vs_range`). Pure computation, no plotting.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _bin_stats(values: np.ndarray, ranges: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Shared binning/aggregation core for the two mean-vs-X functions."""
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


def mean_backscatter_vs_range(
    ds: xr.Dataset,
    value_var: str,
    range_var: str = "ground_range_m",
    bin_width_m: float = 5.0,
    exclude_below_db: float | None = None,
) -> pd.DataFrame:
    """
    Bin samples by range and return per-bin mean/median/std/p10/p90/count.

    Generalizes the manual binning loop from
    `docs/planning/handoff-tvg-range-residual.md`'s "How to reproduce"
    section into a reusable function.

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
