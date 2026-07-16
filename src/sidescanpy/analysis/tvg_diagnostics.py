"""
TVG diagnostic analysis primitives.

Stage 1 of docs/planning/v0.8.0-tvg-diagnostics.md — pure computation
(no matplotlib) supporting investigation of the range-dependent TVG
residual (CLAUDE.md Known Issue #4) and floor-clipped samples (Known
Issue #5), without changing `processing/radiometric.py`'s correction
math. Consumes `apply_tvg()`'s existing public API as-is.

2026 rshomberg@uri.edu
"""

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.pipeline import apply_correction_steps
from sidescanpy.processing.radiometric import TVGParams, apply_tvg

_MIN_LINEAR = 1e-6


def _linear_to_db(x, min_linear: float = _MIN_LINEAR):
    """
    Convert linear amplitude to dB, matching `apply_tvg()`'s step-1
    arithmetic exactly (`20*log10(clip(x, 1e-6))`).

    Deliberately duplicated here rather than imported from
    `radiometric.py` — nothing in that module's public API exposes
    "dB, no TVG gain," and this function must not require modifying
    the module under investigation. If this is needed elsewhere too,
    promoting it to a shared helper is a Stage 2+ cleanup, not now.
    """
    if isinstance(x, xr.DataArray):
        clipped = x.clip(min=min_linear)
        return 20.0 * np.log10(clipped)
    clipped = np.clip(x, min_linear, None)
    return 20.0 * np.log10(clipped)


def before_after_backscatter(
    ds_raw: xr.Dataset, params: TVGParams | None = None
) -> xr.Dataset:
    """
    Compute before/after TVG backscatter alongside a shared ground range.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Valid frequency channel dataset with processing_level == "raw".
    params : TVGParams, optional
        Custom TVG parameters (e.g. for an absorption-coefficient sweep).
        Passed directly to `apply_tvg()` — bypasses
        `apply_correction_steps()`'s fixed step functions, which don't
        expose a way to override TVG parameters.

    Returns
    -------
    xr.Dataset
        Carries `ground_range_m` (shared between the raw and
        TVG-corrected passes — geometry doesn't depend on backscatter
        state), `backscatter_db_raw` (linear->dB, no TVG gain),
        `backscatter_db_corrected` (real `apply_tvg()` output), and
        `tvg_gain_db` (corrected - raw, the actual applied gain curve).
    """
    ds_geom = apply_correction_steps(ds_raw, {"geometric"})
    ds_tvg = apply_tvg(ds_raw, params)

    backscatter_db_raw = _linear_to_db(ds_raw["backscatter_intensity"])
    backscatter_db_corrected = ds_tvg["backscatter_intensity"]
    tvg_gain_db = backscatter_db_corrected - backscatter_db_raw

    out = ds_geom.copy(deep=False)
    out.attrs = dict(ds_geom.attrs)
    out = out.assign(
        backscatter_db_raw=backscatter_db_raw,
        backscatter_db_corrected=backscatter_db_corrected,
        tvg_gain_db=tvg_gain_db,
    )
    return out


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


def mean_backscatter_vs_sample_index(
    ds: xr.Dataset,
    value_var: str,
    bin_width: int = 50,
    exclude_below_db: float | None = None,
) -> pd.DataFrame:
    """
    Bin samples by raw sample index (not range) and return per-bin stats.

    Useful (a) before the geometric step has run at all, and (b) for
    investigating Known Issue #5's floor-clipped leading run, which is
    positionally fixed in *samples*, not physically tied to range.

    Parameters
    ----------
    ds : xr.Dataset
        Must have a `sample` dimension and `value_var` defined over it
        (plus any other dims — flattened internally).
    value_var : str
    bin_width : int, default 50
        Bin width in raw sample-index units.
    exclude_below_db : float, optional
        See `mean_backscatter_vs_range`.

    Returns
    -------
    pd.DataFrame
        Columns: sample_min, sample_max, mean, median, std, p10, p90,
        count.
    """
    values_da = ds[value_var]
    n_samples = ds.sizes["sample"]
    sample_idx_da = xr.DataArray(np.arange(n_samples), dims=["sample"])
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

    edges = np.arange(0.0, n_samples + bin_width, bin_width)
    rows = _bin_stats(values_flat, idx_flat, edges)
    df = pd.DataFrame(rows)
    return df.rename(columns={"min": "sample_min", "max": "sample_max"})[columns]


def bin_pings_by_altitude(
    ds: xr.Dataset,
    n_bins: int = 3,
    altitude_var: str = "altitude_m",
) -> dict[str, xr.Dataset]:
    """
    Split pings into altitude bins (equal-count, quantile-based).

    Used to test the angular/grazing-angle hypothesis for the TVG
    residual: if the residual's shape (via `mean_backscatter_vs_range`
    run per bin) changes with altitude, that's evidence for a missing
    angular correction; if it's roughly altitude-invariant, that leans
    toward a straightforward absorption mistuning instead.

    Parameters
    ----------
    ds : xr.Dataset
        Must have a `ping` dimension and `altitude_var` defined over it.
    n_bins : int, default 3
    altitude_var : str, default "altitude_m"

    Returns
    -------
    dict[str, xr.Dataset]
        Keys are `"altitude_{lo:.2f}_{hi:.2f}m"`, ordered by increasing
        altitude. Pings with non-finite altitude are excluded from every
        bin. Ping subsets via `.isel(ping=...)`.
    """
    altitude = np.asarray(ds[altitude_var].values, dtype=np.float64)
    valid = np.isfinite(altitude)

    edges = np.percentile(altitude[valid], np.linspace(0, 100, n_bins + 1))
    # Guard against a degenerate (all-equal) altitude column collapsing
    # every edge to the same value, which would leave every ping unbinned.
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1.0

    bin_idx = np.full(altitude.shape, -1, dtype=int)
    bin_idx[valid] = np.clip(np.digitize(altitude[valid], edges) - 1, 0, n_bins - 1)

    result: dict[str, xr.Dataset] = {}
    for i in range(n_bins):
        mask = bin_idx == i
        lo, hi = edges[i], edges[i + 1]
        label = f"altitude_{lo:.2f}_{hi:.2f}m"
        result[label] = ds.isel(ping=mask)
    return result
