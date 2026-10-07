"""
Usable range: where the signal falls into the far-range noise floor.

:func:`estimate_usable_range` bins backscatter by range, estimates the
noise floor from the farthest bins, and reports the last range bin whose
SNR against that floor still meets a threshold.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.analysis._common import channel_mask, resolve_channel_idx, water_column_var
from sidescanpy.analysis._result import ResultMixin
from sidescanpy.processing.bottom import mask_before_sample
from sidescanpy.processing.outliers import reject_range_outliers
from sidescanpy.processing.profiles import backscatter_vs_range

logger = logging.getLogger(__name__)


@dataclass
class UsableRangeResult(ResultMixin):
    """Usable-range estimate and the binned profile it came from.

    ``usable_range_m`` is ``None`` when SNR never meets the threshold or
    nothing could be binned. ``binned`` is the
    :func:`~sidescanpy.processing.profiles.backscatter_vs_range` table and
    ``snr`` the per-bin SNR (both serialized as column lists).
    """

    usable_range_m: float | None = None
    bin_width_m: float | None = None
    snr_threshold: float | None = None
    noise_region_fraction: float | None = None
    noise_floor: float | None = None
    noise_std: float | None = None
    far_range_median: float | None = None
    max_range_m: float | None = None
    range_var: str | None = None
    channel: int | None = None
    water_column_n_excluded: int = 0
    nadir_gap_n_excluded: int = 0
    reject_outliers: bool = False
    outlier_mad_k: float | None = None
    outlier_n_rejected: int = 0
    notes: list[str] = field(default_factory=list)
    binned: pd.DataFrame | None = None
    snr: pd.Series | None = None


def estimate_usable_range(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    range_var: str = "slant_range_m",
    *,
    bin_width_m: float = 2.0,
    snr_threshold: float = 3.0,
    noise_region_fraction: float = 0.1,
    exclude_before_sample: np.ndarray | None = None,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
    channel: int | str | None = None,
) -> UsableRangeResult:
    """Estimate the usable range of ``value_var`` along ``range_var``.

    The noise floor is the median of the per-bin medians over the
    farthest ``noise_region_fraction`` of bins; noise std is the mean of
    their stds (falling back to the median nonzero bin std). SNR is
    ``|bin median - floor| / noise std``; the usable range is the far
    edge of the last bin with SNR ``>= snr_threshold``.

    Water-column samples are always excluded using ``ground_range_m``'s
    NaN pattern (when present), independent of ``range_var``.

    Parameters
    ----------
    range_var : str, default "slant_range_m"
        Range axis to bin by (slant range matches the TVG model's axis).
    exclude_before_sample : np.ndarray, optional
        ``(ping, channel)`` sample index; samples before it are dropped
        before binning, e.g. a per-ping first-bottom-return boundary.
    reject_outliers : bool, default False
        Run :func:`~sidescanpy.processing.outliers.reject_range_outliers`
        first (with ``outlier_mad_k`` / ``outlier_min_bin_count``), so the
        floor reflects a flat-bottom-like population.
    channel : int or str, optional
        Restrict to one physical side (index, or ``channel_id`` value such
        as ``"port"``). ``None`` pools both sides. Applied by masking.
    """
    if range_var not in ds:
        raise ValueError(f"estimate_usable_range: {range_var!r} not found — compute geometric ranges first")

    channel_idx = resolve_channel_idx(ds, channel, caller="estimate_usable_range") if channel is not None else None

    wc_var = water_column_var(ds, range_var)
    value_finite = np.isfinite(np.asarray(ds[value_var].values))
    water_column_finite = np.isfinite(np.asarray(ds[wc_var].values))
    n_water_column_excluded = int(np.sum(value_finite & ~water_column_finite))

    original = ds[value_var]
    working = np.where(water_column_finite, original.values, np.nan)

    if channel_idx is not None:
        working = np.where(channel_mask(ds, original, channel_idx), working, np.nan)

    n_nadir_gap_excluded = 0
    if exclude_before_sample is not None:
        masked = mask_before_sample(working, exclude_before_sample)
        n_nadir_gap_excluded = int(np.sum(np.isfinite(working) & ~np.isfinite(masked)))
        working = masked

    ds = ds.assign(**{value_var: (original.dims, working)})

    n_outliers_rejected = 0
    if reject_outliers:
        original = ds[value_var]
        filtered = reject_range_outliers(
            original.values, ds[range_var].values,
            bin_width_m=bin_width_m, mad_k=outlier_mad_k, min_bin_count=outlier_min_bin_count,
        )
        n_outliers_rejected = int(np.sum(np.isfinite(original.values) & ~np.isfinite(filtered)))
        ds = ds.assign(**{value_var: (original.dims, filtered)})

    binned = backscatter_vs_range(ds, value_var=value_var, range_var=range_var, bin_width_m=bin_width_m)

    result = UsableRangeResult(
        bin_width_m=bin_width_m,
        snr_threshold=snr_threshold,
        noise_region_fraction=noise_region_fraction,
        range_var=range_var,
        channel=channel_idx,
        water_column_n_excluded=n_water_column_excluded,
        nadir_gap_n_excluded=n_nadir_gap_excluded,
        reject_outliers=reject_outliers,
        outlier_mad_k=outlier_mad_k if reject_outliers else None,
        outlier_n_rejected=n_outliers_rejected,
        binned=binned,
    )

    if binned.empty:
        result.notes.append("No finite samples to bin — usable range undetermined.")
        return result

    n_tail = max(1, int(round(len(binned) * noise_region_fraction)))
    tail = binned.iloc[-n_tail:]
    noise_floor = float(tail["median"].median())
    noise_std = float(tail["std"].mean())
    if not np.isfinite(noise_std) or noise_std <= 0:
        fallback = binned["std"].replace(0, np.nan).median()
        noise_std = float(fallback) if np.isfinite(fallback) else float("nan")

    if np.isfinite(noise_std) and noise_std > 0:
        snr = (binned["median"] - noise_floor).abs() / noise_std
    else:
        snr = binned["median"] * float("nan")
        result.notes.append("Noise std could not be estimated (zero/NaN) — SNR undefined.")

    above = binned.index[snr >= snr_threshold]
    if len(above) == 0:
        usable_range_m = None
        result.notes.append(f"SNR never reached threshold {snr_threshold} — usable range undetermined.")
    else:
        last_idx = int(above.max())
        usable_range_m = float(binned.loc[last_idx, "range_max_m"])

    result.usable_range_m = usable_range_m
    result.noise_floor = noise_floor
    result.noise_std = noise_std
    result.far_range_median = float(binned["median"].iloc[-1])
    result.max_range_m = float(binned["range_max_m"].iloc[-1])
    result.snr = snr
    return result
