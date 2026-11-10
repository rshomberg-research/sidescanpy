"""
Usable range: where the signal falls into the far-range noise floor.

- :func:`find_noise_knee` — on a level-vs-range profile in dB, the peak,
  the lowest level past it and the noise knee: the first range past the
  peak within ``knee_db`` of that lowest level (:class:`NoiseKnee`).
- :func:`estimate_usable_range` — bins backscatter by range and reports
  where the signal ends: at the noise knee (``method="knee"``, the
  default), or at the last bin whose SNR against the far-range floor
  meets a threshold (``method="snr"``).
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
class NoiseKnee(ResultMixin):
    """Peak, noise floor and noise knee of a level-vs-range profile.

    ``peak_m`` is the highest level in the near half of the profile;
    ``lowest_m`` the lowest level past it; ``knee_m`` the first range past
    the peak whose level is within ``knee_db`` of the lowest. The fit
    window runs from ``peak_m + start_margin_m`` to the knee.
    ``knee_found`` is False when that window would be shorter than
    ``min_window_m`` (the level does not fall to a floor, as on data the
    recorder has already levelled); ``knee_m`` and the window then run to
    the end of the record. ``contrast_db`` is the fall from the peak to
    the lowest level, ``low_contrast`` flags it under the threshold, and
    ``rises_at_far_range`` flags a far-half level above the peak. Ranges
    are bin centers; all fields are ``None`` when the profile has no
    finite level.
    """

    peak_m: float | None = None
    peak_db: float | None = None
    lowest_m: float | None = None
    lowest_db: float | None = None
    knee_m: float | None = None
    knee_found: bool = False
    window_m: tuple[float, float] | None = None
    contrast_db: float | None = None
    low_contrast: bool = False
    rises_at_far_range: bool = False
    knee_db: float | None = None
    start_margin_m: float | None = None
    min_window_m: float | None = None
    notes: list[str] = field(default_factory=list)


def find_noise_knee(
    range_m: np.ndarray,
    level_db: np.ndarray,
    knee_db: float = 3.0,
    *,
    start_margin_m: float = 5.0,
    min_window_m: float = 8.0,
    low_contrast_db: float = 6.0,
    peak_search_fraction: float = 0.5,
    drop_last_bins: int = 2,
) -> NoiseKnee:
    """Find the profile peak, the noise floor and the noise knee.

    The peak is searched in the near ``peak_search_fraction`` of the
    finite profile, so a level that rises again at far range (recorder
    gain on noise) cannot pull it out. The last ``drop_last_bins`` finite
    bins are left out (end-of-record artifacts). Beyond the knee the level
    is noise, which no spreading law describes.

    Parameters
    ----------
    range_m : np.ndarray
        Bin centers (m), increasing.
    level_db : np.ndarray
        Level per bin (dB); NaN for empty bins.
    knee_db : float, default 3.0
        The knee is the first range past the peak within this many dB of
        the lowest level.
    start_margin_m : float, default 5.0
        The fit window starts this far past the peak (inside it the beam
        pattern dominates).
    min_window_m : float, default 8.0
        A window shorter than this counts as no knee: the window then
        runs to the end of the record and ``knee_found`` is False.
    low_contrast_db : float, default 6.0
        ``low_contrast`` is set when the peak is less than this above the
        lowest level.
    peak_search_fraction : float, default 0.5
    drop_last_bins : int, default 2

    Returns
    -------
    NoiseKnee
    """
    range_m = np.asarray(range_m, dtype=np.float64)
    level_db = np.asarray(level_db, dtype=np.float64)
    out = NoiseKnee(knee_db=knee_db, start_margin_m=start_margin_m, min_window_m=min_window_m)
    fin = np.flatnonzero(np.isfinite(level_db))
    if fin.size == 0:
        out.notes.append("Profile has no finite level; no peak or knee.")
        return out
    first = int(fin[0])
    last = int(fin[-1]) - drop_last_bins
    if last <= first:
        last = int(fin[-1])
    half = first + int((last - first) * peak_search_fraction)
    near = level_db[first:half + 1]
    if not np.isfinite(near).any():
        near = level_db[first:last + 1]
    peak = first + int(np.nanargmax(near))
    tail = level_db[peak:last + 1]
    lowest = peak + int(np.nanargmin(tail))
    knee_i = peak + int(np.argmax(tail <= level_db[lowest] + knee_db))
    start = float(range_m[peak] + start_margin_m)
    knee = float(range_m[knee_i])
    knee_found = knee - start >= min_window_m
    if not knee_found:
        knee = float(range_m[last])
        out.notes.append(
            f"No noise knee leaves a window of {min_window_m:g} m past the peak at {range_m[peak]:.1f} m: "
            f"the level does not fall to a floor. Window runs to the end of the record ({knee:.1f} m)."
        )
        if start >= knee:
            start = float(range_m[peak])
            out.notes.append("The record ends within the start margin; the window starts at the peak.")
    out.peak_m, out.peak_db = float(range_m[peak]), float(level_db[peak])
    out.lowest_m, out.lowest_db = float(range_m[lowest]), float(level_db[lowest])
    out.knee_m, out.knee_found = knee, bool(knee_found)
    out.window_m = (start, knee)
    out.contrast_db = float(level_db[peak] - level_db[lowest])
    out.low_contrast = bool(out.contrast_db < low_contrast_db)
    far = level_db[half:last + 1]
    out.rises_at_far_range = bool(np.isfinite(far).any() and np.nanmax(far) > level_db[peak])
    if out.low_contrast:
        out.notes.append(
            f"Low contrast: the level falls only {out.contrast_db:.1f} dB from its peak to its lowest value "
            f"(threshold {low_contrast_db:.0f} dB). The recorder may have levelled it, or there is little "
            "seabed signal."
        )
    if out.rises_at_far_range:
        out.notes.append("The level rises at far range above its near-range peak.")
    return out


@dataclass
class UsableRangeResult(ResultMixin):
    """Usable-range estimate and the binned profile it came from.

    ``method`` is ``"knee"`` or ``"snr"``. With ``"knee"``,
    ``usable_range_m`` is the noise knee of the binned median level in dB
    (``knee`` holds the details; the end of the record when no knee is
    found). With ``"snr"`` it is the far edge of the last bin whose SNR
    meets the threshold, ``None`` when none does. ``binned`` is the
    :func:`~sidescanpy.processing.profiles.backscatter_vs_range` table and
    ``snr`` the per-bin SNR (both serialized as column lists).
    """

    usable_range_m: float | None = None
    method: str | None = None
    knee_db: float | None = None
    knee: NoiseKnee | None = None
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
    method: str = "knee",
    bin_width_m: float = 2.0,
    knee_db: float = 3.0,
    already_db: bool = False,
    snr_threshold: float = 3.0,
    noise_region_fraction: float = 0.1,
    exclude_before_sample: np.ndarray | None = None,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
    channel: int | str | None = None,
) -> UsableRangeResult:
    """Estimate the usable range of ``value_var`` along ``range_var``.

    ``method="knee"`` (default): the per-bin medians are converted to dB
    (``20 log10``, unless ``already_db``) and the usable range is their
    noise knee (:func:`find_noise_knee`): the first range past the
    near-range peak within ``knee_db`` of the lowest level. When the
    level never falls to a floor, the end of the record, with a note.

    ``method="snr"``: the noise floor is the median of the per-bin
    medians over the farthest ``noise_region_fraction`` of bins; noise
    std is the mean of their stds (falling back to the median nonzero bin
    std). SNR is ``|bin median - floor| / noise std``; the usable range is
    the far edge of the last bin with SNR ``>= snr_threshold``, ``None``
    when no bin reaches it (common on data the recorder has levelled).

    Water-column samples are always excluded using ``ground_range_m``'s
    NaN pattern (when present), independent of ``range_var``.

    Parameters
    ----------
    range_var : str, default "slant_range_m"
        Range axis to bin by (slant range matches the TVG model's axis).
    method : {"knee", "snr"}, default "knee"
    bin_width_m : float, default 2.0
    knee_db : float, default 3.0
        ``method="knee"`` only.
    already_db : bool, default False
        ``method="knee"`` only: ``value_var`` is already in dB, so the
        bin medians are used as they are.
    snr_threshold : float, default 3.0
        ``method="snr"`` only.
    noise_region_fraction : float, default 0.1
        ``method="snr"`` only.
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
    if method not in ("knee", "snr"):
        raise ValueError(f"estimate_usable_range: method must be 'knee' or 'snr', got {method!r}")

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
        method=method,
        knee_db=knee_db if method == "knee" else None,
        bin_width_m=bin_width_m,
        snr_threshold=snr_threshold if method == "snr" else None,
        noise_region_fraction=noise_region_fraction if method == "snr" else None,
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

    if method == "knee":
        return _usable_range_from_knee(result, binned, knee_db=knee_db, already_db=already_db)

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


def _usable_range_from_knee(
    result: UsableRangeResult, binned: pd.DataFrame, *, knee_db: float, already_db: bool,
) -> UsableRangeResult:
    """Fill ``result`` from the noise knee of ``binned``'s median level."""
    medians = binned["median"].to_numpy(dtype=np.float64)
    if already_db:
        level = medians
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            level = 20.0 * np.log10(medians)
        level[~np.isfinite(level)] = np.nan
    centers = ((binned["range_min_m"] + binned["range_max_m"]) / 2.0).to_numpy()
    knee = find_noise_knee(centers, level, knee_db)
    result.knee = knee
    result.notes.extend(knee.notes)
    result.max_range_m = float(binned["range_max_m"].iloc[-1])
    result.far_range_median = float(binned["median"].iloc[-1])
    if knee.knee_m is None:
        result.notes.append("No finite level to find a knee in — usable range undetermined.")
        return result
    # In value_var's own units, like the SNR method's floor.
    result.noise_floor = knee.lowest_db if already_db else float(10.0 ** (knee.lowest_db / 20.0))
    result.usable_range_m = knee.knee_m
    return result
