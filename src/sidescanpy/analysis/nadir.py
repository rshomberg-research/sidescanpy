"""
Nadir analysis: how nadir is represented, and where the bottom starts.

:func:`analyze_nadir` classifies the first ``nadir_n_samples`` of each
ping (``gap`` / ``bright_spike`` / ``noisy`` / ``clean``) and reports two
distinct reference points per (ping, channel):

- the **altitude sample** — the vehicle altitude in sample units (where
  ``ground_range_m`` first goes finite). Not a boundary of the data.
- the **first bottom return** — where backscatter actually rises to the
  bottom echo (:func:`~sidescanpy.processing.bottom.detect_first_bottom_return`),
  floored at the altitude sample. Everything before it is the nadir
  gap: blanking, noise, and the span the tilted beam never illuminates.

Built on :mod:`sidescanpy.processing.bottom`.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from sidescanpy.analysis._result import ResultMixin, array_field
from sidescanpy.processing.bottom import (
    altitude_sample_index,
    detect_first_bottom_return,
    smooth_boundary_across_pings,
)

logger = logging.getLogger(__name__)


@dataclass
class NadirResult(ResultMixin):
    """Nadir classification and bottom-boundary statistics.

    ``nadir_type`` is ``"gap"``, ``"bright_spike"``, ``"noisy"`` or
    ``"clean"``. Sample-index statistics are ``None`` when not computed.
    The per-(ping, channel) arrays ``altitude_sample`` and
    ``first_bottom_return_sample`` are left out of ``to_dict()``.
    """

    nadir_type: str | None = None
    nadir_n_samples: int | None = None
    nadir_mean: float | None = None
    nadir_std: float | None = None
    nadir_pct_zero: float | None = None
    offnadir_mean: float | None = None
    offnadir_std: float | None = None
    nadir_offnadir_ratio: float | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    altitude_sample_median: float | None = None
    altitude_sample_p10: float | None = None
    altitude_sample_p90: float | None = None
    first_bottom_return_sample_median: float | None = None
    first_bottom_return_sample_p10: float | None = None
    first_bottom_return_sample_p90: float | None = None
    altitude_to_bottom_gap_sample_median: float | None = None
    signal_step_check_ping: int | None = None
    signal_step_check_channel: int | None = None
    signal_step_pre_mean: float | None = None
    signal_step_post_mean: float | None = None
    signal_step_ratio: float | None = None
    notes: list[str] = field(default_factory=list)
    altitude_sample: np.ndarray | None = array_field()
    first_bottom_return_sample: np.ndarray | None = array_field()


def analyze_nadir(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    range_var: str = "ground_range_m",
    *,
    nadir_n_samples: int = 50,
    bright_spike_ratio: float = 1.5,
    noisy_std_ratio: float = 1.5,
    gap_pct_zero_threshold: float = 20.0,
    water_column_post_window: int = 20,
    strong_return_smooth_window: int = 10,
    strong_return_threshold_fraction: float = 0.2,
    strong_return_min_run: int = 10,
    strong_return_min_sample_buffer: float = 0.0,
    strong_return_peak_percentile: float = 100.0,
    min_first_bottom_return_sample: float | None = None,
    boundary_smoothing_window: int = 15,
) -> NadirResult:
    """Classify nadir and locate the altitude sample and first bottom return.

    Classification (first match wins), comparing samples
    ``[0, nadir_n_samples)`` against the rest of each ping:

    - ``gap`` — at least ``gap_pct_zero_threshold`` % of nadir values are 0
    - ``bright_spike`` — nadir/off-nadir mean ratio ``>= bright_spike_ratio``
    - ``noisy`` — nadir/off-nadir std ratio ``>= noisy_std_ratio``
    - ``clean`` — otherwise

    A corroborating step check compares the mean of
    ``water_column_post_window`` samples on either side of the first
    bottom return on the middle ping (channel 0) and adds a note if the
    signal doesn't step up.

    Parameters
    ----------
    range_var : str, default "ground_range_m"
        Used for the altitude sample. If absent, that part is skipped
        with a note; the first bottom return is still detected (without
        the altitude floor).
    strong_return_smooth_window, strong_return_threshold_fraction,
    strong_return_min_run, strong_return_min_sample_buffer,
    strong_return_peak_percentile
        Passed to :func:`~sidescanpy.processing.bottom.detect_first_bottom_return`
        as ``smooth_window`` / ``threshold_fraction`` / ``min_run`` /
        ``min_sample_buffer`` / ``peak_percentile``.
    min_first_bottom_return_sample : float, optional
        Extra scalar floor on the first bottom return, combined with the
        altitude floor by element-wise max.
    boundary_smoothing_window : int, default 15
        :func:`~sidescanpy.processing.bottom.smooth_boundary_across_pings`
        window for the first bottom return; ``<= 1`` disables.
    """
    arr = np.asarray(ds[value_var].values, dtype=np.float64)  # (ping, channel, sample)
    n_samples = arr.shape[-1]
    n = max(1, min(nadir_n_samples, n_samples))

    nadir = arr[..., :n]
    offnadir = arr[..., n:]
    nadir_finite = nadir[np.isfinite(nadir)]
    off_finite = offnadir[np.isfinite(offnadir)]

    nadir_mean = float(nadir_finite.mean()) if nadir_finite.size else float("nan")
    nadir_std = float(nadir_finite.std()) if nadir_finite.size else float("nan")
    nadir_pct_zero = (
        100.0 * float(np.count_nonzero(nadir_finite == 0)) / nadir_finite.size
        if nadir_finite.size
        else float("nan")
    )
    off_mean = float(off_finite.mean()) if off_finite.size else float("nan")
    off_std = float(off_finite.std()) if off_finite.size else float("nan")

    ratio = nadir_mean / off_mean if np.isfinite(off_mean) and off_mean != 0 else float("nan")

    if np.isfinite(nadir_pct_zero) and nadir_pct_zero >= gap_pct_zero_threshold:
        nadir_type = "gap"
    elif np.isfinite(ratio) and ratio >= bright_spike_ratio:
        nadir_type = "bright_spike"
    elif np.isfinite(nadir_std) and np.isfinite(off_std) and off_std > 0 and (nadir_std / off_std) >= noisy_std_ratio:
        nadir_type = "noisy"
    else:
        nadir_type = "clean"

    result = NadirResult(
        nadir_type=nadir_type,
        nadir_n_samples=n,
        nadir_mean=nadir_mean,
        nadir_std=nadir_std,
        nadir_pct_zero=nadir_pct_zero,
        offnadir_mean=off_mean,
        offnadir_std=off_std,
        nadir_offnadir_ratio=ratio,
        thresholds={
            "bright_spike_ratio": bright_spike_ratio,
            "noisy_std_ratio": noisy_std_ratio,
            "gap_pct_zero_threshold": gap_pct_zero_threshold,
        },
    )
    notes = result.notes
    altitude_sample = None

    if range_var in ds:
        altitude_sample = altitude_sample_index(ds, range_var)
        finite_first = altitude_sample[np.isfinite(altitude_sample)]
        if finite_first.size:
            result.altitude_sample_median = float(np.median(finite_first))
            result.altitude_sample_p10 = float(np.percentile(finite_first, 10))
            result.altitude_sample_p90 = float(np.percentile(finite_first, 90))
        else:
            note = f"{range_var!r} present but has no finite values — altitude sample not computed."
            if "slant_range_m" in ds and "altitude_m" in ds:
                slant_max = float(np.nanmax(ds["slant_range_m"].values)) if np.isfinite(ds["slant_range_m"].values).any() else float("nan")
                altitude_min = float(np.nanmin(ds["altitude_m"].values)) if np.isfinite(ds["altitude_m"].values).any() else float("nan")
                if np.isfinite(slant_max) and np.isfinite(altitude_min) and slant_max < altitude_min:
                    note += (
                        f" Likely cause: this channel's max recorded slant range ({slant_max:.1f}m) "
                        f"never reaches the vehicle's own altitude ({altitude_min:.1f}m min) at any "
                        f"sample -- the entire recorded window is water column for every ping (a "
                        f"short-range acquisition setting for this altitude, not a processing bug)."
                    )
            notes.append(note)
    else:
        notes.append(f"{range_var!r} not found — altitude sample not computed (run compute_geometric_ranges first).")

    effective_min_sample = altitude_sample
    if min_first_bottom_return_sample is not None:
        if altitude_sample is None:
            effective_min_sample = np.full(arr.shape[:-1], float(min_first_bottom_return_sample))
        else:
            # fmax, not maximum: a NaN altitude sample means "unknown",
            # so fall back to the explicit floor instead of propagating NaN.
            effective_min_sample = np.fmax(altitude_sample, min_first_bottom_return_sample)

    first_bottom_sample = detect_first_bottom_return(
        arr,
        smooth_window=strong_return_smooth_window,
        threshold_fraction=strong_return_threshold_fraction,
        min_run=strong_return_min_run,
        min_sample=effective_min_sample,
        min_sample_buffer=strong_return_min_sample_buffer,
        peak_percentile=strong_return_peak_percentile,
    )
    if boundary_smoothing_window > 1:
        first_bottom_sample = smooth_boundary_across_pings(first_bottom_sample, window=boundary_smoothing_window)
    finite_bottom = first_bottom_sample[np.isfinite(first_bottom_sample)]
    if finite_bottom.size:
        result.first_bottom_return_sample_median = float(np.median(finite_bottom))
        result.first_bottom_return_sample_p10 = float(np.percentile(finite_bottom, 10))
        result.first_bottom_return_sample_p90 = float(np.percentile(finite_bottom, 90))

        if altitude_sample is not None:
            both_finite = np.isfinite(altitude_sample) & np.isfinite(first_bottom_sample)
            if both_finite.any():
                gap = (first_bottom_sample - altitude_sample)[both_finite]
                result.altitude_to_bottom_gap_sample_median = float(np.median(gap))
                if np.median(gap) > 2 * nadir_n_samples:
                    notes.append(
                        f"The altitude sample and the amplitude-detected first bottom return "
                        f"disagree by a median of {np.median(gap):.0f} samples — expected when the "
                        f"beam's near edge is tilted away from vertical, not a detector bug. Use "
                        f"first_bottom_return_sample for anything that must avoid the "
                        f"un-illuminated span."
                    )

        rep_ping = arr.shape[0] // 2
        rep_ch = 0
        b_val = first_bottom_sample[rep_ping, rep_ch]
        w = water_column_post_window
        if np.isfinite(b_val) and b_val >= w:
            b = int(round(b_val))
            pre = arr[rep_ping, rep_ch, b - w:b]
            post = arr[rep_ping, rep_ch, b:b + w]
            pre_mean = float(np.nanmean(pre)) if pre.size else float("nan")
            post_mean = float(np.nanmean(post)) if post.size else float("nan")
            step_ratio = post_mean / pre_mean if np.isfinite(pre_mean) and pre_mean > 0 else float("nan")
            result.signal_step_check_ping = rep_ping
            result.signal_step_check_channel = rep_ch
            result.signal_step_pre_mean = pre_mean
            result.signal_step_post_mean = post_mean
            result.signal_step_ratio = step_ratio
            if np.isfinite(step_ratio) and step_ratio < 1.2:
                notes.append(
                    f"Signal doesn't clearly step up in the {w} samples right at the detected "
                    f"first bottom return (post/pre mean ratio={step_ratio:.2f} on ping "
                    f"{rep_ping}) — check this data directly rather than trusting the detector."
                )
        elif np.isfinite(b_val):
            notes.append(
                f"First bottom return detected at sample {b_val:.0f} on ping {rep_ping} — too "
                f"close to the start of the ping for a {w}-sample pre/post step check."
            )
    else:
        notes.append("No sustained first bottom return detected in any ping — detect_first_bottom_return found nothing.")

    result.altitude_sample = altitude_sample
    result.first_bottom_return_sample = first_bottom_sample
    return result
