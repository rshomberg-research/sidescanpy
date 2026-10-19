"""
Nadir and first-bottom-return detection, and the masks built on them.

Along each ping, a sidescan record runs through three zones: a leading
nadir-noise burst (transmit ring-down; some hardware zeros it before
writing the file, some doesn't), a quiet span the beam never
illuminates, and the real bottom return. The functions here find those
boundaries per (ping, channel) and blank or mask the data before them.

- :func:`detect_nadir_noise_extent` / :func:`blank_nadir_noise` — find
  and zero the leading noise burst (a processing step).
- :func:`detect_first_bottom_return` — amplitude-based onset of the real
  bottom return, optionally floored at the altitude sample.
- :func:`smooth_boundary_across_pings` — rolling median of a per-ping
  boundary.
- :func:`uniform_exclusion_boundary` — collapse a per-ping boundary to
  one conservative value for aggregate binning.
- :func:`mask_nadir_gap` — NaN everything before the first bottom
  return (display/export copy).
- :func:`mask_before_sample`, :func:`altitude_sample_index` — helpers.
- :func:`altitude_from_bottom_track` — fill missing ``altitude_m`` from
  the first bottom return (the ``"bottom_track"`` correction step).

All detectors are vectorized over ``(ping, channel, sample)`` arrays and
return ``(ping, channel)`` float sample indices (NaN = not found).
"""

import logging
import warnings

import numpy as np
import xarray as xr

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_BOTTOM_TRACK, append_history
from sidescanpy.processing.inversion import uncorrected_backscatter
from sidescanpy.processing.sample_axis import sample_index, sample_interval_m

logger = logging.getLogger(__name__)


def detect_first_bottom_return(
    value_arr: np.ndarray,
    *,
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    min_sample: np.ndarray | float | None = None,
    min_sample_buffer: float = 0.0,
    peak_percentile: float = 100.0,
    min_run_fraction: float = 1.0,
) -> np.ndarray:
    """Per-(ping, channel) sample index where the real bottom return starts.

    Smooths each ping with a ``smooth_window``-sample rolling mean, then
    finds the first point where the smoothed value stays at or above
    ``threshold_fraction`` of that ping's own peak for ``min_run``
    consecutive samples. The threshold is relative per ping and channel,
    since amplitude scale varies by channel and survey.

    This is not the altitude crossing (``slant_range_m > altitude_m``):
    a real beam's near edge is tilted away from vertical, so the first
    bottom return lands further out than altitude alone predicts.

    Parameters
    ----------
    value_arr : np.ndarray
        ``(ping, channel, sample)`` linear amplitude (not dB).
    smooth_window : int, default 10
    threshold_fraction : float, default 0.2
        Fraction of the ping's peak smoothed value.
    min_run : int, default 10
        Samples the rise must be sustained for.
    min_sample : array-like or float, optional
        Per-(ping, channel) or scalar floor. Samples before it are
        ignored for both the peak and the onset search. Normally the
        altitude sample: the bottom can't return before it. NaN entries
        mean "no floor" for that row.
    min_sample_buffer : float, default 0.0
        Added to ``min_sample`` before use, to step past residual
        near-nadir noise.
    peak_percentile : float, default 100.0
        Percentile of the smoothed values used as the peak reference
        (100 = max). Lower it to stop one bright residual sample from
        setting the threshold.
    min_run_fraction : float, default 1.0
        Fraction of the ``min_run`` samples that must be at or above the
        threshold (the onset sample itself always must be). Below 1 a long
        run tolerates gaps, such as shadows in a rough seabed return.

    Returns
    -------
    np.ndarray
        ``(ping, channel)`` float sample index, NaN where no sustained
        rise was found. Accurate to about ``smooth_window`` samples.
    """
    arr = np.asarray(value_arr, dtype=np.float64)
    n_samples = arr.shape[-1]
    if n_samples <= smooth_window + min_run:
        return np.full(arr.shape[:-1], np.nan)

    csum = np.cumsum(np.insert(arr, 0, 0.0, axis=-1), axis=-1)
    roll = (csum[..., smooth_window:] - csum[..., :-smooth_window]) / smooth_window
    n_roll = roll.shape[-1]

    valid_roll = None
    if min_sample is not None:
        min_sample_arr = np.broadcast_to(np.asarray(min_sample, dtype=np.float64), arr.shape[:-1])
        if min_sample_buffer:
            min_sample_arr = min_sample_arr + min_sample_buffer
        roll_idx = np.arange(n_roll, dtype=np.float64).reshape((1,) * min_sample_arr.ndim + (n_roll,))
        valid_roll = roll_idx >= min_sample_arr[..., None]
        # No floor for a row with an unknown (NaN) min_sample -- don't
        # treat "we don't know the floor" as "nothing is ever valid".
        valid_roll = np.where(np.isfinite(min_sample_arr)[..., None], valid_roll, True)

    roll_for_peak = np.where(valid_roll, roll, np.nan) if valid_roll is not None else roll
    if peak_percentile >= 100.0:
        peak = np.nanmax(roll_for_peak, axis=-1, keepdims=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice -> NaN
            peak = np.nanpercentile(roll_for_peak, peak_percentile, axis=-1, keepdims=True)
    threshold = threshold_fraction * peak
    above = roll >= threshold
    if valid_roll is not None:
        above = above & valid_roll
    above = above.astype(np.int32)

    csum2 = np.cumsum(np.insert(above, 0, 0, axis=-1), axis=-1)
    run_sum = csum2[..., min_run:] - csum2[..., :-min_run]
    need = min_run if min_run_fraction >= 1.0 else max(1, int(np.ceil(min_run_fraction * min_run)))
    sustained = (run_sum >= need) & (above[..., : run_sum.shape[-1]] == 1)

    has_any = sustained.any(axis=-1)
    first = np.argmax(sustained, axis=-1).astype(np.float64)
    first[~has_any] = np.nan
    return first


def smooth_boundary_across_pings(boundary: np.ndarray, *, window: int = 15) -> np.ndarray:
    """NaN-aware rolling median of a ``(ping, channel)`` boundary along pings.

    Per-ping detection is independent, so occasional pings jump far from
    their neighbours. A median (not mean) removes those without smearing
    real multi-ping trends.

    Parameters
    ----------
    boundary : np.ndarray
        ``(ping, channel)``.
    window : int, default 15
        Window in pings, centred; an even value is bumped up by one.

    Returns
    -------
    np.ndarray
        Same shape. A ping whose whole window is NaN stays NaN.
    """
    if window < 1:
        raise ValueError("smooth_boundary_across_pings: window must be >= 1")
    if window % 2 == 0:
        window += 1
    if boundary.shape[0] < 2:
        return boundary.astype(np.float64).copy()

    half = window // 2
    padded = np.pad(boundary.astype(np.float64), ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (ping, channel, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice -> NaN
        return np.nanmedian(windows, axis=-1)


def detect_nadir_noise_extent(
    value_arr: np.ndarray,
    *,
    search_window: int = 100,
    smooth_window: int = 5,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    margin_samples: int = 0,
    min_extent_samples: float = 0.0,
) -> np.ndarray:
    """Per-(ping, channel) sample index where the leading nadir-noise burst ends.

    Mirror image of :func:`detect_first_bottom_return`: finds where an
    initially strong signal falls to a sustained low level. Some
    hardware (e.g. EdgeTech JSF) zeros this burst before writing the
    file; others (e.g. Klein SDF) leave it in, where it can fool the
    first-bottom-return detector.

    Parameters
    ----------
    value_arr : np.ndarray
        ``(ping, channel, sample)`` linear amplitude.
    search_window : int, default 100
        Only the first ``search_window`` samples set the burst's peak
        reference, so a bright bottom return further out can't inflate
        the threshold.
    smooth_window, threshold_fraction, min_run
        As in :func:`detect_first_bottom_return`, applied to a fall
        below the threshold.
    margin_samples : int, default 0
        Added to each detected extent (NaN rows stay NaN).
    min_extent_samples : float, default 0.0
        Hard floor on the result, applied last; unlike
        ``margin_samples`` it also replaces NaN rows.

    Returns
    -------
    np.ndarray
        ``(ping, channel)`` float sample index. 0 where the leading
        samples are already quiet. NaN for pings too short to evaluate
        or with no sustained drop (unless ``min_extent_samples`` is set).
    """
    arr = np.asarray(value_arr, dtype=np.float64)
    n_samples = arr.shape[-1]
    if n_samples <= smooth_window + min_run:
        first = np.full(arr.shape[:-1], np.nan)
        if min_extent_samples:
            first = np.full(arr.shape[:-1], float(min_extent_samples))
        return first

    csum = np.cumsum(np.insert(arr, 0, 0.0, axis=-1), axis=-1)
    roll = (csum[..., smooth_window:] - csum[..., :-smooth_window]) / smooth_window

    window = max(min(search_window, n_samples) - smooth_window, 1)
    peak = np.nanmax(roll[..., :window], axis=-1, keepdims=True)
    threshold = threshold_fraction * peak
    below = (roll <= threshold).astype(np.int32)

    csum2 = np.cumsum(np.insert(below, 0, 0, axis=-1), axis=-1)
    run_sum = csum2[..., min_run:] - csum2[..., :-min_run]
    sustained = run_sum >= min_run

    has_any = sustained.any(axis=-1)
    first = np.argmax(sustained, axis=-1).astype(np.float64)
    first[~has_any] = np.nan
    if margin_samples:
        first = np.where(np.isfinite(first), first + margin_samples, first)
    if min_extent_samples:
        first = np.where(np.isfinite(first), np.maximum(first, min_extent_samples), min_extent_samples)
    return first


def blank_nadir_noise(
    ds: xr.Dataset,
    *,
    value_vars: tuple[str, ...] = ("backscatter_intensity",),
    detector_value_var: str = "backscatter_intensity",
    search_window: int = 100,
    smooth_window: int = 5,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    margin_samples: int = 0,
    min_extent_samples: float = 0.0,
) -> xr.Dataset:
    """Return a copy of ``ds`` with each ping's leading nadir-noise burst set to 0.

    Uses :func:`detect_nadir_noise_extent` on ``detector_value_var`` and
    zeros the same span in every ``value_vars`` entry present in ``ds``
    (others are skipped). A no-op where the burst is already zero.
    Contrast :func:`mask_nadir_gap`, which NaNs the much larger span out
    to the first bottom return for display.

    The remaining parameters are passed to
    :func:`detect_nadir_noise_extent`.
    """
    boundary = detect_nadir_noise_extent(
        _detector_values(ds, detector_value_var),
        search_window=search_window,
        smooth_window=smooth_window,
        threshold_fraction=threshold_fraction,
        min_run=min_run,
        margin_samples=margin_samples,
        min_extent_samples=min_extent_samples,
    )
    out = ds.copy()
    for var in value_vars:
        if var in out.data_vars:
            dtype = out[var].dtype
            zeroed = mask_before_sample(out[var].values, boundary, fill_value=0.0)
            out[var] = (out[var].dims, zeroed.astype(dtype))
    return out


def _detector_values(ds: xr.Dataset, var: str) -> np.ndarray:
    """Values a boundary detector runs on: for ``backscatter_intensity``, the
    uncorrected signal (recorded radiometric steps inverted), so detection
    is identical before and after TVG."""
    if var == "backscatter_intensity":
        return np.asarray(uncorrected_backscatter(ds).values)
    return np.asarray(ds[var].values)


def mask_before_sample(values: np.ndarray, boundary: np.ndarray, *, fill_value: float = np.nan) -> np.ndarray:
    """Copy of ``values`` ``(..., sample)`` with samples before ``boundary`` set to ``fill_value``.

    ``boundary`` matches the leading dims of ``values``. A NaN boundary
    entry masks nothing for that row. Returns float64.
    """
    n_samples = values.shape[-1]
    sample_idx = np.arange(n_samples, dtype=np.float64)
    before = sample_idx < boundary[..., None]
    before = np.where(np.isfinite(boundary)[..., None], before, False)
    out = np.asarray(values, dtype=np.float64).copy()
    out[before] = fill_value
    return out


def altitude_sample_index(ds: xr.Dataset, range_var: str = "ground_range_m") -> np.ndarray:
    """First finite sample of ``range_var`` per (ping, channel), as float.

    ``ground_range_m`` goes finite where slant range first exceeds
    altitude, so this is altitude expressed in sample units. NaN where a
    row has no finite value.
    """
    finite_mask = np.isfinite(np.asarray(ds[range_var].values, dtype=np.float64))
    has_any = finite_mask.any(axis=-1)
    first_idx = np.argmax(finite_mask, axis=-1).astype(np.float64)
    first_idx[~has_any] = np.nan
    return first_idx


def uniform_exclusion_boundary(per_ping_boundary: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Collapse a per-(ping, channel) boundary to one conservative value.

    For aggregate range binning, a per-ping boundary biases the near
    bins: only the pings whose boundary is earliest (usually the
    shallowest, brightest ones) contribute there. A single cutoff keeps
    every bin's contributing pings the same.

    Parameters
    ----------
    per_ping_boundary : np.ndarray
        ``(ping, channel)``, e.g. a first-bottom-return array.
    percentile : float, default 90.0
        Percentile across all finite entries.

    Returns
    -------
    np.ndarray
        Same shape, filled with that one value; all-NaN if the input has
        no finite entries.
    """
    finite = per_ping_boundary[np.isfinite(per_ping_boundary)]
    if finite.size == 0:
        return np.full_like(per_ping_boundary, np.nan)
    value = float(np.percentile(finite, percentile))
    return np.full_like(per_ping_boundary, value)


def mask_nadir_gap(
    ds: xr.Dataset,
    *,
    value_vars: tuple[str, ...] = ("backscatter_intensity",),
    detector_value_var: str = "backscatter_intensity",
    range_var: str = "ground_range_m",
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    min_sample_buffer: float = 0.0,
    peak_percentile: float = 100.0,
    min_first_bottom_return_sample: float | None = None,
    boundary_smoothing_window: int = 15,
) -> xr.Dataset:
    """Return a copy of ``ds`` with the whole nadir gap set to NaN.

    Everything before :func:`detect_first_bottom_return`'s per-ping
    boundary (noise burst plus un-illuminated span) is masked in each
    ``value_vars`` entry present. For display and export copies; uses
    the exact per-ping boundary.

    Parameters
    ----------
    value_vars : tuple of str, default ("backscatter_intensity",)
        Variables to mask; absent ones are skipped.
    detector_value_var : str, default "backscatter_intensity"
        Variable the boundary is detected on (``backscatter_intensity``
        is read uncorrected, so masking doesn't depend on TVG state).
    range_var : str, default "ground_range_m"
        If present, its altitude sample floors the detection (the bottom
        can't return before altitude).
    smooth_window, threshold_fraction, min_run, min_sample_buffer, peak_percentile
        Passed to :func:`detect_first_bottom_return`.
    min_first_bottom_return_sample : float, optional
        Extra scalar floor, combined with the altitude floor by
        element-wise max.
    boundary_smoothing_window : int, default 15
        :func:`smooth_boundary_across_pings` window; ``<= 1`` disables.

    Returns
    -------
    xr.Dataset
        Copy of ``ds``; unmodified if none of ``value_vars`` are present.
    """
    altitude_sample = altitude_sample_index(ds, range_var) if range_var in ds else None

    effective_min_sample = altitude_sample
    if min_first_bottom_return_sample is not None:
        if altitude_sample is None:
            effective_min_sample = np.full(
                _detector_values(ds, detector_value_var).shape[:-1], float(min_first_bottom_return_sample),
            )
        else:
            effective_min_sample = np.fmax(altitude_sample, min_first_bottom_return_sample)

    boundary = detect_first_bottom_return(
        _detector_values(ds, detector_value_var),
        smooth_window=smooth_window,
        threshold_fraction=threshold_fraction,
        min_run=min_run,
        min_sample=effective_min_sample,
        min_sample_buffer=min_sample_buffer,
        peak_percentile=peak_percentile,
    )
    if boundary_smoothing_window > 1:
        boundary = smooth_boundary_across_pings(boundary, window=boundary_smoothing_window)
    out = ds.copy()
    for var in value_vars:
        if var in out.data_vars:
            out[var] = (out[var].dims, mask_before_sample(out[var].values, boundary))
    return out


def _refine_onset(values: np.ndarray, first: np.ndarray, smooth_window: int, min_run: int) -> np.ndarray:
    """Move each detected onset to the first sample reaching half the return level.

    :func:`detect_first_bottom_return` reports where a ``smooth_window``
    rolling mean crosses its threshold, which is up to ``smooth_window``
    samples before the rise. The return level is the mean of the
    ``min_run`` samples after the window; the refined onset is the first
    sample in the window at or above half of it. NaN onsets stay NaN.
    """
    n = values.shape[-1]
    found = np.isfinite(first)
    start = np.nan_to_num(first, nan=0.0).astype(np.int64)[..., None]
    window = np.clip(start + np.arange(smooth_window + 1), 0, n - 1)
    after = np.clip(start + smooth_window + np.arange(max(min_run, 1)), 0, n - 1)
    level = np.take_along_axis(values, after, axis=-1).mean(axis=-1, keepdims=True)
    hit = np.take_along_axis(values, window, axis=-1) >= 0.5 * level
    offset = np.where(hit.any(axis=-1), np.argmax(hit, axis=-1), 0)
    return np.where(found, start[..., 0] + offset, np.nan)


BOTTOM_TRACK_REPLACE = ("missing", "all")
"""Accepted ``replace`` values of :func:`altitude_from_bottom_track`."""


def altitude_from_bottom_track(
    ds: xr.Dataset,
    *,
    replace: str = "missing",
    max_diff_m: float | None = None,
    min_altitude_m: float = 0.5,
    min_return_m: float = 5.0,
    min_return_fraction: float = 0.7,
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    smooth_pings: int = 1,
) -> xr.Dataset:
    """Set ``altitude_m`` from the first bottom return (the ``"bottom_track"`` step).

    Many recordings carry no usable altitude: no altimeter, no bottom lock,
    or a recorder that writes 0. Without it the geometric step cannot
    compute ground range (NaN altitude) or computes it as if the towfish
    sat on the seabed (altitude 0). This step measures altitude from the
    data instead: per ping and channel, the onset of the bottom return
    (:func:`detect_first_bottom_return` on the uncorrected amplitude,
    refined to the first sample reaching half the return level that
    follows it, since the detector's rolling mean fires up to
    ``smooth_window`` samples early), the nearer of the two channels,
    converted to metres of slant range.

    The first return comes from the beam's near edge, which is tilted away
    from vertical, so on sloping or rough seabed the tracked value is the
    slant range to the nearest bottom point, at or slightly above the true
    altitude. Recorded altitude, when valid, is the better value; the
    default only fills pings that lack it.

    Parameters
    ----------
    ds : xr.Dataset
        One frequency channel. Radiometric steps already applied are undone
        for detection only; ``backscatter_intensity`` is not changed.
    replace : {"missing", "all"}, default "missing"
        ``"missing"``: replace pings whose ``altitude_m`` is NaN or <= 0.
        ``"all"``: replace every ping with a detection.
    max_diff_m : float, optional
        With ``"missing"``, also replace pings whose recorded altitude
        differs from the tracked one by more than this (a lost bottom lock
        that still writes a value).
    min_altitude_m : float, default 0.5
        Detection ignores samples nearer than this.
    min_return_m : float, default 5.0
        Slant distance the return must stay above threshold, converted to
        samples and used as the detector's ``min_run`` when longer. A
        transmit burst (some recorders keep it; it can start after a quiet
        lead-in and be as bright as the seabed) lasts a metre or two; the
        seabed return lasts tens of metres.
    min_return_fraction : float, default 0.7
        Fraction of that distance that must be above threshold
        (``min_run_fraction`` of the detector), so shadows and gaps in a
        rough seabed return do not push the onset further out.
    smooth_window, threshold_fraction, min_run
        Passed to :func:`detect_first_bottom_return` (``min_run`` in
        samples, raised to ``min_return_m`` when that is longer).
    smooth_pings : int, default 1
        Rolling median of the tracked altitude over this many pings
        (:func:`smooth_boundary_across_pings`); 1 = none. Smoothing looks
        across pings, so a chunked :func:`~sidescanpy.processing.correct_file`
        run can differ from an unchunked one within ``smooth_pings // 2``
        pings of each chunk edge.

    Returns
    -------
    xr.Dataset
        ``ds`` with ``altitude_m`` replaced where selected and a
        ``"bottom_track"`` history record. Pings with no detection keep
        their recorded value. ``fathometer_reading_m`` is not changed.
    """
    if replace not in BOTTOM_TRACK_REPLACE:
        raise ValueError(f"replace must be one of {BOTTOM_TRACK_REPLACE}, got {replace!r}")
    if max_diff_m is not None and replace != "missing":
        raise ValueError("max_diff_m applies only with replace='missing'")

    interval = sample_interval_m(ds)                       # (ping,)
    samples = sample_index(ds)                             # absolute index per position
    floor = np.searchsorted(samples, min_altitude_m / interval)  # (ping,) array position
    run = max(int(min_run), int(np.ceil(min_return_m / float(np.median(interval)))))
    values = np.nan_to_num(np.asarray(uncorrected_backscatter(ds).values, dtype=np.float64), nan=0.0)
    first = detect_first_bottom_return(
        values,
        smooth_window=smooth_window,
        threshold_fraction=threshold_fraction,
        min_run=run,
        min_sample=np.repeat(floor[:, None].astype(np.float64), values.shape[1], axis=1),
        min_run_fraction=min_return_fraction if run > min_run else 1.0,
    )
    first = _refine_onset(values, first, smooth_window, run)
    # A flat ping (no rise anywhere) "detects" at the floor itself: not a return.
    first[first <= floor[:, None]] = np.nan
    if smooth_pings > 1:
        first = smooth_boundary_across_pings(first, window=smooth_pings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN ping -> NaN
        nearest = np.nanmin(first, axis=1)
    found = np.isfinite(nearest)
    pos = np.clip(np.nan_to_num(nearest, nan=0.0).astype(np.int64), 0, samples.size - 1)
    tracked = np.where(found, samples[pos] * interval, np.nan)

    recorded = ds["altitude_m"].values.astype(np.float64)
    missing = ~np.isfinite(recorded) | (recorded <= 0)
    if replace == "all":
        selected = np.ones_like(missing)
    else:
        selected = missing.copy()
        if max_diff_m is not None:
            with np.errstate(invalid="ignore"):
                selected |= np.abs(recorded - tracked) > max_diff_m
    use = selected & found
    n_unresolved = int(np.sum(selected & ~found))
    logger.debug(
        "bottom_track: altitude from the first return in %d of %d pings (%d selected had none)",
        int(use.sum()), recorded.size, n_unresolved,
    )
    if n_unresolved:
        logger.warning(
            "bottom_track: no bottom return found in %d of %d selected pings; their altitude_m is unchanged",
            n_unresolved, int(selected.sum()),
        )

    altitude = np.where(use, tracked, recorded).astype(ds["altitude_m"].dtype)
    out = ds.assign(altitude_m=(ds["altitude_m"].dims, altitude, ds["altitude_m"].attrs))
    return append_history(
        out,
        STEP_BOTTOM_TRACK,
        {
            "replace": replace,
            "max_diff_m": max_diff_m,
            "min_altitude_m": min_altitude_m,
            "min_return_m": min_return_m,
            "min_return_fraction": min_return_fraction,
            "smooth_window": smooth_window,
            "threshold_fraction": threshold_fraction,
            "min_run": min_run,
            "smooth_pings": smooth_pings,
        },
        function="sidescanpy.processing.bottom.altitude_from_bottom_track",
        version=__version__,
    )
