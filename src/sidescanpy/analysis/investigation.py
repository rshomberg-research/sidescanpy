"""
Investigation pipeline (Stages 1-5): methodical characterization of an
unknown sidescan data source's units, TVG status, nadir handling, and
usable range.

Five pure-compute stage functions, each returning a `StageResult`
(JSON-serializable metrics + classification + notes, plus a `data`
scratch dict for objects only the `viz.investigation` plotting layer
needs). Built directly on top of this package's own
`tvg_diagnostics` binning/fitting primitives rather than duplicating
them — Stage 3 reuses `mean_backscatter_vs_range`, Stage 4 reuses that
plus `fit_hardware_tvg_curve`.

Orchestration (loading a survey, wiring stages together, saving plots)
lives in `sidescanpy.investigation`, which also imports the plotting
layer — this module stays matplotlib-free, mirroring the analysis/viz
split already established by `tvg_diagnostics.py`/`viz/tvg_diagnostics.py`.

2026 rshomberg@uri.edu
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from sidescanpy.analysis.tvg_diagnostics import fit_hardware_tvg_curve, mean_backscatter_vs_range
from sidescanpy.processing.radiometric import absorption_for_frequency

_MIN_LINEAR = 1e-6


@dataclass
class StageResult:
    """One investigation stage's output.

    `metrics`/`classification`/`notes` are JSON-serializable (see
    `to_json_dict`). `data` is a scratch dict for larger intermediate
    objects (DataFrames, fit results) that only `viz.investigation`
    needs, kept out of the JSON export deliberately.
    """

    stage: str
    metrics: dict[str, Any]
    classification: str | None = None
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "metrics": self.metrics,
            "classification": self.classification,
            "notes": list(self.notes),
        }


# ============================================================
# OUTLIER REJECTION (shared by Stage 3 / Stage 4)
# ============================================================


def reject_range_outliers(
    values: np.ndarray,
    ranges: np.ndarray,
    *,
    bin_width_m: float = 5.0,
    mad_k: float = 3.5,
    min_bin_count: int = 5,
) -> np.ndarray:
    """Robust, per-range-bin outlier rejection: flag any sample more
    than `mad_k` median-absolute-deviations from its own range bin's
    median, and return a copy of `values` with those samples set to
    NaN (same shape as input).

    Purpose-built for Stage 3's noise-floor/usable-range estimate and
    Stage 4's decay fit — both assume a locally-flat, sandy-bottom-like
    population, and both get corrupted by point-target returns
    (boulders, debris, wrecks) and complex-terrain excursions, which
    are exactly what this is meant to strip. **Not** a general-purpose
    despiking filter for display or export — it deliberately discards
    real, physically genuine returns (hard targets included) whenever
    they're not part of the calibration-relevant flat-bottom population,
    which is the opposite of what a display/export pipeline should do
    (see the investigation report's own "why is LF's range so high"
    finding, which argues these same bright returns should usually be
    *kept*, not removed, outside this specific calibration use case).

    Deliberately median/MAD-based, not mean/std-based — robust to the
    same right-skewed backscatter distributions that would otherwise
    corrupt a naive mean/std outlier test with the very outliers it's
    trying to detect.

    Off by default everywhere this is used
    (`reject_outliers=False` on `stage3_usable_range`/`stage4_decay`) —
    an explicit opt-in, not a silent transformation.

    Parameters
    ----------
    values, ranges : np.ndarray
        Same shape, arbitrary dims (flattened internally, reshaped back
        on return).
    bin_width_m : float, default 5.0
        Range-bin width used to compute each sample's local median/MAD.
        Independent of any bin width used downstream for the actual
        aggregation — pass the same value if in doubt.
    mad_k : float, default 3.5
        Reject samples with `|value - bin_median| > mad_k * 1.4826 *
        bin_MAD` (1.4826 is the standard scale factor making MAD a
        consistent estimator of std under normality — used here only
        as a distance unit, not as a claim the data is normal). 3.5 is
        the commonly-cited Iglewicz & Hoaglin "definite outlier"
        threshold — a defensible, literature-backed default, not
        arbitrary. **Was 2.5 originally** ("fairly aggressive, even
        removing targets" per the request that added this function);
        real-data check on `survey-a` found 2.5 rejected
        ~5-6% of all samples per channel — visibly more than a thin
        tail-trim, reaching into the ordinary speckle population, not
        just rare point-target spikes (see the investigation report's
        Stage 1.5 section). 3.5 cuts that to ~2-3%, still a real trim
        but closer to "the statistically unusual tail" than "a
        meaningful slice of the whole distribution." Lower for more
        aggressive rejection, higher for more conservative.
    min_bin_count : int, default 5
        Bins with fewer than this many finite samples are left
        unfiltered — not enough samples for a meaningful MAD.

    Returns
    -------
    np.ndarray
        Copy of `values`, same shape, with rejected samples set to NaN.
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


def detect_first_bottom_return(
    value_arr: np.ndarray,
    *,
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    min_sample: np.ndarray | float | None = None,
    min_sample_buffer: float = 0.0,
    peak_percentile: float = 100.0,
) -> np.ndarray:
    """
    Vectorized, per-(ping, channel) amplitude-based detection of the
    real first bottom return — a genuine signal detector, not a
    geometric computation.

    Exists because `slant_range_m > altitude_m` (the condition
    `processing.geometric.compute_geometric_ranges` uses for
    `ground_range_m` to go finite) is **not** "where the real bottom
    echo begins" — it's just the sample where slant range crosses the
    vehicle's own altitude, i.e. altitude re-expressed in sample units,
    nothing more. A real sidescan beam is mounted at some angle from
    vertical and has a nonzero width; its near (lower) edge is what
    actually determines where the beam first reaches bottom, and that
    edge is never at 0° from vertical — so the real first bottom return
    always lands at a *larger* slant range than the altitude crossing.
    Everything in between isn't weak backscatter, it's simply not
    illuminated at all — part of the same nadir gap as the acquisition
    blanking at the very start of the ping, not a separate physical
    zone. An earlier version of this stage conflated the altitude
    crossing with "the boundary," which a direct real-data check (a
    human visually reading the raw waterfall) caught as substantially
    wrong. Real-data check (`survey-a`, HF, ping 4000): the
    altitude crossing lands at sample 141, but amplitude stays at
    genuine noise-floor level (0.05-0.4 out of a ~22 peak) all the way
    out to sample ~300 — not illuminated, not weak signal. Direct
    sample-by-sample inspection confirmed the actual step to the real
    first bottom return in that ping happens at sample ~325-330 — over
    twice the altitude-crossing sample, and much closer to a human's
    direct visual read of the same waterfall (an independent check that
    caught this discrepancy in the first place) than the altitude
    number was. The angle implied by this real boundary (see Stage
    2.5) is the beam's own near/lower-edge angle — a real, measurable
    hardware property, not a "grazing incidence" explanation.

    Algorithm: smooth with a `smooth_window`-sample rolling mean (raw
    per-sample noise would trigger or miss on individual spikes), then
    find the first point where that smoothed value sustains at least
    `threshold_fraction` of *that ping's own* peak smoothed value for
    at least `min_run` consecutive samples — a single strong sample
    (a target return, a noise spike) shouldn't be mistaken for the
    bottom onset. The threshold is per-ping-and-channel relative (a
    fraction of that specific ping's own peak), not an absolute value,
    since raw amplitude scale varies by channel and survey.

    Parameters
    ----------
    value_arr : np.ndarray
        Shape (ping, channel, sample), linear amplitude (e.g.
        `backscatter_raw`) — not dB. This looks for a multiplicative
        step, which reads clearly in linear amplitude.
    smooth_window : int, default 10
    threshold_fraction : float, default 0.2
        Real-data check found the detected index barely moves across
        0.1-0.3 (320 to 333, for a ping whose true onset was
        independently confirmed at ~325-330 by direct inspection) —
        not a knife-edge parameter.
    min_run : int, default 10
    min_sample : array-like or float, optional
        Per-(ping, channel) (or scalar) hard floor — samples before it
        never count toward the peak reference or the sustained-rise
        search, enforcing the one absolute physical law this problem
        has: **the real first bottom return can never land before the
        vehicle's own altitude** (`stage2_nadir`'s `altitude_sample`).
        Real-data check that found this was necessary, not optional:
        on a real `project-b` Klein survey, without this floor the
        detector returned sample ~36 — *less than* the nadir-noise
        burst's own detected extent (~43) and far short of
        `altitude_sample` (~451) — a physically impossible chain
        (bottom before noise-burst-end before altitude). Residual
        noise past the blanked burst (not every source of per-ping
        noise is caught by `blank_nadir_noise`) or a thin, non-bottom
        noise line (a real `project-b`-specific positioning artifact,
        unrelated to nadir noise) can otherwise pull the "peak"
        reference down and let a short, unsustained bump satisfy the
        relative threshold well before real bottom.
        `stage2_nadir` always passes its own `altitude_sample` here.
    min_sample_buffer : float, default 0.0
        Added on top of `min_sample` before it's used as the floor
        (i.e. the effective floor is `min_sample + min_sample_buffer`,
        not just `min_sample` itself) — a second, independent line of
        defense against residual near-nadir noise that survives
        `blank_nadir_noise`, found necessary on real vendor-format data
        (2026-08-24): even with the `min_sample` floor at
        `altitude_sample` already in place, a handful of bright points
        `blank_nadir_noise` didn't fully zero (right at its own
        boundary) still occasionally satisfied the sustained-rise
        check immediately past that floor, producing a per-ping trace
        that visibly alternates between "stuck near nadir" and "the
        real bottom return" rather than tracking the terrain smoothly
        — `smooth_boundary_across_pings`'s median filter can't recover
        from this because the bad fraction is too large a share of
        pings to be an "outlier." Pushing the search's effective start
        further out, past where this residual noise typically lingers,
        is a direct, low-risk fix (0 preserves every prior caller's
        behavior exactly). No universal default — how large a buffer
        is safe depends on how close a real beam's near edge can
        legitimately be to altitude on the hardware in question; start
        small and check the result against a visual read of the
        waterfall before trusting a specific value.
    peak_percentile : float, default 100.0
        Percentile (not maximum) of the valid smoothed values used as
        the "peak" the relative threshold is computed from. 100.0 is
        exactly `np.nanmax` (the original, still-default behavior) —
        lowering it (e.g. 99.0) makes the peak reference, and so the
        threshold, robust to a single unusually bright residual sample
        dominating it. Complementary to `min_sample_buffer`, not a
        substitute for it: this only helps when the residual noise is
        a small minority of the valid range: a sustained noisy region
        wide enough to itself become the requested percentile isn't
        fixed by this alone.

    Returns
    -------
    np.ndarray
        Shape (ping, channel) — the detected sample index (float, NaN
        where no sustained rise was found). Accurate to within about
        `smooth_window` samples of the true onset, not sample-exact.
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
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice (e.g. too-short ping) -> NaN, not an error here
            peak = np.nanpercentile(roll_for_peak, peak_percentile, axis=-1, keepdims=True)
    threshold = threshold_fraction * peak
    above = roll >= threshold
    if valid_roll is not None:
        above = above & valid_roll
    above = above.astype(np.int32)

    csum2 = np.cumsum(np.insert(above, 0, 0, axis=-1), axis=-1)
    run_sum = csum2[..., min_run:] - csum2[..., :-min_run]
    sustained = run_sum >= min_run

    has_any = sustained.any(axis=-1)
    first = np.argmax(sustained, axis=-1).astype(np.float64)
    first[~has_any] = np.nan
    return first


def smooth_boundary_across_pings(boundary: np.ndarray, *, window: int = 15) -> np.ndarray:
    """
    Per-channel rolling median of a per-ping boundary array (e.g.
    `detect_first_bottom_return`'s output) across the ping axis,
    NaN-aware.

    `detect_first_bottom_return` runs independently per ping — real
    per-ping threshold noise (not every ping's own detection is
    equally clean) produces occasional large, physically implausible
    jumps between adjacent pings, confirmed on real `project-b` data:
    a smooth ~1000-sample decline over dozens of pings punctuated by
    single-ping spikes of 100-3000+ samples that don't correspond to
    any real terrain change (adjacent pings are meters apart, not tens
    of meters). A median (not mean) filter: robust to a handful of
    outlier pings within the window without smearing across a real
    step change the way a running mean would.

    Parameters
    ----------
    boundary : np.ndarray
        Shape (ping, channel).
    window : int, default 15
        Odd window size in pings, centered on each ping (bumped up by
        one if given even). Real-data check: this survey's genuine
        ping-to-ping variation (excluding the large jumps this is
        meant to remove) has a median absolute difference of ~18
        samples between adjacent pings — a window a few times that
        width smooths the jumps without erasing genuine multi-ping
        trends. Not validated as a universal default across hardware;
        revisit if a different survey's real terrain changes faster
        than this window can follow.

    Returns
    -------
    np.ndarray
        Same shape as `boundary`. A ping whose entire window is NaN
        stays NaN.
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
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice -> NaN, not an error here
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
    """
    Vectorized, per-(ping, channel) detection of a high-noise burst
    right at the very start of each ping (near-zero slant range) --
    electronic/transmit-ringdown noise, not a real echo, and not the
    same thing as the beam-geometry null zone `detect_first_bottom_return`
    finds the far edge of.

    EdgeTech (JSF) hardware already zeros this burst out before a file
    ever reaches `sidescanpy` -- it's why JSF's nadir reads as an exact
    `gap` (Stage 2). Klein (SDF) hardware leaves it in unprocessed.
    Confirmed on real `project-b` data: this burst is both bright and
    sustained enough to fool `detect_first_bottom_return` into
    reporting sample 0 as the first bottom return -- it's the noise
    burst, not bottom. `blank_nadir_noise` (this module) uses this
    detector's output to zero the burst out before any other stage
    runs, matching the EdgeTech convention, so `detect_first_bottom_return`
    then sees the same three-zone structure (blanked noise -> quiet
    null zone -> real bottom return) regardless of hardware.

    Mirror image of `detect_first_bottom_return`: instead of finding
    where a weak signal rises to a sustained strong level, finds where
    an initially strong (noisy) signal falls to a sustained low level.

    Parameters
    ----------
    value_arr : np.ndarray
        Shape (ping, channel, sample), linear amplitude.
    search_window : int, default 100
        Only the first `search_window` samples set the noise burst's
        own peak reference -- deliberately narrow so a bright real
        bottom return much further out (which can be brighter than the
        noise burst itself) doesn't inflate the threshold and mask a
        genuinely short burst.
    smooth_window, threshold_fraction, min_run
        Same meaning as `detect_first_bottom_return`, applied in the
        opposite direction (falling below threshold, not rising above
        it).
    margin_samples : int, default 0
        Added to the detected extent before returning (only where a
        boundary was actually found -- a NaN row stays NaN, not
        `NaN + margin`). Found necessary on real vendor-format data
        (2026-08-24): the detected boundary itself is accurate on
        average, but the samples right at its edge aren't always
        cleanly below the noise floor -- a handful of residual bright
        points there survived blanking and were bright/sustained
        enough to fool `detect_first_bottom_return` into locking onto
        them instead of continuing out to the real bottom return. 0
        preserves every prior caller's exact behavior (this burst
        already reads as an exact `gap` with 0 margin on both JSF and
        the Klein surveys checked earlier this session -- don't widen
        it there without a reason). Start small and check
        `nadir_zoom.png` for a visibly cleaner blanked region before
        trusting a specific value on new hardware.
    min_extent_samples : float, default 0.0
        Hard floor on the returned extent, applied last (after
        `margin_samples`) -- unlike `margin_samples`, this **does**
        override a NaN row (nothing detected) rather than leaving it
        alone. The two params answer different questions:
        `margin_samples` widens whatever was actually found (a
        detection-quality fix); `min_extent_samples` asserts a known
        hardware fact ("this channel always has at least N samples of
        nadir junk, whether or not the detector finds a sustained drop
        this particular ping") that should apply even when detection
        fails entirely. 0.0 preserves every prior caller's exact
        behavior, including NaN rows staying NaN.

    Returns
    -------
    np.ndarray
        Shape (ping, channel) -- sample index where the noise burst
        ends. 0 where the leading samples are already at/near zero
        (nothing to blank -- the normal EdgeTech case). NaN only for
        pings too short to evaluate, or where the signal never
        sustains a drop (treat conservatively -- caller should not
        blank an unbounded extent) -- unless `min_extent_samples` is
        set, in which case NaN rows are floored to it too.
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
    value_vars: tuple[str, ...] = ("backscatter_raw", "backscatter_intensity"),
    detector_value_var: str = "backscatter_raw",
    search_window: int = 100,
    smooth_window: int = 5,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    margin_samples: int = 0,
    min_extent_samples: float = 0.0,
) -> xr.Dataset:
    """Return a copy of `ds` with each ping's leading nadir-noise burst
    (`detect_nadir_noise_extent`) set to exactly 0.0 in each of
    `value_vars` -- a real processing step meant to match the
    convention EdgeTech hardware already applies before `sidescanpy`
    ever sees a JSF file, not a display-only mask (contrast
    `mask_nadir_gap`, which NaNs for display/export instead). A no-op
    on data where the burst is already zero (its own detected extent
    comes out 0 samples), so safe to always run regardless of format.

    `run_investigation()` calls this early, before Stage 1, so every
    downstream stage (nadir classification, first-bottom-return
    detection, beam geometry, usable range, decay fit) sees consistent
    data regardless of whether the source hardware already blanked
    this burst or not.

    Parameters
    ----------
    value_vars : tuple of str, default ("backscatter_raw", "backscatter_intensity")
        Every var present in `ds` gets zeroed identically. Vars not
        present are skipped rather than raising.
    detector_value_var : str, default "backscatter_raw"
        Which variable `detect_nadir_noise_extent` computes the
        boundary from.
    margin_samples : int, default 0
        Passed through to `detect_nadir_noise_extent` -- see that
        function's docstring. 0 preserves existing behavior exactly.
    min_extent_samples : float, default 0.0
        Passed through to `detect_nadir_noise_extent` -- see that
        function's docstring. 0.0 preserves existing behavior exactly,
        including leaving a no-detection row unblanked.

    Returns
    -------
    xr.Dataset
        Copy of `ds`; unmodified if none of `value_vars` are present.
    """
    boundary = detect_nadir_noise_extent(
        np.asarray(ds[detector_value_var].values),
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
            zeroed = _mask_before_sample(out[var].values, boundary, fill_value=0.0)
            out[var] = (out[var].dims, zeroed.astype(dtype))
    return out


def _mask_before_sample(values: np.ndarray, boundary: np.ndarray, *, fill_value: float = np.nan) -> np.ndarray:
    """Return a copy of `values` (..., sample) with every sample before
    `boundary` (..., matching the leading dims) set to `fill_value`
    (NaN by default -- exclusion from stats; `blank_nadir_noise` passes
    0.0 -- real zeroing, matching what EdgeTech hardware already does).
    A NaN entry in `boundary` (no detection for that row) masks nothing
    for that row, rather than masking everything or raising.
    """
    n_samples = values.shape[-1]
    sample_idx = np.arange(n_samples, dtype=np.float64)
    before = sample_idx < boundary[..., None]
    before = np.where(np.isfinite(boundary)[..., None], before, False)
    out = np.asarray(values, dtype=np.float64).copy()
    out[before] = fill_value
    return out


def uniform_exclusion_boundary(per_ping_boundary: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Collapse a per-(ping, channel) boundary (e.g.
    `stage2_nadir()`'s `data["first_bottom_return_sample"]`) into a
    single conservative value, broadcast back to the same shape, for
    use as `stage4_decay`'s `exclude_before_sample`.

    **Use this for Stage 4, not Stage 3** — `run_investigation()`
    passes the raw per-ping array to `stage3_usable_range` directly
    (exact, no downside there — see its docstring) and only collapses
    it to this uniform value for `stage4_decay`, where the mechanism
    below actually matters.

    **Using the raw per-ping array directly for aggregate binning is a
    real bug, found via real-data testing, not a hypothetical
    concern:** `mean_backscatter_vs_range` pools samples from every
    ping into shared range bins. If the exclusion boundary varies
    per-ping, near-range bins end up populated *only* by the subset of
    pings whose strong return happened to start earliest (typically
    the shallowest-altitude pings) — a systematically biased, thinned
    sample, not a representative one. On `survey-a` (HF) this
    produced a spuriously strong near-field bin (mean 6.3 vs. 2.9 in
    the next bin out) that `fit_hardware_tvg_curve`'s max-mean
    `r_ref_m` auto-selection picked up, moving the reference from a
    physically sensible 7.5m to 2.5m and nearly quadrupling the fit
    RMSE (0.22dB -> 0.87dB). A single conservative boundary (this
    function, default: the 90th percentile across all pings/channels)
    applies uniformly, so every bin's contributing pings are the same
    regardless of range — no selection bias. Verified on the same real
    survey to restore `r_ref_m` to its un-excluded value (7.5m /
    12.5m for HF/LF) while still meaningfully excluding the nadir gap.

    Parameters
    ----------
    per_ping_boundary : np.ndarray
        Shape (ping, channel), as returned by `stage2_nadir`.
    percentile : float, default 90.0
        Conservative on purpose — a lower percentile would still leave
        some pings' nadir gaps only partially excluded.

    Returns
    -------
    np.ndarray
        Same shape as `per_ping_boundary`, every finite entry replaced
        by the single `percentile`-th value across all finite entries.
        All-NaN input (no detections anywhere) returns an all-NaN array
        unchanged, rather than raising.
    """
    finite = per_ping_boundary[np.isfinite(per_ping_boundary)]
    if finite.size == 0:
        return np.full_like(per_ping_boundary, np.nan)
    value = float(np.percentile(finite, percentile))
    return np.full_like(per_ping_boundary, value)


def mask_nadir_gap(
    ds: xr.Dataset,
    *,
    value_vars: tuple[str, ...] = ("backscatter_raw", "backscatter_intensity"),
    detector_value_var: str = "backscatter_raw",
    range_var: str = "ground_range_m",
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
    min_sample_buffer: float = 0.0,
    peak_percentile: float = 100.0,
    min_first_bottom_return_sample: float | None = None,
    boundary_smoothing_window: int = 15,
) -> xr.Dataset:
    """Return a copy of `ds` with the full nadir gap (everything before
    `detect_first_bottom_return`'s per-(ping, channel) boundary --
    acquisition blanking *and* the beam-geometry null zone the beam's
    own near edge doesn't illuminate at all, not two separate
    phenomena) set to NaN in each of `value_vars`, for display/export
    -- not curve fitting, which stays exact via
    `stage2_nadir`/`stage3_usable_range`/`stage4_decay`'s own
    `exclude_before_sample` plumbing instead of mutating the dataset
    (see `uniform_exclusion_boundary`'s docstring for why Stage 3 and
    Stage 4 need different granularities of this same boundary
    internally -- this function always uses the exact per-ping one,
    appropriate for a display copy since there's no aggregate binning
    downstream to bias).

    **Not the same thing as `blank_nadir_noise`** -- that's a real
    processing step (zeros a genuine electronic-noise burst, applied
    early so every stage sees consistent data). This is display-only
    (NaN, not zero, and includes the much larger un-illuminated span
    out to the real first bottom return, not just the noise burst at
    the very start of it).

    Exists because the earlier fix (Stage 3/4 no longer *fit* against
    this span) left it still visible in the final waterfall
    exports/plots -- a real gap the user caught directly on the
    `survey-a` progression figure: the recommended
    absorption-only + geometric panel still showed the water-column
    return, even though nothing downstream was fit to it anymore.

    Parameters
    ----------
    value_vars : tuple of str, default ("backscatter_raw", "backscatter_intensity")
        Every var present in `ds` gets masked identically (same
        boundary, both raw and TVG-corrected data occupy the same
        un-illuminated span). Vars not present in `ds` are skipped
        rather than raising, so this works whether or not TVG has been
        applied yet.
    detector_value_var : str, default "backscatter_raw"
        Which variable `detect_first_bottom_return` computes the
        boundary from -- raw amplitude, deliberately, so masking is
        identical regardless of which `value_vars` are present or what
        TVG has already been applied to them.
    range_var : str, default "ground_range_m"
        If present, used to compute the altitude sample and pass it as
        `detect_first_bottom_return`'s `min_sample` floor -- the real
        first bottom return can never land closer than the vehicle's
        own altitude (see that function's docstring for the real bug
        this floor fixes). Skipped with no floor if absent, matching
        `stage2_nadir`'s own graceful degradation.
    min_sample_buffer, peak_percentile, min_first_bottom_return_sample
        Passed straight through to `detect_first_bottom_return` -- see
        that function's docstring. All three default to no-ops (0.0 /
        100.0 / None) that reproduce this function's exact prior
        behavior. Set them to keep this function's masking boundary
        consistent with a `stage2_nadir()`/`run_investigation()` call
        using the same overrides elsewhere on the same data -- both
        calls are deterministic given identical inputs, so passing the
        same values here as there is sufficient for consistency without
        needing to thread a precomputed boundary array between them.
    boundary_smoothing_window : int, default 15
        Passed to `smooth_boundary_across_pings` — see `stage2_nadir`'s
        same-named parameter. `<= 1` disables smoothing.

    Returns
    -------
    xr.Dataset
        Copy of `ds`; unmodified if none of `value_vars` are present.
    """
    altitude_sample = None
    if range_var in ds:
        finite_mask = np.isfinite(np.asarray(ds[range_var].values))
        has_any = finite_mask.any(axis=-1)
        altitude_sample = np.argmax(finite_mask, axis=-1).astype(np.float64)
        altitude_sample[~has_any] = np.nan

    effective_min_sample = altitude_sample
    if min_first_bottom_return_sample is not None:
        if altitude_sample is None:
            effective_min_sample = np.full(
                np.asarray(ds[detector_value_var].values).shape[:-1], float(min_first_bottom_return_sample),
            )
        else:
            effective_min_sample = np.fmax(altitude_sample, min_first_bottom_return_sample)

    boundary = detect_first_bottom_return(
        np.asarray(ds[detector_value_var].values),
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
            out[var] = (out[var].dims, _mask_before_sample(out[var].values, boundary))
    return out


# ============================================================
# STAGE 1 — BASIC CHARACTERIZATION
# ============================================================


def stage1_characterize(ds: xr.Dataset, value_var: str = "backscatter_raw") -> StageResult:
    """Data type, range, and distribution of `value_var`, no interpretation."""
    arr = np.asarray(ds[value_var].values)
    finite = arr[np.isfinite(arr)]
    n_total = int(arr.size)
    n_finite = int(finite.size)

    metrics: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "n_values": n_total,
        "n_finite": n_finite,
        "pct_nan": 100.0 * (n_total - n_finite) / n_total if n_total else float("nan"),
        "min": float(finite.min()) if n_finite else float("nan"),
        "max": float(finite.max()) if n_finite else float("nan"),
        "mean": float(finite.mean()) if n_finite else float("nan"),
        "median": float(np.median(finite)) if n_finite else float("nan"),
        "std": float(finite.std()) if n_finite else float("nan"),
        "pct_zero": 100.0 * float(np.count_nonzero(finite == 0)) / n_finite if n_finite else float("nan"),
        "pct_negative": 100.0 * float(np.count_nonzero(finite < 0)) / n_finite if n_finite else float("nan"),
    }
    notes: list[str] = []
    if n_finite == 0:
        notes.append(f"{value_var!r} has no finite values.")

    return StageResult(stage="stage1_characterization", metrics=metrics, notes=notes)


# ============================================================
# STAGE 1.5 — OUTLIER COMPARISON
# ============================================================


def stage1_5_outlier_comparison(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    range_var: str = "ground_range_m",
    *,
    bin_width_m: float = 5.0,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
) -> StageResult:
    """Re-run Stage 1's characterization on `value_var` before and after
    `reject_range_outliers`, so the effect of Stage 3/4's
    `reject_outliers` option is visible directly on the value
    distribution, not just inferred from a rejected-count number.

    Purely diagnostic/qualitative — doesn't feed into or change
    anything Stage 2-5 compute; a companion view for deciding whether
    `reject_outliers` looks reasonable on a given survey before turning
    it on for Stage 3/4, and for showing what got discarded.

    Parameters
    ----------
    ds, value_var, range_var
        Same as the other stages.
    bin_width_m, outlier_mad_k, outlier_min_bin_count
        Passed straight through to `reject_range_outliers`.
    """
    if range_var not in ds:
        raise ValueError(f"stage1_5_outlier_comparison: {range_var!r} not found — compute geometric ranges first")

    before = stage1_characterize(ds, value_var=value_var)

    filtered_arr = reject_range_outliers(
        ds[value_var].values, ds[range_var].values,
        bin_width_m=bin_width_m, mad_k=outlier_mad_k, min_bin_count=outlier_min_bin_count,
    )
    filtered_var = f"_{value_var}_outliers_removed"
    filtered_ds = ds.assign(**{filtered_var: (ds[value_var].dims, filtered_arr)})
    after = stage1_characterize(filtered_ds, value_var=filtered_var)

    original = np.asarray(ds[value_var].values)
    n_rejected = int(np.sum(np.isfinite(original) & ~np.isfinite(filtered_arr)))
    pct_rejected = 100.0 * n_rejected / original.size if original.size else float("nan")

    metrics = {
        "bin_width_m": bin_width_m,
        "outlier_mad_k": outlier_mad_k,
        "outlier_n_rejected": n_rejected,
        "outlier_pct_rejected": pct_rejected,
        "before": before.metrics,
        "after": after.metrics,
    }
    return StageResult(
        stage="stage1_5_outlier_comparison",
        metrics=metrics,
        data={"filtered_array": filtered_arr},
    )


# ============================================================
# STAGE 2 — NADIR ANALYSIS
# ============================================================


def stage2_nadir(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
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
) -> StageResult:
    """Classify how nadir (sample 0..`nadir_n_samples` of each channel,
    per this schema's near-to-far sample convention — see
    `viz.waterfall`'s "sample" x-axis mode) is represented: gap,
    bright_spike, noisy, or clean.

    Also surfaces **two genuinely different reference points near
    nadir** — an early version of this function called both of them
    "boundaries" and conflated what they mean, which a direct real-data
    check (a human visually reading the raw waterfall) caught as
    substantially wrong:

    - `altitude_sample_*` — **not a boundary of anything, just the
      vehicle's own altitude re-expressed in sample units.** An
      explicit readout of the condition
      `processing.geometric.compute_geometric_ranges` already applies
      (`range_var` goes finite wherever `slant_range_m > altitude_m`).
      A real sidescan beam is mounted at some angle from vertical with
      a nonzero width; its near (lower) edge is never at 0° from
      vertical, so the beam's real footprint never starts this close
      in. Everything between here and `first_bottom_return_sample_*`
      below isn't a separate physical zone — it's simply not
      illuminated at all, the same underlying reason as the nadir gap
      itself.
    - `first_bottom_return_sample_*` — the real first bottom return: a
      genuine amplitude-based detector (`detect_first_bottom_return`,
      see that function's docstring), finding where backscatter
      actually rises to a sustained strong return. **This is the one
      that matters for excluding the un-illuminated span from curve
      fitting**, and its implied angle (Stage 2.5) is the beam's own
      near-edge angle from vertical — a real hardware property, not a
      "grazing incidence" story. Real-data check (`survey-a`,
      HF): the altitude sample landed at ~141, but backscatter stayed
      at noise-floor level (0.05-0.4 out of a ~22 peak) all the way to
      sample ~300 — not illuminated, not weak signal. The actual step
      to the real first bottom return in that ping was independently
      confirmed (direct sample inspection) at ~325-330 — over 2x the
      altitude sample, reported in `altitude_to_bottom_gap_sample_median`
      when the two disagree by more than `2 * nadir_n_samples`.

    Conflating "the nadir gap" (a short, near-zero acquisition
    blanking artifact, ~50 samples, plus on some hardware a noise burst
    right at the very start -- see `blank_nadir_noise`) with either of
    the above (which together, once the beam-geometry null zone is
    included, can run several times longer) was a real gap in earlier
    Stage 2 output — all now distinct metrics, not merged into one
    "boundary."

    A corroborating check is also computed against
    `first_bottom_return_sample`: does `value_var` actually step up
    across that boundary on a representative ping? Should reliably
    show a clear step by construction (the detector is defined by that
    step); a note is added if it doesn't, since that would mean this
    survey's data isn't behaving the way the detector assumes.

    Parameters
    ----------
    range_var : str, default "ground_range_m"
        If present in `ds`, used to compute `altitude_sample_*`. If
        absent (geometric ranges not yet computed), that's skipped with
        a note, but `first_bottom_return_sample` (which doesn't need
        it) is still computed — this function still runs and
        classifies the nadir gap either way, unlike Stage 3/4 which
        require `range_var`.
    water_column_post_window : int, default 20
        Sample window on each side of `first_bottom_return_sample` used
        for the signal-step corroboration check.
    strong_return_smooth_window, strong_return_threshold_fraction,
    strong_return_min_run, strong_return_min_sample_buffer,
    strong_return_peak_percentile
        Passed through to `detect_first_bottom_return` (as
        `min_sample_buffer`/`peak_percentile` for the last two) — see
        that function's docstring for what they mean and why the
        defaults were chosen. The two buffer/percentile params default
        to 0.0/100.0 (no-ops, exact prior behavior) — set them when a
        real-data check (e.g. `first_bottom_return_sample` visibly
        alternating between the altitude floor and the real value
        across pings, not smoothly tracking terrain) shows residual
        near-nadir noise is still winning past `blank_nadir_noise`.
    min_first_bottom_return_sample : float, optional
        An explicit, user-supplied floor on `first_bottom_return_sample`,
        independent of and in addition to the altitude-derived floor
        above -- combined via element-wise `max(altitude_sample,
        min_first_bottom_return_sample)` before being passed to
        `detect_first_bottom_return` as `min_sample`. `None` (default)
        preserves exact prior behavior (altitude-only floor). Exists
        for a case the altitude floor alone can't cover: a survey/
        hardware combination where the real first bottom return is
        known (from direct inspection) to sit well past what altitude
        alone would predict, but per-ping altitude noise or a short
        recorded window makes that per-ping floor too weak on its own
        to keep the detector off residual near-nadir noise. A single
        scalar, not per-ping -- this is a known, roughly-constant
        hardware/survey property being asserted, not a per-ping
        computation.
    boundary_smoothing_window : int, default 15
        Passed to `smooth_boundary_across_pings` as `window` — a
        rolling median across pings applied to
        `first_bottom_return_sample` (real-data check found the raw
        per-ping detection has occasional large, physically implausible
        jumps between adjacent pings; see that function's docstring).
        Set to `1` (or any value `<= 1`) to disable and use the raw
        per-ping detection unsmoothed.
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
        classification = "gap"
    elif np.isfinite(ratio) and ratio >= bright_spike_ratio:
        classification = "bright_spike"
    elif np.isfinite(nadir_std) and np.isfinite(off_std) and off_std > 0 and (nadir_std / off_std) >= noisy_std_ratio:
        classification = "noisy"
    else:
        classification = "clean"

    metrics = {
        "nadir_n_samples": n,
        "nadir_mean": nadir_mean,
        "nadir_std": nadir_std,
        "nadir_pct_zero": nadir_pct_zero,
        "offnadir_mean": off_mean,
        "offnadir_std": off_std,
        "nadir_offnadir_ratio": ratio,
        "thresholds": {
            "bright_spike_ratio": bright_spike_ratio,
            "noisy_std_ratio": noisy_std_ratio,
            "gap_pct_zero_threshold": gap_pct_zero_threshold,
        },
    }
    notes: list[str] = []
    altitude_sample = None  # slant_range == altitude, in sample units -- not a real boundary, see docstring
    first_bottom_sample = None  # amplitude-detected real first bottom return -- the one that matters for exclusion

    if range_var in ds:
        range_arr = np.asarray(ds[range_var].values, dtype=np.float64)  # (ping, channel, sample)
        finite_mask = np.isfinite(range_arr)
        has_any = finite_mask.any(axis=-1)
        first_idx = np.argmax(finite_mask, axis=-1).astype(np.float64)  # first True per (ping, channel)
        first_idx[~has_any] = np.nan
        altitude_sample = first_idx

        finite_first = first_idx[np.isfinite(first_idx)]
        if finite_first.size:
            metrics["altitude_sample_median"] = float(np.median(finite_first))
            metrics["altitude_sample_p10"] = float(np.percentile(finite_first, 10))
            metrics["altitude_sample_p90"] = float(np.percentile(finite_first, 90))
        else:
            note = f"{range_var!r} present but has no finite values — altitude sample not computed."
            if "slant_range_m" in ds and "altitude_m" in ds:
                slant_max = float(np.nanmax(ds["slant_range_m"].values)) if np.isfinite(ds["slant_range_m"].values).any() else float("nan")
                altitude_min = float(np.nanmin(ds["altitude_m"].values)) if np.isfinite(ds["altitude_m"].values).any() else float("nan")
                if np.isfinite(slant_max) and np.isfinite(altitude_min) and slant_max < altitude_min:
                    note += (
                        f" Likely cause: this channel's max recorded slant range ({slant_max:.1f}m) "
                        f"never reaches the vehicle's own altitude ({altitude_min:.1f}m min) at any "
                        f"sample -- the entire recorded window is water column for every ping, the beam "
                        f"never gets far enough out to hit bottom (a short-range acquisition setting on "
                        f"this channel vs. how high the vehicle was actually flying, not a processing bug)."
                    )
            notes.append(note)
    else:
        notes.append(f"{range_var!r} not found — altitude sample not computed (run compute_geometric_ranges first).")

    effective_min_sample = altitude_sample
    if min_first_bottom_return_sample is not None:
        if altitude_sample is None:
            effective_min_sample = np.full(arr.shape[:-1], float(min_first_bottom_return_sample))
        else:
            # np.fmax (not np.maximum): a NaN altitude_sample for one
            # ping/channel means "altitude unknown here," not "no
            # explicit floor either" -- fmax falls back to whichever
            # operand is a real number instead of propagating NaN.
            effective_min_sample = np.fmax(altitude_sample, min_first_bottom_return_sample)

    first_bottom_sample = detect_first_bottom_return(
        arr,
        smooth_window=strong_return_smooth_window,
        threshold_fraction=strong_return_threshold_fraction,
        min_run=strong_return_min_run,
        min_sample=effective_min_sample,  # physical floor: bottom can never be closer than altitude (or the explicit override, whichever is stricter)
        min_sample_buffer=strong_return_min_sample_buffer,
        peak_percentile=strong_return_peak_percentile,
    )
    if boundary_smoothing_window > 1:
        # detect_first_bottom_return runs independently per ping --
        # real per-ping threshold noise produces occasional large,
        # physically implausible jumps between adjacent pings (see
        # smooth_boundary_across_pings' docstring for the real-data
        # numbers). A rolling median across pings removes those without
        # smearing across genuine multi-ping trends the way a mean
        # filter would.
        first_bottom_sample = smooth_boundary_across_pings(first_bottom_sample, window=boundary_smoothing_window)
    finite_bottom = first_bottom_sample[np.isfinite(first_bottom_sample)]
    if finite_bottom.size:
        metrics["first_bottom_return_sample_median"] = float(np.median(finite_bottom))
        metrics["first_bottom_return_sample_p10"] = float(np.percentile(finite_bottom, 10))
        metrics["first_bottom_return_sample_p90"] = float(np.percentile(finite_bottom, 90))

        if altitude_sample is not None:
            both_finite = np.isfinite(altitude_sample) & np.isfinite(first_bottom_sample)
            if both_finite.any():
                gap = (first_bottom_sample - altitude_sample)[both_finite]
                metrics["altitude_to_bottom_gap_sample_median"] = float(np.median(gap))
                if np.median(gap) > 2 * nadir_n_samples:
                    notes.append(
                        f"The altitude sample and the amplitude-detected first bottom return "
                        f"disagree by a median of {np.median(gap):.0f} samples — real, expected "
                        f"physics (the beam's near edge is set at a nonzero angle from vertical, so "
                        f"it never illuminates all the way in to the altitude crossing), not a bug in "
                        f"either detector. Use `first_bottom_return_sample_*` for anything downstream "
                        f"that needs to avoid this un-illuminated span — the altitude sample alone "
                        f"underestimates where usable flat-bottom signal starts."
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
            metrics["signal_step_check_ping"] = rep_ping
            metrics["signal_step_check_channel"] = rep_ch
            metrics["signal_step_pre_mean"] = pre_mean
            metrics["signal_step_post_mean"] = post_mean
            metrics["signal_step_ratio"] = step_ratio
            if np.isfinite(step_ratio) and step_ratio < 1.2:
                notes.append(
                    f"Signal doesn't clearly step up in the {w} samples right at the detected "
                    f"first bottom return (post/pre mean ratio={step_ratio:.2f} on ping "
                    f"{rep_ping}) — unexpected given how this boundary is defined; worth checking "
                    f"this survey's data directly rather than trusting the detector blindly here."
                )
        elif np.isfinite(b_val):
            notes.append(
                f"First bottom return detected at sample {b_val:.0f} on ping {rep_ping} — too "
                f"close to the start of the ping for a {w}-sample pre/post step check (no "
                f"un-illuminated span before it to compare against, which on its own is a "
                f"reasonable finding)."
            )
    else:
        notes.append("No sustained first bottom return detected in any ping — detect_first_bottom_return found nothing.")

    return StageResult(
        stage="stage2_nadir", metrics=metrics, classification=classification, notes=notes,
        data={
            "altitude_sample": altitude_sample,  # slant_range == altitude, not a real boundary
            "first_bottom_return_sample": first_bottom_sample,  # amplitude-detected -- use this one
        },
    )


# ============================================================
# STAGE 2.5 — BEAM GEOMETRY
# ============================================================


def stage2_5_beam_geometry(
    ds: xr.Dataset,
    stage2: StageResult,
    stage3: StageResult,
    value_var: str = "backscatter_raw",
    range_var: str = "across_track_m",
    slant_range_var: str = "slant_range_m",
    *,
    ping_idx: int | None = None,
    near_angle_max_std_deg: float = 15.0,
    min_valid_fraction: float = 0.2,
) -> StageResult:
    """
    Empirically determines the sidescan beam's own near-edge angle,
    far-edge angle, center angle, and width — from two real boundaries
    this pipeline already computes, not by sweeping arbitrary candidate
    angles and asking which one looks closest.

    - **Near edge**: the beam's lower edge — never at 0 deg from
      vertical for a real beam, which is why the un-illuminated span
      described in `stage2_nadir` exists at all. Implied by the real
      first bottom return (`stage2`'s `first_bottom_return_sample`)
      and the vehicle's own altitude:

          near_angle_from_vertical = arccos(altitude_m / slant_range_m_at_first_bottom_return)

      Computed across every ping (not just `ping_idx`) for a robust
      median/std — a single ping's detector noise is a weak basis for
      a survey-wide number.
    - **Far edge**: where the beam's illumination stops mattering in
      practice — `stage3`'s `usable_range_m` (where SNR drops below
      threshold), not a hardware limit:

          far_angle_from_vertical = arctan(usable_range_m / altitude_m)

      A single survey-aggregate number, since `usable_range_m` already
      is one.
    - **Beam center angle** = (near + far) / 2, **beam width** = far -
      near.

    Replaces an earlier version of this stage that swept a fixed set of
    candidate angles (0/15/30/45/60/70/80 deg from vertical) and asked
    "which one best matches" — once both real edges are already
    measured from data this pipeline computes anyway, there's no reason
    to guess at candidates instead of triangulating the real answer
    directly.

    **The near-edge measurement is what `classification` reports
    confidence in** (`"determined"` / `"ambiguous"` / `"undetermined"`)
    — it's the one derived per-ping from real amplitude detections, so
    its consistency across pings is a meaningful confidence signal. The
    far edge is a single number already vetted by Stage 3's own SNR
    logic; if `usable_range_m` is `None` there (a real, documented
    outcome on some hardware — see `stage3_usable_range`'s docstring),
    the far edge/beam width/center angle come back `None` too, with a
    note explaining why, rather than silently omitted.

    Nadir convention note: a survey where nadir reads *bright* rather
    than *gap* (Stage 2) implies a genuinely different near-nadir beam
    geometry — re-run this stage on that survey's own data rather than
    reusing another survey's angle numbers. **Also make sure
    `blank_nadir_noise` has run first** on any hardware that doesn't
    already zero its own nadir noise (Klein/SDF, confirmed on real
    data) — otherwise the near-edge measurement above is corrupted by
    the noise burst itself, not the beam's real near edge, and this
    stage should (and does) report `"undetermined"` rather than a
    confident wrong number.

    Parameters
    ----------
    ds : xr.Dataset
    stage2 : StageResult
        A `stage2_nadir()` result — supplies `first_bottom_return_sample`.
    stage3 : StageResult
        A `stage3_usable_range()` result — supplies `usable_range_m`
        for the far edge.
    value_var : str, default "backscatter_raw"
    range_var : str, default "across_track_m"
        Signed cross-track distance (port negative, starboard
        positive) — the natural shared x-axis between the geometry
        diagram and the ping profile.
    slant_range_var : str, default "slant_range_m"
        Attitude-invariant range, used for the near-angle
        back-calculation.
    ping_idx : int, optional
        Which single ping's near/far ray geometry to report for the
        diagram plot. Defaults to the middle ping, matching Stage 2's
        other single-ping plots — `classification` and the cross-ping
        near-angle stats below are independent of this choice.
    near_angle_max_std_deg : float, default 15.0
        Cross-ping near-angle std-dev above this is "too inconsistent
        to trust" (`ambiguous`), regardless of the median.
    min_valid_fraction : float, default 0.2
        Fraction of (ping, channel) pairs that must yield a valid near
        angle (finite altitude, `slant_range >= altitude`,
        `first_bottom_return_sample` detected) before reporting a
        confident measurement at all — below this, `"undetermined"`.

    Returns
    -------
    StageResult
        `metrics["near_angle_deg_median"]`, `["near_angle_deg_std"]`,
        `["near_angle_valid_fraction"]` — the cross-ping measurement
        `classification` is based on. `metrics["far_angle_deg"]`,
        `["beam_center_angle_deg"]`, `["beam_width_deg"]` — `None` if
        `usable_range_m` is `None`. `metrics["near_ray"]`/`["far_ray"]`
        hold the single-`ping_idx` geometry the diagram plot draws (per
        channel, `None` where undefined).
    """
    for required in (range_var, slant_range_var, "altitude_m"):
        if required not in ds:
            raise ValueError(f"stage2_5_beam_geometry: {required!r} not found in dataset")

    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2

    altitude = float(ds["altitude_m"].isel(ping=ping_idx).values)
    notes: list[str] = []
    if not np.isfinite(altitude) or altitude <= 0:
        notes.append(f"altitude_m is non-finite/non-positive ({altitude}) for ping {ping_idx} — geometry undefined.")
        return StageResult(
            stage="stage2_5_beam_geometry",
            metrics={"ping_idx": ping_idx, "altitude_m": altitude},
            notes=notes,
        )

    first_bottom = stage2.data.get("first_bottom_return_sample")
    near_ray: dict[str, dict] = {}
    for ch, label in ((0, "port"), (1, "starboard")):
        entry = {
            "observed_sample": None, "observed_slant_range_m": None,
            "observed_across_track_m": None, "angle_from_vertical_deg": None,
            "angle_from_horizontal_deg": None,
        }
        b_val = first_bottom[ping_idx, ch] if first_bottom is not None and ping_idx < first_bottom.shape[0] else np.nan
        if np.isfinite(b_val):
            b = int(round(b_val))
            n_samples = ds.sizes["sample"]
            if b < n_samples:
                slant = float(ds[slant_range_var].isel(ping=ping_idx, channel=ch, sample=b).values)
                across = float(ds[range_var].isel(ping=ping_idx, channel=ch, sample=b).values)
                entry["observed_sample"] = b
                entry["observed_slant_range_m"] = slant if np.isfinite(slant) else None
                entry["observed_across_track_m"] = across if np.isfinite(across) else None
                if np.isfinite(slant) and slant >= altitude:
                    cos_angle = np.clip(altitude / slant, -1.0, 1.0)
                    angle_from_vertical = float(np.degrees(np.arccos(cos_angle)))
                    entry["angle_from_vertical_deg"] = angle_from_vertical
                    entry["angle_from_horizontal_deg"] = 90.0 - angle_from_vertical
                    if angle_from_vertical > 88.0:
                        notes.append(
                            f"{label}: near-edge angle ({angle_from_vertical:.1f} deg from vertical) "
                            f"is within 2 deg of horizontal — physically implausible for a real beam "
                            f"edge; more likely altitude_m is wrong for this ping, or the detected "
                            f"first bottom return is itself unreliable, than that the beam's near "
                            f"edge is genuinely almost-horizontal."
                        )
                elif np.isfinite(slant):
                    notes.append(
                        f"{label}: observed slant range ({slant:.2f}m) is less than altitude_m "
                        f"({altitude:.2f}m) at the detected first bottom return — geometrically "
                        f"impossible for any real beam angle, suggesting altitude_m or the detection "
                        f"itself is unreliable for this ping/channel."
                    )
        near_ray[label] = entry

    usable_range_m = stage3.metrics.get("usable_range_m") if stage3 else None
    far_ray: dict[str, dict] = {}
    for ch, label in ((0, "port"), (1, "starboard")):
        entry = {
            "predicted_across_track_m": None, "predicted_slant_range_m": None,
            "angle_from_vertical_deg": None, "angle_from_horizontal_deg": None,
        }
        if usable_range_m is not None and usable_range_m > 0:
            # usable_range_m is a single survey-aggregate magnitude (not
            # per-channel) -- across_track_m's own sign convention (port
            # negative, starboard positive) is applied here so both
            # sides actually get a far ray, not just whichever sign this
            # magnitude happened to carry. slant range is a true
            # distance (hypotenuse), never signed.
            sign = -1.0 if label == "port" else 1.0
            angle_from_vertical = float(np.degrees(np.arctan2(usable_range_m, altitude)))
            entry["predicted_across_track_m"] = sign * float(usable_range_m)
            entry["predicted_slant_range_m"] = float(np.hypot(usable_range_m, altitude))
            entry["angle_from_vertical_deg"] = angle_from_vertical
            entry["angle_from_horizontal_deg"] = 90.0 - angle_from_vertical
        far_ray[label] = entry
    if usable_range_m is None:
        notes.append(
            "stage3's usable_range_m is None (SNR never reached the far-range threshold on this "
            "survey -- a real, documented outcome on some hardware, see stage3_usable_range's "
            "docstring) -- the far edge, beam width, and beam center angle can't be computed."
        )

    # Cross-ping near-angle measurement -- the single ping_idx entry
    # above is for the diagram plot only; the classification the
    # automated pipeline reports is based on every ping, not one, since
    # a single ping's detector noise is a weak basis for a survey-wide
    # measurement.
    near_angle_deg_median = None
    near_angle_deg_std = None
    near_angle_valid_fraction = 0.0
    classification = "undetermined"

    if first_bottom is not None:
        n_p, n_c = first_bottom.shape
        n_samples = ds.sizes["sample"]
        altitude_all = ds["altitude_m"].values.astype(np.float64)  # (ping,)
        altitude_valid = np.isfinite(altitude_all) & (altitude_all > 0)

        boundary_valid = np.isfinite(first_bottom) & altitude_valid[:, None]
        b_idx = np.where(boundary_valid, np.round(first_bottom), 0).astype(int)
        b_idx = np.clip(b_idx, 0, n_samples - 1)

        slant_arr = ds[slant_range_var].values  # (ping, channel, sample)
        p_grid, c_grid = np.meshgrid(np.arange(n_p), np.arange(n_c), indexing="ij")
        slant_at_boundary = slant_arr[p_grid, c_grid, b_idx]  # (ping, channel)

        altitude_bc = np.broadcast_to(altitude_all[:, None], (n_p, n_c))
        geom_valid = boundary_valid & np.isfinite(slant_at_boundary) & (slant_at_boundary >= altitude_bc)

        cos_angle = np.full((n_p, n_c), np.nan)
        np.divide(altitude_bc, slant_at_boundary, out=cos_angle, where=geom_valid)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_from_vertical_all = np.degrees(np.arccos(cos_angle))
        angle_from_vertical_all = np.where(geom_valid, angle_from_vertical_all, np.nan)

        finite_angles = angle_from_vertical_all[np.isfinite(angle_from_vertical_all)]
        near_angle_valid_fraction = float(finite_angles.size / angle_from_vertical_all.size)
        if finite_angles.size > 0:
            near_angle_deg_median = float(np.median(finite_angles))
            near_angle_deg_std = float(np.std(finite_angles))

        if near_angle_deg_median is None or near_angle_valid_fraction < min_valid_fraction:
            classification = "undetermined"
            notes.append(
                f"Only {near_angle_valid_fraction:.0%} of (ping, channel) pairs yielded a valid "
                f"near-edge angle -- too few to report a confident beam-geometry measurement "
                f"(need >= {min_valid_fraction:.0%}). If this is bright-nadir hardware, check "
                f"whether blank_nadir_noise ran first -- an un-blanked noise burst fools "
                f"first_bottom_return_sample into firing at sample~0 on nearly every ping, which "
                f"looks exactly like this."
            )
        elif near_angle_deg_std is not None and near_angle_deg_std <= near_angle_max_std_deg:
            classification = "determined"
            notes.append(
                f"Cross-ping near-edge angle: median {near_angle_deg_median:.1f} deg from vertical "
                f"(std {near_angle_deg_std:.1f} deg, {near_angle_valid_fraction:.0%} of "
                f"pings/channels valid) -- consistent measurement."
            )
        else:
            classification = "ambiguous"
            notes.append(
                f"Cross-ping near-edge angle is inconsistent (median {near_angle_deg_median:.1f} "
                f"deg, std {near_angle_deg_std:.1f} deg across {near_angle_valid_fraction:.0%} "
                f"valid pings/channels) -- not a confident single measurement for this survey."
            )

    far_angle_deg = None
    beam_center_angle_deg = None
    beam_width_deg = None
    if usable_range_m is not None and usable_range_m > 0:
        altitude_finite = ds["altitude_m"].values[np.isfinite(ds["altitude_m"].values)]
        altitude_median = float(np.median(altitude_finite)) if altitude_finite.size else altitude
        far_angle_deg = float(np.degrees(np.arctan2(usable_range_m, altitude_median)))
        if near_angle_deg_median is not None:
            beam_center_angle_deg = (near_angle_deg_median + far_angle_deg) / 2.0
            beam_width_deg = far_angle_deg - near_angle_deg_median

    metrics = {
        "ping_idx": ping_idx,
        "altitude_m": altitude,
        "near_ray": near_ray,
        "far_ray": far_ray,
        "near_angle_deg_median": near_angle_deg_median,
        "near_angle_deg_std": near_angle_deg_std,
        "near_angle_valid_fraction": near_angle_valid_fraction,
        "far_angle_deg": far_angle_deg,
        "beam_center_angle_deg": beam_center_angle_deg,
        "beam_width_deg": beam_width_deg,
        "usable_range_m": usable_range_m,
    }
    return StageResult(stage="stage2_5_beam_geometry", metrics=metrics, classification=classification, notes=notes)


# ============================================================
# STAGE 3 — USABLE RANGE
# ============================================================


def stage3_usable_range(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    range_var: str = "slant_range_m",
    *,
    bin_width_m: float = 2.0,
    snr_threshold: float = 3.0,
    noise_region_fraction: float = 0.1,
    exclude_before_sample: np.ndarray | None = None,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
) -> StageResult:
    """Bin `value_var` by `range_var`, estimate the far-range noise floor
    from the tail `noise_region_fraction` of bins, and find the last
    (by range) bin whose SNR relative to that floor is still >=
    `snr_threshold`.

    `range_var` defaults to `slant_range_m` (2026-09-02, changed from
    `ground_range_m`) — this is the physics-correct choice for anything
    downstream that treats range as a TVG/propagation-loss axis, since
    `apply_tvg()` itself computes its gain curve against slant range,
    not ground range (see `stage4_decay`'s own note on this). Water-
    column exclusion does NOT follow `range_var` — it's always keyed to
    `ground_range_m`'s own NaN pattern (finite wherever `slant_range_m >
    altitude_m`, see `compute_geometric_ranges`), independent of which
    variable `range_var` names, because `slant_range_m` is pure
    time-of-flight and is finite everywhere, including the water column.
    Using `range_var`'s own finiteness for this (the pre-2026-09-02
    behavior, still correct when `range_var="ground_range_m"`) would
    silently stop excluding the water column at all once `range_var` is
    `slant_range_m`. Falls back to `range_var`'s own finiteness only if
    `ground_range_m` isn't present in `ds` at all.

    Parameters
    ----------
    exclude_before_sample : np.ndarray, optional
        Shape (ping, channel) — for each ping/channel, mask samples
        before this sample index to NaN before binning, on top of
        whatever `range_var` already excludes geometrically. Meant to
        be `stage2_nadir()`'s `data["first_bottom_return_sample"]`: the
        geometric water-column mask alone only excludes the earliest,
        weakest part of the near-grazing zone — real data
        (`survey-a`) shows genuine backscatter stays weak
        (near-grazing incidence, not yet a real flat-bottom echo) well
        past that boundary, sometimes 2x+ further out (see
        `stage2_nadir`'s docstring). `None` (default) applies no
        additional exclusion, matching this function's behavior before
        this parameter existed.
    reject_outliers : bool, default False
        If True, run `reject_range_outliers` (per-range-bin MAD-based
        rejection) on `value_var` before binning, so the noise-floor
        estimate reflects a flat-bottom-like population rather than
        being dragged around by point-target returns or complex
        terrain. Off by default — an explicit opt-in, see
        `reject_range_outliers`'s own docstring for why this shouldn't
        be a silent default.
    outlier_mad_k, outlier_min_bin_count
        Passed through to `reject_range_outliers` when
        `reject_outliers=True`; ignored otherwise.
    """
    if range_var not in ds:
        raise ValueError(f"stage3_usable_range: {range_var!r} not found — compute geometric ranges first")

    # Water-column mask always keyed to ground_range_m specifically, not
    # range_var -- see this function's own docstring for why.
    water_column_var = "ground_range_m" if "ground_range_m" in ds else range_var
    value_finite = np.isfinite(np.asarray(ds[value_var].values))
    water_column_finite = np.isfinite(np.asarray(ds[water_column_var].values))
    n_water_column_excluded = int(np.sum(value_finite & ~water_column_finite))

    # Applied explicitly (not left to mean_backscatter_vs_range's own
    # range_var-finiteness check below) so this holds even when
    # range_var is slant_range_m, which has no water-column NaNs of its
    # own to rely on.
    original = ds[value_var]
    working = np.where(water_column_finite, original.values, np.nan)

    n_nadir_gap_excluded = 0
    if exclude_before_sample is not None:
        masked = _mask_before_sample(working, exclude_before_sample)
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

    binned = mean_backscatter_vs_range(ds, value_var=value_var, range_var=range_var, bin_width_m=bin_width_m)

    if binned.empty:
        return StageResult(
            stage="stage3_usable_range",
            metrics={
                "usable_range_m": None,
                "range_var": range_var,
                "water_column_n_excluded": n_water_column_excluded,
                "nadir_gap_n_excluded": n_nadir_gap_excluded,
            },
            notes=["No finite samples to bin — usable range undetermined."],
            data={"binned": binned},
        )

    n_tail = max(1, int(round(len(binned) * noise_region_fraction)))
    tail = binned.iloc[-n_tail:]
    noise_floor = float(tail["median"].median())
    noise_std = float(tail["std"].mean())
    if not np.isfinite(noise_std) or noise_std <= 0:
        fallback = binned["std"].replace(0, np.nan).median()
        noise_std = float(fallback) if np.isfinite(fallback) else float("nan")

    notes: list[str] = []
    if np.isfinite(noise_std) and noise_std > 0:
        snr = (binned["median"] - noise_floor).abs() / noise_std
    else:
        snr = binned["median"] * float("nan")
        notes.append("Noise std could not be estimated (zero/NaN) — SNR undefined.")

    above = binned.index[snr >= snr_threshold]
    if len(above) == 0:
        usable_range_m = None
        notes.append(f"SNR never reached threshold {snr_threshold} — usable range undetermined.")
    else:
        last_idx = int(above.max())
        usable_range_m = float(binned.loc[last_idx, "range_max_m"])

    metrics = {
        "bin_width_m": bin_width_m,
        "snr_threshold": snr_threshold,
        "noise_region_fraction": noise_region_fraction,
        "noise_floor": noise_floor,
        "noise_std": noise_std,
        "usable_range_m": usable_range_m,
        "far_range_median": float(binned["median"].iloc[-1]),
        "max_range_m": float(binned["range_max_m"].iloc[-1]),
        "range_var": range_var,
        "water_column_n_excluded": n_water_column_excluded,
        "nadir_gap_n_excluded": n_nadir_gap_excluded,
        "reject_outliers": reject_outliers,
        "outlier_mad_k": outlier_mad_k if reject_outliers else None,
        "outlier_n_rejected": n_outliers_rejected,
    }
    return StageResult(
        stage="stage3_usable_range",
        metrics=metrics,
        notes=notes,
        data={"binned": binned, "snr": snr},
    )


# ============================================================
# STAGE 4 — DECAY ANALYSIS (TVG DETECTION)
# ============================================================

# fit_hardware_tvg_curve()'s candidate names describe which physics
# term is still PRESENT (uncompensated) in the observed raw curve —
# the opposite sense of this stage's classification vocabulary, which
# describes what's already been REMOVED. E.g. candidate
# "absorption_only" means "only the absorption term is present" i.e.
# spreading has already been removed -> status "spreading_only".
# Getting this backwards silently inverts the whole stage's
# conclusion, so it's spelled out explicitly rather than left as a
# same-named passthrough.
_TVG_CANDIDATE_TO_STATUS = {
    "flat": "full",
    "spreading_only": "absorption_only",
    "absorption_only": "spreading_only",
    "full_physics": "none",
}


def stage4_decay(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    range_var: str = "slant_range_m",
    *,
    already_db: bool = False,
    db_reference_percentile: float = 99.0,
    db_reference_value: float | None = None,
    freq_hz: float | None = None,
    bin_width_m: float = 5.0,
    usable_range_m: float | None = None,
    min_bins: int = 6,
    ambiguous_rel_tol: float = 0.2,
    min_total_count: int = 30,
    exclude_before_sample: np.ndarray | None = None,
    exclude_first_n_bins: int = 0,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
    fit_value_col: str = "mean",
    r_ref_m: float | None = None,
    r_fit_min_m: float | None = None,
    r_ref_search_fraction: float = 0.5,
) -> StageResult:
    """Fit the four `fit_hardware_tvg_curve` candidate models against
    `value_var` vs. `range_var` and classify what TVG (if any) the data
    already carries: none, spreading_only, absorption_only, full, or
    ambiguous (winner not clearly separated from the runner-up).

    `range_var` defaults to `slant_range_m` (2026-09-02, changed from
    `ground_range_m`) — the physics-correct choice, since `apply_tvg()`
    itself computes its TVG gain curve against slant range (real
    acoustic path length), not ground range (the horizontal projection).
    Fitting a physical loss-vs-range model against the wrong one of the
    two is most wrong exactly where they diverge most: near the boundary,
    where altitude is a large fraction of range. Water-column exclusion
    does NOT follow `range_var` — always keyed to `ground_range_m`'s own
    NaN pattern regardless (`slant_range_m` is pure time-of-flight, finite
    everywhere including the water column, so it can't detect the water
    column on its own — see the `water_column_var` note in this
    function's body).

    Classification is ranked by each candidate's **fixed-scale** RMSE
    (`TVGCandidateFit.fixed_scale_rmse_db` — the literal, uncalibrated
    physics model, scale pinned to 1.0), not the free-fit RMSE
    (2026-08-24 change; see `TVGCandidateFit`'s own docstring for the
    full rationale and the real-data case that motivated it). Both
    numbers are still reported per candidate in `metrics["candidates"]`,
    and `metrics["best_free_fit_scale"]`/`["best_free_fit_rmse_db"]`
    carry the free-fit values for the winning candidate specifically —
    `metrics["best_rmse_db"]` itself now means the fixed-scale RMSE
    that actually decided the winner, not the free-fit one.

    dB reference convention
    ------------------------
    Without genuine sensor/system calibration (never available for this
    kind of data — see the investigation's own recommendations), an
    absolute dB value has no physical meaning; only a *reference* choice
    makes one comparable across datasets. This stage's working dB scale
    is therefore self-referenced to the dataset's own signal level —
    `db_reference_percentile` (default 99th) of `value_var`'s finite
    values, not the fixed `20*log10(clip(x, 1e-6))` "dB re 1" convention
    `analysis.tvg_diagnostics._linear_to_db`/`apply_tvg()` use for
    *producing* `backscatter_intensity` (that convention is correct for
    them — it's what actually ends up in the schema — but arbitrary as
    an investigation reference). The choice is recorded in this stage's
    `metrics["db_reference"]` for reproducibility (Tier 2 of the
    reference-value strategy this stage follows).

    This choice provably does not affect any of this function's
    classification output: every metric downstream is either a
    difference of two dB values at different ranges (`observed_loss_db
    = means[i_ref] - means`, inside `fit_hardware_tvg_curve`) or a fit
    residual computed from those differences — a constant dB reference
    shift cancels out of every difference. Changing
    `db_reference_percentile` changes what an intermediate absolute dB
    number *would* mean if reported directly (never done here) but
    leaves `best_candidate`/`best_scale`/`best_rmse_db`/classification
    bit-for-bit identical — see
    `tests/unit/analysis/test_investigation.py::TestStage4Decay::test_classification_invariant_to_db_reference`.

    Parameters
    ----------
    db_reference_value : float, optional
        Use this exact linear value as the dB reference instead of
        computing `db_reference_percentile` from the data — e.g. `1.0`
        reproduces the fixed "dB re 1" convention for a direct,
        side-by-side comparison against the default self-referenced
        scale (both should classify identically; see the module-level
        comparison the investigation report's "Reference comparison"
        section runs). Ignored when `already_db` is True.
    exclude_before_sample : np.ndarray, optional
        Shape (ping, channel) — mask samples before this sample index
        to NaN (in dB space, after conversion) before binning, on top
        of whatever `range_var` already excludes geometrically. Meant
        to be `stage2_nadir()`'s `data["first_bottom_return_sample"]`
        — see `stage3_usable_range`'s identical parameter for why the
        geometric mask alone isn't enough. `None` (default) applies no
        additional exclusion.
    exclude_first_n_bins : int, default 0 (no-op, prior behavior)
        Drop this many *bins* (not samples) from the near end of
        `binned` — the range-binned mean-vs-range curve — before
        `r_ref_m` auto-selection or fitting ever see it. Distinct from
        `exclude_before_sample`, which masks individual *samples*
        before binning happens: this instead trims whole bins of the
        already-aggregated curve, for a real, separate artifact — a
        bright/anomalous return immediately at the first-bottom-return
        boundary (a leading-edge specular spike, common in sidescan)
        that can land inside the near-field search window `r_ref_m`'s
        auto-selection scans (see that parameter's own docstring on
        `fit_hardware_tvg_curve` for the *different*, already-handled
        ring-down artifact that reads anomalously low, not high, right
        at the very start of a ping — this parameter is for the
        opposite-sign case, at the bottom-return edge rather than the
        ping's start). Applied after `bin_width_m` binning, so "1 or 2
        bins" is in units of whatever `bin_width_m` ends up being (post
        any `min_bins` auto-shrink), not meters. Bins are dropped in
        range order (nearest first), regardless of `binned`'s row
        order. `0` (default) reproduces this function's exact prior
        behavior — every existing caller unaffected.
    reject_outliers : bool, default False
        If True, run `reject_range_outliers` on the dB-converted values
        before binning, so the fit targets a flat-bottom-like
        population rather than being pulled around by point-target
        returns or complex terrain. Applied in dB space (after
        conversion, if any) rather than on `value_var` directly, since
        that's the space the fit itself operates in. Off by default —
        see `reject_range_outliers`'s docstring for why this shouldn't
        be a silent default.
    fit_value_col : {"mean", "median"}, default "mean"
        Which of `mean_backscatter_vs_range`'s per-bin aggregate columns
        `fit_hardware_tvg_curve` fits against. **History worth knowing:**
        an earlier real-data check found `reject_outliers=True` made
        RMSE *worse*, and pairing it with `"median"` sometimes flipped
        the classification to `"ambiguous"`. That check turned out to
        be confounded by a separate bug — `exclude_before_sample` (see
        `stage3_usable_range`'s docstring) was, at the time, being fed
        a per-ping-varying boundary straight into aggregate binning,
        which biases near-range bins on its own (see
        `analysis.investigation.uniform_exclusion_boundary`'s
        docstring for the mechanism). **After fixing that, re-running
        the same comparison (`survey-a`, both channels)
        found `reject_outliers=True` actually *improves* RMSE (HF
        0.70->0.36-0.39dB, LF 0.37->0.30-0.38dB depending on
        `fit_value_col`), and classification stayed `spreading_only`
        in every combination tried — no flip to `"ambiguous"` with
        either `"mean"` or `"median"`.** Left here as a caution about
        how one bug can masquerade as a different feature's
        limitation: verify a surprising real-data result isn't
        actually a symptom of something else before writing it down as
        a property of the thing you were testing.
    outlier_mad_k, outlier_min_bin_count
        Passed through to `reject_range_outliers` when
        `reject_outliers=True`; ignored otherwise.
    r_ref_m : float, optional
        Passed straight through to `fit_hardware_tvg_curve` — pins the
        loss reference to this exact range instead of letting it
        auto-select the near-field max-mean bin. Use this when the
        RMSE-best candidate is unstable between runs/files on the same
        survey (a real, observed failure mode on at least one real
        survey this session: two different representative files from
        the same mission picked different winning candidates, one of
        them within ~1% of tripping `ambiguous_rel_tol`) — the
        near-field auto-pick sits close to the still-transitioning edge
        of the illuminated beam, where per-ping noise is highest, not
        because the heuristic is wrong in general but because that
        particular survey's near field is thin. Picking a point further
        into the beam's well-illuminated middle (definitely past the
        nadir gap, comfortably before the usable-range/noise-floor
        edge) is expected to be more stable — not yet confirmed as a
        universal rule, and deliberately not the new default: this
        remains a per-survey judgment call the caller makes by eye
        (`viz.tvg_diagnostics.plot_candidate_fit`), same as every other
        override in this module.
    r_fit_min_m : float, optional
        Passed straight through to `fit_hardware_tvg_curve` — `None`
        (default) reproduces the prior one-sided fit window (starts at
        `r_ref_m`). Set explicitly (e.g. `0.0`) to fit **both sides** of
        `r_ref_m` instead — the natural pairing with pinning `r_ref_m`
        well into the swath (see that parameter's docstring): with the
        one-sided default, moving the reference out that far would
        throw away most of the near-field data for no reason. Combine
        with `exclude_first_n_bins` to drop only the handful of bins
        immediately at the first-bottom-return boundary rather than the
        whole near half of the range.
    r_ref_search_fraction : float, default 0.5
        Passed straight through to `fit_hardware_tvg_curve` — only
        matters when `r_ref_m` is `None` (auto-select). See that
        function's own docstring.
    min_bins : int, default 6
        Minimum number of range bins the fit window (`r_ref_m` to
        `usable_range_m`) must contain — `bin_width_m` is automatically
        reduced (never increased) before binning if the requested width
        would leave fewer than this. **Real-data motivation (2026-08-25):**
        a survey flown very close to the bottom, with `usable_range_m`
        as short as ~15m, left the default `bin_width_m=5.0` with only
        2-3 bins in the fit window — barely enough points to distinguish
        4 competing single-parameter physics models, let alone reliably.
        `bin_width_m=5.0` was originally tuned against surveys with
        `usable_range_m` in the 40-70m range (6-14 bins there
        naturally), where this never mattered; it silently stopped being
        enough once a genuinely short-range survey showed up. The
        adjustment only ever shrinks the bin width (never widens it
        past what was requested), so a survey with plenty of natural
        bins is completely unaffected — verify via
        `metrics["bin_width_m"]` (the width actually used) vs.
        `metrics["bin_width_m_requested"]` (what the caller passed) to
        see whether adaptation happened on a given run. The estimate is
        necessarily approximate: it targets `min_bins` bins across the
        *whole* range up to `usable_range_m`, not the fit window
        specifically, since `r_ref_m`'s own auto-selection (when not
        pinned) isn't known until after binning — a real fit window can
        still end up with somewhat more or fewer than `min_bins`
        depending on where `r_ref_m` lands.
    """
    if range_var not in ds:
        raise ValueError(f"stage4_decay: {range_var!r} not found — compute geometric ranges first")

    # Water-column mask always keyed to ground_range_m specifically, not
    # range_var -- range_var defaults to slant_range_m (2026-09-02,
    # matching apply_tvg()'s own convention), which is pure time-of-flight
    # and finite everywhere including the water column, so it has no NaN
    # pattern of its own to detect the water column with. Falls back to
    # range_var's own finiteness only if ground_range_m isn't present.
    water_column_var = "ground_range_m" if "ground_range_m" in ds else range_var
    water_column_finite = np.isfinite(np.asarray(ds[water_column_var].values))
    n_water_column_excluded = int(np.sum(
        np.isfinite(np.asarray(ds[value_var].values)) & ~water_column_finite
    ))

    # Shrink (never widen) bin_width_m so the fit window has at least
    # min_bins bins -- see min_bins's own docstring for the real-data
    # case (a short usable_range_m survey collapsing to 2-3 bins) this
    # exists to fix. Computed against usable_range_m, not the (not yet
    # known) fit window itself, since r_ref_m may still be auto-selected
    # below.
    bin_width_m_requested = bin_width_m
    if usable_range_m is not None and usable_range_m > 0 and min_bins > 0:
        min_width_for_bins = usable_range_m / min_bins
        if min_width_for_bins < bin_width_m:
            bin_width_m = min_width_for_bins

    freq_hz = float(freq_hz) if freq_hz is not None else float(ds.attrs["freq_hz_start"])

    if already_db:
        db_da = ds[value_var]
        db_reference = {"method": "already_db", "value": None}
    else:
        arr = ds[value_var]
        if db_reference_value is not None:
            x_ref = float(db_reference_value)
            db_reference = {"method": "fixed_value", "value": x_ref}
        else:
            finite = np.asarray(arr.values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            x_ref = float(np.percentile(finite, db_reference_percentile)) if finite.size else float("nan")
            if not np.isfinite(x_ref) or x_ref <= 0:
                x_ref = _MIN_LINEAR  # degenerate (all-zero/empty) fallback
            db_reference = {"method": "percentile", "percentile": db_reference_percentile, "value": x_ref}
        db_da = 20.0 * np.log10((np.abs(arr) + _MIN_LINEAR) / x_ref)

    # Applied explicitly (not left to mean_backscatter_vs_range's own
    # range_var-finiteness check further down) so water-column exclusion
    # holds even when range_var is slant_range_m -- see this function's
    # water_column_var note above.
    db_da = xr.DataArray(
        np.where(water_column_finite, db_da.values, np.nan), dims=db_da.dims, coords=db_da.coords,
    )

    n_nadir_gap_excluded = 0
    if exclude_before_sample is not None:
        db_before = np.asarray(db_da.values)
        db_masked = _mask_before_sample(db_before, exclude_before_sample)
        n_nadir_gap_excluded = int(np.sum(np.isfinite(db_before) & ~np.isfinite(db_masked)))
        db_da = xr.DataArray(db_masked, dims=db_da.dims, coords=db_da.coords)

    n_outliers_rejected = 0
    if reject_outliers:
        filtered = reject_range_outliers(
            db_da.values, ds[range_var].values,
            bin_width_m=bin_width_m, mad_k=outlier_mad_k, min_bin_count=outlier_min_bin_count,
        )
        n_outliers_rejected = int(np.sum(np.isfinite(np.asarray(db_da.values)) & ~np.isfinite(filtered)))
        db_da = xr.DataArray(filtered, dims=db_da.dims, coords=db_da.coords)

    work = ds[[range_var]].assign(_investigation_db=db_da)
    binned = mean_backscatter_vs_range(
        work, value_var="_investigation_db", range_var=range_var, bin_width_m=bin_width_m
    )

    n_bins_excluded_after_bottom = 0
    if exclude_first_n_bins > 0 and not binned.empty:
        binned = binned.sort_values("range_min_m").reset_index(drop=True)
        n_bins_excluded_after_bottom = min(exclude_first_n_bins, len(binned))
        binned = binned.iloc[n_bins_excluded_after_bottom:].reset_index(drop=True)

    if binned.empty or int(binned["count"].sum()) < min_total_count:
        return StageResult(
            stage="stage4_decay",
            metrics={
                "freq_hz": freq_hz, "db_reference": db_reference,
                "range_var": range_var,
                "water_column_n_excluded": n_water_column_excluded,
                "nadir_gap_n_excluded": n_nadir_gap_excluded,
                "reject_outliers": reject_outliers,
                "outlier_mad_k": outlier_mad_k if reject_outliers else None,
                "outlier_n_rejected": n_outliers_rejected,
                "fit_value_col": fit_value_col,
                "bin_width_m": bin_width_m,
                "bin_width_m_requested": bin_width_m_requested,
                "exclude_first_n_bins": exclude_first_n_bins,
                "n_bins_excluded_after_bottom": n_bins_excluded_after_bottom,
            },
            classification="ambiguous",
            notes=["Not enough binned samples to fit a decay model."],
            data={"binned": binned},
        )

    absorption_db_per_m = absorption_for_frequency(freq_hz)
    fit_result = fit_hardware_tvg_curve(
        binned, absorption_db_per_m=absorption_db_per_m, r_ref_m=r_ref_m, r_fit_min_m=r_fit_min_m,
        r_fit_max_m=usable_range_m, r_ref_search_fraction=r_ref_search_fraction, value_col=fit_value_col,
    )

    # Ranked by fixed_scale_rmse_db (literal, uncalibrated physics
    # model, scale pinned to 1.0), not the free-fit rmse_db -- a
    # 2026-08-24 change. A free scale on one term can chase a low RMSE
    # by absorbing whatever's left over (beam pattern, AGC roll-off,
    # turbidity, anything roughly range-linear), not just genuine
    # absorption -- see TVGCandidateFit's docstring for the real-data
    # case that motivated this (a candidate needing a ~2.4x scale to
    # "win" while another's free-fit scale independently landed within
    # 3% of 1.0 unscaled). Comparing literal models is harder to fool.
    ranked = sorted(fit_result.candidates, key=lambda c: c.fixed_scale_rmse_db)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    notes: list[str] = []
    if second is not None and best.fixed_scale_rmse_db > 0 and (
        second.fixed_scale_rmse_db - best.fixed_scale_rmse_db
    ) / best.fixed_scale_rmse_db < ambiguous_rel_tol:
        classification = "ambiguous"
        notes.append(
            f"Best ({best.name}, RMSE={best.fixed_scale_rmse_db:.2f}dB) not clearly separated from "
            f"runner-up ({second.name}, RMSE={second.fixed_scale_rmse_db:.2f}dB)."
        )
    else:
        classification = _TVG_CANDIDATE_TO_STATUS[best.name]
        if second is not None:
            notes.append(f"Runner-up: {second.name} (RMSE={second.fixed_scale_rmse_db:.2f}dB).")

    if bin_width_m < bin_width_m_requested:
        notes.append(
            f"bin_width_m auto-reduced from {bin_width_m_requested:.2f}m to {bin_width_m:.2f}m "
            f"to keep at least min_bins={min_bins} bins given usable_range_m={usable_range_m:.1f}m."
        )

    if n_bins_excluded_after_bottom > 0:
        notes.append(
            f"Dropped {n_bins_excluded_after_bottom} bin(s) immediately following the first bottom "
            f"return (exclude_first_n_bins={exclude_first_n_bins}) before r_ref_m selection/fitting."
        )

    metrics = {
        "freq_hz": freq_hz,
        "db_reference": db_reference,
        "absorption_db_per_m": absorption_db_per_m,
        "bin_width_m": bin_width_m,
        "bin_width_m_requested": bin_width_m_requested,
        "exclude_first_n_bins": exclude_first_n_bins,
        "n_bins_excluded_after_bottom": n_bins_excluded_after_bottom,
        "r_ref_m": best.r_ref_m,
        "r_ref_m_override": r_ref_m,
        "r_fit_min_m": best.r_fit_min_m,
        "r_fit_min_m_override": r_fit_min_m,
        "r_fit_max_m": best.r_fit_max_m,
        "n_bins_used": best.n_bins_used,
        "best_candidate": best.name,
        "best_scale": best.scale,
        "best_rmse_db": best.fixed_scale_rmse_db,
        "best_free_fit_scale": best.scale,
        "best_free_fit_rmse_db": best.rmse_db,
        "ambiguous_rel_tol": ambiguous_rel_tol,
        "candidates": {
            c.name: {
                "scale": c.scale, "rmse_db": c.rmse_db, "fixed_scale_rmse_db": c.fixed_scale_rmse_db,
            }
            for c in fit_result.candidates
        },
        "range_var": range_var,
        "water_column_n_excluded": n_water_column_excluded,
        "nadir_gap_n_excluded": n_nadir_gap_excluded,
        "reject_outliers": reject_outliers,
        "outlier_mad_k": outlier_mad_k if reject_outliers else None,
        "outlier_n_rejected": n_outliers_rejected,
        "fit_value_col": fit_value_col,
    }
    return StageResult(
        stage="stage4_decay",
        metrics=metrics,
        classification=classification,
        notes=notes,
        data={"fit_result": fit_result, "binned": binned},
    )


# ============================================================
# STAGE 5 — UNIT IDENTIFICATION
# ============================================================


def stage5_units(stage1: StageResult, stage4: StageResult | None = None) -> StageResult:
    """Classify likely units (dB / raw_counts / volts / normalized /
    unknown) from Stage 1's value-range signature, cross-checked
    against Stage 4's decay-fit quality when available.
    """
    m = stage1.metrics
    min_v, max_v = m.get("min"), m.get("max")
    pct_neg = m.get("pct_negative")
    pct_zero = m.get("pct_zero")

    notes: list[str] = []
    if min_v is None or max_v is None or not np.isfinite(min_v) or not np.isfinite(max_v):
        classification = "unknown"
        notes.append("Stage 1 min/max not available or non-finite.")
    elif pct_neg is not None and pct_neg > 50.0 and min_v > -200.0 and max_v <= 5.0:
        classification = "dB"
    elif min_v >= 0.0 and 0.0 < max_v <= 1.0:
        classification = "normalized"
    elif min_v >= 0.0 and max_v > 1.0:
        classification = "raw_counts"
        notes.append(
            "Non-negative, unbounded range is consistent with raw ADC counts or "
            "uncalibrated linear amplitude; not distinguishable from calibrated "
            "volts by value distribution alone."
        )
    else:
        classification = "unknown"

    if stage4 is not None and stage4.classification not in (None, "ambiguous"):
        best_rmse = stage4.metrics.get("best_rmse_db")
        best_candidate = stage4.metrics.get("best_candidate")
        if best_rmse is not None and best_rmse < 3.0:
            notes.append(
                f"Stage 4's decay fit ({best_candidate}, RMSE={best_rmse:.2f}dB) is "
                f"consistent with genuine physical backscatter amplitude, supporting "
                f"the {classification!r} unit call."
            )

    metrics = {"value_range": [min_v, max_v], "pct_negative": pct_neg, "pct_zero": pct_zero}
    return StageResult(stage="stage5_units", metrics=metrics, classification=classification, notes=notes)
