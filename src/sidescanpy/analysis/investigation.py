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


def detect_first_strong_return(
    value_arr: np.ndarray,
    *,
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
) -> np.ndarray:
    """
    Vectorized, per-(ping, channel) amplitude-based detection of where
    backscatter actually rises to a strong return — a genuine signal
    detector, not a geometric computation.

    Exists because the geometric water-column boundary (`ground_range_m`
    becoming finite, i.e. `slant_range_m > altitude_m`) is **not** the
    same thing as "where the real bottom echo begins," and an earlier
    version of this stage conflated the two. Real-data check
    (`survey-a`, HF, ping 4000): the geometric crossing lands
    at sample 141, but amplitude stays weak (0.05-0.4 out of a ~22
    peak) all the way out to sample ~300 — a real, physically genuine
    near-grazing-incidence return, not the strong signal a "flat sandy
    bottom" curve fit should be built from. Direct sample-by-sample
    inspection confirmed the actual step to strong return in that ping
    happens at sample ~325-330 — over twice the geometric boundary,
    and much closer to a human's direct visual read of the same
    waterfall (an independent check that caught this discrepancy in
    the first place) than the geometric number was.

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
    peak = np.nanmax(roll, axis=-1, keepdims=True)
    threshold = threshold_fraction * peak
    above = (roll >= threshold).astype(np.int32)

    csum2 = np.cumsum(np.insert(above, 0, 0, axis=-1), axis=-1)
    run_sum = csum2[..., min_run:] - csum2[..., :-min_run]
    sustained = run_sum >= min_run

    has_any = sustained.any(axis=-1)
    first = np.argmax(sustained, axis=-1).astype(np.float64)
    first[~has_any] = np.nan
    return first


def _mask_before_sample(values: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Return a copy of `values` (..., sample) with every sample before
    `boundary` (..., matching the leading dims) set to NaN. A NaN entry
    in `boundary` (no detection for that row) masks nothing for that
    row, rather than masking everything or raising.
    """
    n_samples = values.shape[-1]
    sample_idx = np.arange(n_samples, dtype=np.float64)
    before = sample_idx < boundary[..., None]
    before = np.where(np.isfinite(boundary)[..., None], before, False)
    out = np.asarray(values, dtype=np.float64).copy()
    out[before] = np.nan
    return out


def uniform_exclusion_boundary(per_ping_boundary: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Collapse a per-(ping, channel) boundary (e.g.
    `stage2_nadir()`'s `data["first_strong_return_sample"]`) into a
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
    12.5m for HF/LF) while still meaningfully excluding the weak zone.

    Parameters
    ----------
    per_ping_boundary : np.ndarray
        Shape (ping, channel), as returned by `stage2_nadir`.
    percentile : float, default 90.0
        Conservative on purpose — a lower percentile would still leave
        some pings' weak zones only partially excluded.

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


def mask_weak_zone(
    ds: xr.Dataset,
    *,
    value_vars: tuple[str, ...] = ("backscatter_raw", "backscatter_intensity"),
    detector_value_var: str = "backscatter_raw",
    smooth_window: int = 10,
    threshold_fraction: float = 0.2,
    min_run: int = 10,
) -> xr.Dataset:
    """Return a copy of `ds` with the nadir gap + weak near-grazing zone
    (everything before `detect_first_strong_return`'s per-(ping, channel)
    boundary) set to NaN in each of `value_vars`, for display/export --
    not curve fitting, which stays exact via `stage2_nadir`/`stage3_usable_range`/
    `stage4_decay`'s own `exclude_before_sample` plumbing instead of
    mutating the dataset (see `uniform_exclusion_boundary`'s docstring
    for why Stage 3 and Stage 4 need different granularities of this
    same boundary internally -- this function always uses the exact
    per-ping one, appropriate for a display copy since there's no
    aggregate binning downstream to bias).

    Exists because the earlier fix (Stage 3/4 no longer *fit* against
    the weak zone) left it still visible in the final waterfall
    exports/plots -- a real gap the user caught directly on the
    `survey-a` progression figure: the recommended
    absorption-only + geometric panel still showed the water-column
    return, even though nothing downstream was fit to it anymore.

    Parameters
    ----------
    value_vars : tuple of str, default ("backscatter_raw", "backscatter_intensity")
        Every var present in `ds` gets masked identically (same
        boundary, both raw and TVG-corrected data occupy the same
        water column). Vars not present in `ds` are skipped rather
        than raising, so this works whether or not TVG has been applied
        yet.
    detector_value_var : str, default "backscatter_raw"
        Which variable `detect_first_strong_return` computes the
        boundary from -- raw amplitude, deliberately, so masking is
        identical regardless of which `value_vars` are present or what
        TVG has already been applied to them.

    Returns
    -------
    xr.Dataset
        Copy of `ds`; unmodified if none of `value_vars` are present.
    """
    boundary = detect_first_strong_return(
        np.asarray(ds[detector_value_var].values),
        smooth_window=smooth_window,
        threshold_fraction=threshold_fraction,
        min_run=min_run,
    )
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
) -> StageResult:
    """Classify how nadir (sample 0..`nadir_n_samples` of each channel,
    per this schema's near-to-far sample convention — see
    `viz.waterfall`'s "sample" x-axis mode) is represented: gap,
    bright_spike, noisy, or clean.

    Also surfaces **two distinct water-column-related boundaries** —
    an early version of this function conflated them, which a direct
    real-data check (a human visually reading the raw waterfall) caught
    as substantially wrong:

    - `geometric_bottom_contact_sample_*` — **not** a new detection
      algorithm, an explicit readout of the condition
      `processing.geometric.compute_geometric_ranges` already applies
      (`range_var` is NaN wherever `slant_range_m <= altitude_m`).
      This is where the beam mathematically first reaches flat-bottom
      range, *not* where the return is actually strong.
    - `first_strong_return_sample_*` — a genuine amplitude-based
      detector (`detect_first_strong_return`, see that function's
      docstring), finding where backscatter actually rises to a
      sustained strong return. **This is the one that matters for
      excluding the water column from curve fitting.** Real-data check
      (`survey-a`, HF): the geometric boundary landed at
      sample ~141, but backscatter stayed weak (0.05-0.4 out of a ~22
      peak) all the way to sample ~300 — a real, physically genuine
      near-grazing-incidence return, not yet the strong signal a "flat
      sandy bottom" fit should be built from. The actual step to
      strong return in that ping was independently confirmed (direct
      sample inspection) at ~325-330 — over 2x the geometric boundary,
      reported in `geometric_to_strong_gap_sample_median` when the two
      disagree by more than `2 * nadir_n_samples`.

    Conflating "the nadir gap" (a short, near-zero acquisition
    blanking artifact, ~50 samples) with either of the above (which
    can run several times longer) was a real gap in earlier Stage 2
    output — all three are now distinct metrics.

    A corroborating check is also computed against
    `first_strong_return_sample`: does `value_var` actually step up
    across that boundary on a representative ping? Should reliably
    show a clear step by construction (the detector is defined by that
    step); a note is added if it doesn't, since that would mean this
    survey's data isn't behaving the way the detector assumes.

    Parameters
    ----------
    range_var : str, default "ground_range_m"
        If present in `ds`, used to compute the geometric boundary. If
        absent (geometric ranges not yet computed), that boundary is
        skipped with a note, but `first_strong_return_sample` (which
        doesn't need it) is still computed — this function still runs
        and classifies the nadir gap either way, unlike Stage 3/4 which
        require `range_var`.
    water_column_post_window : int, default 20
        Sample window on each side of `first_strong_return_sample` used
        for the signal-step corroboration check.
    strong_return_smooth_window, strong_return_threshold_fraction,
    strong_return_min_run
        Passed through to `detect_first_strong_return` — see that
        function's docstring for what they mean and why the defaults
        were chosen.
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
    first_bottom_sample = None  # geometric (weaker, earlier) boundary — kept for backward compat
    first_strong_sample = None  # amplitude-detected (real bottom echo) boundary — the one that matters for exclusion

    if range_var in ds:
        range_arr = np.asarray(ds[range_var].values, dtype=np.float64)  # (ping, channel, sample)
        finite_mask = np.isfinite(range_arr)
        has_any = finite_mask.any(axis=-1)
        first_idx = np.argmax(finite_mask, axis=-1).astype(np.float64)  # first True per (ping, channel)
        first_idx[~has_any] = np.nan
        first_bottom_sample = first_idx

        finite_first = first_idx[np.isfinite(first_idx)]
        if finite_first.size:
            metrics["geometric_bottom_contact_sample_median"] = float(np.median(finite_first))
            metrics["geometric_bottom_contact_sample_p10"] = float(np.percentile(finite_first, 10))
            metrics["geometric_bottom_contact_sample_p90"] = float(np.percentile(finite_first, 90))
        else:
            note = f"{range_var!r} present but has no finite values — geometric boundary not computed."
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
        notes.append(f"{range_var!r} not found — geometric boundary not computed (run compute_geometric_ranges first).")

    first_strong_sample = detect_first_strong_return(
        arr,
        smooth_window=strong_return_smooth_window,
        threshold_fraction=strong_return_threshold_fraction,
        min_run=strong_return_min_run,
    )
    finite_strong = first_strong_sample[np.isfinite(first_strong_sample)]
    if finite_strong.size:
        metrics["first_strong_return_sample_median"] = float(np.median(finite_strong))
        metrics["first_strong_return_sample_p10"] = float(np.percentile(finite_strong, 10))
        metrics["first_strong_return_sample_p90"] = float(np.percentile(finite_strong, 90))

        if first_bottom_sample is not None:
            both_finite = np.isfinite(first_bottom_sample) & np.isfinite(first_strong_sample)
            if both_finite.any():
                gap = (first_strong_sample - first_bottom_sample)[both_finite]
                metrics["geometric_to_strong_gap_sample_median"] = float(np.median(gap))
                if np.median(gap) > 2 * nadir_n_samples:
                    notes.append(
                        f"The geometric bottom-contact boundary and the amplitude-detected strong-"
                        f"return boundary disagree by a median of {np.median(gap):.0f} samples — "
                        f"real, expected physics (grazing angle just past the geometric crossing is "
                        f"shallow, so backscatter stays weak for a while before ramping up), not a "
                        f"bug in either detector. Use `first_strong_return_sample_*` for anything "
                        f"downstream that needs to avoid this weak near-grazing zone — the geometric "
                        f"boundary alone underestimates where usable flat-bottom signal starts."
                    )

        rep_ping = arr.shape[0] // 2
        rep_ch = 0
        b_val = first_strong_sample[rep_ping, rep_ch]
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
                    f"strong-return boundary (post/pre mean ratio={step_ratio:.2f} on ping "
                    f"{rep_ping}) — unexpected given how this boundary is defined; worth checking "
                    f"this survey's data directly rather than trusting the detector blindly here."
                )
        elif np.isfinite(b_val):
            notes.append(
                f"Strong-return boundary detected at sample {b_val:.0f} on ping {rep_ping} — too "
                f"close to the start of the ping for a {w}-sample pre/post step check (no water "
                f"column before it to compare against, which on its own is a reasonable finding)."
            )
    else:
        notes.append("No sustained strong return detected in any ping — detect_first_strong_return found nothing.")

    return StageResult(
        stage="stage2_nadir", metrics=metrics, classification=classification, notes=notes,
        data={
            "first_bottom_sample": first_bottom_sample,  # geometric boundary — kept for compat
            "first_strong_return_sample": first_strong_sample,  # amplitude boundary — use this one
        },
    )


# ============================================================
# STAGE 2.5 — BEAM GEOMETRY
# ============================================================


def stage2_5_beam_geometry(
    ds: xr.Dataset,
    stage2: StageResult,
    value_var: str = "backscatter_raw",
    range_var: str = "across_track_m",
    slant_range_var: str = "slant_range_m",
    *,
    ping_idx: int | None = None,
    candidate_angles_from_vertical_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 80.0),
    near_vertical_threshold_deg: float = 5.0,
    max_consistent_std_deg: float = 15.0,
    min_valid_fraction: float = 0.2,
) -> StageResult:
    """
    Test a competing explanation for the Stage 2 "weak zone" (the gap
    between the geometric bottom-contact boundary and the empirical
    first-strong-return boundary): **not** near-grazing-incidence
    scattering that gradually strengthens (Stage 2's working
    assumption, which implicitly took the beam as pointing straight
    down at nadir — a 0-degrees-from-vertical candidate, one of several
    tested here), but a sidescan beam mounted at a genuine tilt that
    simply doesn't illuminate anywhere near directly below the towfish
    at all. If that's right, the "weak zone" isn't weak backscatter —
    it's the sonar looking at nothing (or noise) until the beam's own
    footprint reaches the seafloor.

    Purely geometric, flat-bottom, "straight angles for now" (no real
    beam-pattern/directivity modeling, no roll, small-pitch assumed) —
    deliberately simple. For each candidate depression angle (measured
    from vertical/nadir; 0 = straight down), computes where a ray at
    that angle from the towfish would first intersect a flat seafloor
    at the ping's own `altitude_m`:

        predicted_across_track_m = altitude_m * tan(angle_from_vertical)
        predicted_slant_range_m  = altitude_m / cos(angle_from_vertical)

    Also back-calculates the angle **implied** by the actual data: given
    the empirically detected `first_strong_return_sample` (from
    `stage2_nadir`) and that sample's real `slant_range_var` value,

        implied_angle_from_vertical = arccos(altitude_m / observed_slant_range_m)

    If that implied angle lands in a physically ordinary sidescan tilt
    range, that's evidence for the "tilted beam" explanation. If it
    comes out absurd (e.g. within a degree or two of 90, or undefined
    because `observed_slant_range_m < altitude_m`), that's evidence
    against it here — or evidence that `altitude_m` itself is wrong,
    which this function cannot distinguish on its own (no independent
    depth/altitude truth source is used) — see the returned notes.

    **Reports a `classification` for this, computed across every ping
    in `ds` (not just `ping_idx`), not left implicit** — the earlier
    `survey-a` finding (25 deg HF / 18 deg LF, i.e. genuinely
    tilted) is one dataset's answer, not a property of this stage or a
    default assumption; a different survey's hardware/mounting can
    come out the other way, and the automated pipeline needs to say
    which one it found rather than requiring a human to re-read the
    diagnostic plot every time:

    - `"water_column_tilted_beam"` — median implied angle is clearly
      nonzero (>= `near_vertical_threshold_deg`) and consistent across
      pings/channels (std <= `max_consistent_std_deg`). The Stage 2
      "weak zone" is best read as *water column*, not weak bottom
      backscatter — the beam simply isn't illuminating that range at
      all, until its own footprint reaches the seafloor.
    - `"near_grazing_weak_backscatter"` — median implied angle is near
      0 (beam close to straight down). The weak zone is best read as
      genuine near-grazing-incidence weak backscatter, not a distinct
      off-vertical water column — Stage 2's original assumption holds
      here.
    - `"ambiguous"` — implied angle is inconsistent across pings/
      channels (high std) or too close to horizontal to trust; neither
      reading is well supported.
    - `"undetermined"` — too few pings/channels yielded a valid implied
      angle (bad altitude, mostly-impossible geometry) to classify at
      all.

    This is a heuristic over straight-line geometry with no real
    beam-pattern modeling — treat it as a strong prior to report and
    look at, not a substitute for a human checking
    `viz.investigation.plot_stage2_5_beam_geometry` on a genuinely
    unfamiliar dataset.

    Nadir convention note: this survey's nadir reads as a *gap*
    (Stage 2). A survey where nadir instead reads *bright* implies a
    different beam geometry near nadir (e.g. a beam that does
    illuminate close to vertical, or overlapping port/starboard
    coverage there) — the candidate angle set and the resulting
    picture would look different; don't reuse this survey's numbers
    for a differently-behaved one without re-running this stage.

    Parameters
    ----------
    ds : xr.Dataset
    stage2 : StageResult
        A `stage2_nadir()` result — supplies `first_strong_return_sample`.
    value_var : str, default "backscatter_raw"
    range_var : str, default "across_track_m"
        Signed cross-track distance (port negative, starboard
        positive) — the natural shared x-axis between the geometry
        diagram and the ping profile.
    slant_range_var : str, default "slant_range_m"
        Attitude-invariant range, used for the implied-angle
        back-calculation.
    ping_idx : int, optional
        Which single ping's implied angle/geometry to report in
        `metrics["implied"]` for the diagram plot. Defaults to the
        middle ping, matching Stage 2's other single-ping plots — the
        `classification` below is independent of this choice, computed
        across every ping instead.
    candidate_angles_from_vertical_deg : tuple of float
        Straight-line depression angles to test, measured from
        vertical (0 = nadir-pointing, matching Stage 2's original
        assumption; larger = more tilted toward the horizon).
    near_vertical_threshold_deg : float, default 5.0
        Median cross-ping implied angle below this counts as
        "near-vertical" (`near_grazing_weak_backscatter`), at or above
        it counts as a genuine tilt.
    max_consistent_std_deg : float, default 15.0
        Cross-ping implied-angle std-dev above this is "too
        inconsistent to trust" (`ambiguous`), regardless of the median.
    min_valid_fraction : float, default 0.2
        Fraction of (ping, channel) pairs that must yield a valid
        implied angle (finite altitude, `slant_range >= altitude`,
        `first_strong_return_sample` detected) before classifying at
        all — below this, `"undetermined"`.

    Returns
    -------
    StageResult
        `metrics["candidates"]` is a list of dicts (one per candidate
        angle): `angle_from_vertical_deg`, `angle_from_horizontal_deg`,
        `predicted_across_track_m`, `predicted_slant_range_m`.
        `metrics["implied"]` holds the back-calculated angle for
        `ping_idx` only, per channel (`None` where undefined) — for the
        diagram plot. `metrics["implied_angle_deg_median"]`,
        `["implied_angle_deg_std"]`, `["implied_angle_valid_fraction"]`
        are the cross-ping aggregate this stage's `classification` is
        actually based on. `data` carries nothing extra — the plotting
        layer reads `ds`/`stage2` directly.
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

    candidates = []
    for angle_deg in candidate_angles_from_vertical_deg:
        angle_rad = np.deg2rad(angle_deg)
        candidates.append({
            "angle_from_vertical_deg": angle_deg,
            "angle_from_horizontal_deg": 90.0 - angle_deg,
            "predicted_across_track_m": float(altitude * np.tan(angle_rad)),
            "predicted_slant_range_m": float(altitude / np.cos(angle_rad)),
        })

    first_strong = stage2.data.get("first_strong_return_sample")
    implied: dict[str, dict] = {}
    for ch, label in ((0, "port"), (1, "starboard")):
        entry = {
            "observed_sample": None, "observed_slant_range_m": None,
            "observed_across_track_m": None, "angle_from_vertical_deg": None,
            "angle_from_horizontal_deg": None,
        }
        b_val = first_strong[ping_idx, ch] if first_strong is not None and ping_idx < first_strong.shape[0] else np.nan
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
                            f"{label}: implied angle ({angle_from_vertical:.1f} deg from vertical) is "
                            f"within 2 deg of horizontal — physically implausible for a real beam tilt; "
                            f"more likely altitude_m is wrong for this ping, or the detected strong-"
                            f"return sample is itself unreliable, than that the beam is genuinely "
                            f"almost-horizontal."
                        )
                elif np.isfinite(slant):
                    notes.append(
                        f"{label}: observed slant range ({slant:.2f}m) is less than altitude_m "
                        f"({altitude:.2f}m) at the detected strong-return sample — geometrically "
                        f"impossible for any real beam angle, suggesting altitude_m or the detection "
                        f"itself is unreliable for this ping/channel."
                    )
        implied[label] = entry

    # Cross-ping classification -- the single ping_idx entry above is
    # for the diagram plot only; the classification the automated
    # pipeline reports is based on every ping, not one, since a single
    # ping's detector noise is a weak basis for "is this survey's beam
    # tilted or not."
    implied_angle_deg_median = None
    implied_angle_deg_std = None
    implied_angle_valid_fraction = 0.0
    classification = "undetermined"

    if first_strong is not None:
        n_p, n_c = first_strong.shape
        n_samples = ds.sizes["sample"]
        altitude_all = ds["altitude_m"].values.astype(np.float64)  # (ping,)
        altitude_valid = np.isfinite(altitude_all) & (altitude_all > 0)

        boundary_valid = np.isfinite(first_strong) & altitude_valid[:, None]
        b_idx = np.where(boundary_valid, np.round(first_strong), 0).astype(int)
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
        implied_angle_valid_fraction = float(finite_angles.size / angle_from_vertical_all.size)
        if finite_angles.size > 0:
            implied_angle_deg_median = float(np.median(finite_angles))
            implied_angle_deg_std = float(np.std(finite_angles))

        if implied_angle_deg_median is None or implied_angle_valid_fraction < min_valid_fraction:
            classification = "undetermined"
            notes.append(
                f"Only {implied_angle_valid_fraction:.0%} of (ping, channel) pairs yielded a "
                f"valid implied angle -- too few to classify the weak zone's physical nature "
                f"(need >= {min_valid_fraction:.0%})."
            )
        elif implied_angle_deg_median < near_vertical_threshold_deg:
            classification = "near_grazing_weak_backscatter"
            notes.append(
                f"Cross-ping implied angle: median {implied_angle_deg_median:.1f} deg from "
                f"vertical (std {implied_angle_deg_std:.1f} deg, "
                f"{implied_angle_valid_fraction:.0%} of pings/channels valid) -- close to "
                f"straight down. The Stage 2 weak zone here is best read as genuine "
                f"near-grazing-incidence weak backscatter, not a distinct off-vertical water "
                f"column."
            )
        elif implied_angle_deg_std is not None and implied_angle_deg_std <= max_consistent_std_deg:
            classification = "water_column_tilted_beam"
            notes.append(
                f"Cross-ping implied angle: median {implied_angle_deg_median:.1f} deg from "
                f"vertical (std {implied_angle_deg_std:.1f} deg, "
                f"{implied_angle_valid_fraction:.0%} of pings/channels valid) -- consistent "
                f"with a genuinely tilted beam. The Stage 2 weak zone is best read as *water "
                f"column*, not weak bottom backscatter: the beam isn't illuminating that "
                f"range at all until its own footprint reaches the seafloor."
            )
        else:
            classification = "ambiguous"
            notes.append(
                f"Cross-ping implied angle is inconsistent (median "
                f"{implied_angle_deg_median:.1f} deg, std {implied_angle_deg_std:.1f} deg "
                f"across {implied_angle_valid_fraction:.0%} valid pings/channels) -- neither "
                f"a clean tilted-beam nor a clean straight-down interpretation is well "
                f"supported for this survey."
            )

    metrics = {
        "ping_idx": ping_idx,
        "altitude_m": altitude,
        "candidates": candidates,
        "implied": implied,
        "implied_angle_deg_median": implied_angle_deg_median,
        "implied_angle_deg_std": implied_angle_deg_std,
        "implied_angle_valid_fraction": implied_angle_valid_fraction,
    }
    return StageResult(stage="stage2_5_beam_geometry", metrics=metrics, classification=classification, notes=notes)


# ============================================================
# STAGE 3 — USABLE RANGE
# ============================================================


def stage3_usable_range(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    range_var: str = "ground_range_m",
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

    Parameters
    ----------
    exclude_before_sample : np.ndarray, optional
        Shape (ping, channel) — for each ping/channel, mask samples
        before this sample index to NaN before binning, on top of
        whatever `range_var` already excludes geometrically. Meant to
        be `stage2_nadir()`'s `data["first_strong_return_sample"]`: the
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

    value_finite = np.isfinite(np.asarray(ds[value_var].values))
    range_finite = np.isfinite(np.asarray(ds[range_var].values))
    n_water_column_excluded = int(np.sum(value_finite & ~range_finite))

    n_weak_zone_excluded = 0
    if exclude_before_sample is not None:
        original = ds[value_var]
        masked = _mask_before_sample(original.values, exclude_before_sample)
        n_weak_zone_excluded = int(np.sum(np.isfinite(original.values) & range_finite & ~np.isfinite(masked)))
        ds = ds.assign(**{value_var: (original.dims, masked)})

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
                "water_column_n_excluded": n_water_column_excluded,
                "weak_zone_n_excluded": n_weak_zone_excluded,
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
        "water_column_n_excluded": n_water_column_excluded,
        "weak_zone_n_excluded": n_weak_zone_excluded,
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
    range_var: str = "ground_range_m",
    *,
    already_db: bool = False,
    db_reference_percentile: float = 99.0,
    db_reference_value: float | None = None,
    freq_hz: float | None = None,
    bin_width_m: float = 5.0,
    usable_range_m: float | None = None,
    ambiguous_rel_tol: float = 0.2,
    min_total_count: int = 30,
    exclude_before_sample: np.ndarray | None = None,
    reject_outliers: bool = False,
    outlier_mad_k: float = 3.5,
    outlier_min_bin_count: int = 5,
    fit_value_col: str = "mean",
) -> StageResult:
    """Fit the four `fit_hardware_tvg_curve` candidate models against
    `value_var` vs. `range_var` and classify what TVG (if any) the data
    already carries: none, spreading_only, absorption_only, full, or
    ambiguous (winner not clearly separated from the runner-up).

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
        to be `stage2_nadir()`'s `data["first_strong_return_sample"]`
        — see `stage3_usable_range`'s identical parameter for why the
        geometric mask alone isn't enough. `None` (default) applies no
        additional exclusion.
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
    """
    if range_var not in ds:
        raise ValueError(f"stage4_decay: {range_var!r} not found — compute geometric ranges first")

    n_water_column_excluded = int(np.sum(
        np.isfinite(np.asarray(ds[value_var].values)) & ~np.isfinite(np.asarray(ds[range_var].values))
    ))

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

    n_weak_zone_excluded = 0
    if exclude_before_sample is not None:
        db_before = np.asarray(db_da.values)
        db_masked = _mask_before_sample(db_before, exclude_before_sample)
        n_weak_zone_excluded = int(np.sum(np.isfinite(db_before) & ~np.isfinite(db_masked)))
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

    if binned.empty or int(binned["count"].sum()) < min_total_count:
        return StageResult(
            stage="stage4_decay",
            metrics={
                "freq_hz": freq_hz, "db_reference": db_reference,
                "water_column_n_excluded": n_water_column_excluded,
                "weak_zone_n_excluded": n_weak_zone_excluded,
                "reject_outliers": reject_outliers,
                "outlier_mad_k": outlier_mad_k if reject_outliers else None,
                "outlier_n_rejected": n_outliers_rejected,
                "fit_value_col": fit_value_col,
            },
            classification="ambiguous",
            notes=["Not enough binned samples to fit a decay model."],
            data={"binned": binned},
        )

    absorption_db_per_m = absorption_for_frequency(freq_hz)
    fit_result = fit_hardware_tvg_curve(
        binned, absorption_db_per_m=absorption_db_per_m, r_fit_max_m=usable_range_m,
        value_col=fit_value_col,
    )

    ranked = sorted(fit_result.candidates, key=lambda c: c.rmse_db)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    notes: list[str] = []
    if second is not None and best.rmse_db > 0 and (second.rmse_db - best.rmse_db) / best.rmse_db < ambiguous_rel_tol:
        classification = "ambiguous"
        notes.append(
            f"Best ({best.name}, RMSE={best.rmse_db:.2f}dB) not clearly separated from "
            f"runner-up ({second.name}, RMSE={second.rmse_db:.2f}dB)."
        )
    else:
        classification = _TVG_CANDIDATE_TO_STATUS[best.name]
        if second is not None:
            notes.append(f"Runner-up: {second.name} (RMSE={second.rmse_db:.2f}dB).")

    metrics = {
        "freq_hz": freq_hz,
        "db_reference": db_reference,
        "absorption_db_per_m": absorption_db_per_m,
        "bin_width_m": bin_width_m,
        "r_ref_m": best.r_ref_m,
        "r_fit_max_m": best.r_fit_max_m,
        "n_bins_used": best.n_bins_used,
        "best_candidate": best.name,
        "best_scale": best.scale,
        "best_rmse_db": best.rmse_db,
        "ambiguous_rel_tol": ambiguous_rel_tol,
        "candidates": {c.name: {"scale": c.scale, "rmse_db": c.rmse_db} for c in fit_result.candidates},
        "water_column_n_excluded": n_water_column_excluded,
        "weak_zone_n_excluded": n_weak_zone_excluded,
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
