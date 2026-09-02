"""
TVG diagnostic analysis primitives.

Stage 1 of docs/planning/v0.8.0-tvg-diagnostics.md — pure computation
(no matplotlib) supporting investigation of the range-dependent TVG
residual (CLAUDE.md Known Issue #4) and floor-clipped samples (Known
Issue #5), without changing `processing/radiometric.py`'s correction
math. Consumes `apply_tvg()`'s existing public API as-is.

2026 rshomberg@uri.edu
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.pipeline import apply_correction_steps
from sidescanpy.processing.radiometric import TVGParams, apply_tvg, tvg_gain_db

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
        Passed directly to `apply_tvg()` rather than through
        `apply_correction_steps(ds_raw, {"tvg"}, step_params=...)` —
        both now support custom `TVGParams` (`step_params` added
        alongside this module's `fit_hardware_tvg_curve()`), this
        function just has no reason to route through the extra
        machinery for a single step it already calls directly.

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


# ============================================================
# HARDWARE-TVG CANDIDATE FITTING
# ============================================================
#
# What follows formalizes an investigation this project's own history
# (CLAUDE.md Known Issue #4, and the "TVG Parameters or Sound Velocity?"
# artifact it led to) did by hand: real sonar hardware appears to
# pre-apply *partial* TVG correction to "raw" backscatter before it ever
# reaches this library, and apply_tvg()'s own physics-based correction
# then double-corrects whatever hardware already fixed. There's no way
# to know this in advance for a new survey — the functions below fit a
# small set of candidate explanations against that survey's own observed
# raw-vs-range curve and report which one the data actually looks like.

_TVG_CANDIDATE_NAMES = ("flat", "spreading_only", "absorption_only", "full_physics")


@dataclass(frozen=True)
class TVGCandidateFit:
    """One candidate hardware-correction hypothesis, fit against an
    observed raw-backscatter-vs-range curve.

    `scale` is the fitted multiplier on the candidate's model curve:
    close to 1.0 means the observed data matches that candidate's model
    at face value (e.g. `absorption_only` at scale~1 means the raw
    signal decays exactly like uncompensated absorption loss alone, as
    if hardware had already removed spreading loss); close to 0.0 means
    that term contributes essentially nothing to what's observed, i.e.
    hardware likely already compensated for it. `rmse_db` is this
    free-fit scale's residual — kept as a diagnostic (a scale far from
    1.0 is itself informative, e.g. real survey checks found
    `absorption_only` scales of ~2-2.4 alongside a near-1.0
    `full_physics` scale — worth seeing, not hidden) but **not** what
    `stage4_decay`'s classification is based on (see `fixed_scale_rmse_db`).

    `fixed_scale_rmse_db` is the residual with the scale pinned to
    exactly 1.0 — the literal, uncalibrated physics model, no fitting
    at all. This is what `stage4_decay` actually ranks candidates by
    (2026-08-24 change). Real-data testing found the free-fit RMSE
    ranking exploitable: a single free multiplier on one term can chase
    down a low RMSE by absorbing whatever residual is left over — beam
    pattern effects, AGC roll-off, turbidity, anything with a roughly
    range-linear shape — not just genuine absorption, the same
    collinearity risk this module's own joint-fit precedent already
    warns about (see the module-level comment above `_TVG_CANDIDATE_NAMES`).
    Comparing literal models instead is harder to fool: a candidate
    only wins by actually matching the physics as specified, not by
    however much free scaling happens to minimize residual within one
    fit window. `flat`'s value is identical to `rmse_db` — it has no
    scale to begin with (its model is identically zero), so pinning the
    (nonexistent) scale changes nothing.
    """

    name: str  # one of _TVG_CANDIDATE_NAMES
    params: TVGParams  # TVGParams used to generate this candidate's model curve
    scale: float
    rmse_db: float
    fixed_scale_rmse_db: float
    r_ref_m: float
    r_fit_max_m: float
    n_bins_used: int
    # Near edge of the fit window -- None (default) means "not set by this
    # caller," in which case it's equivalent to r_ref_m (the prior,
    # one-sided-only behavior). Callers building a TVGCandidateFit by hand
    # (e.g. test fixtures predating this field) don't need to set it.
    r_fit_min_m: float | None = None


@dataclass(frozen=True)
class TVGCandidateFitResult:
    """All four candidate fits for one raw-vs-range curve, plus the
    winner. `best` is ranked by `fixed_scale_rmse_db` (literal,
    uncalibrated model), not the free-fit `rmse_db` -- see
    `TVGCandidateFit`'s docstring."""

    candidates: list[TVGCandidateFit]  # fixed order: _TVG_CANDIDATE_NAMES
    best: TVGCandidateFit
    observed_range_m: np.ndarray  # bin centers used for the fit (for plotting)
    observed_loss_db: np.ndarray  # observed_loss(r) at those centers (for plotting)


def fit_hardware_tvg_curve(
    binned: pd.DataFrame,
    *,
    spreading_loss_db: float = 20.0,
    absorption_db_per_m: float,
    r_ref_m: float | None = None,
    r_fit_min_m: float | None = None,
    r_fit_max_m: float | None = None,
    min_count_for_r_ref: int = 30,
    fit_fraction: float = 0.6,
    r_ref_search_fraction: float = 0.5,
    value_col: str = "mean",
) -> TVGCandidateFitResult:
    """
    Fit four candidate hardware-correction models against an observed
    raw-backscatter-vs-range curve, to characterize what a sonar's
    hardware/acquisition chain may have already applied before this
    library ever saw the data.

    Each candidate is one term (or combination) of `tvg_gain_db()`'s own
    formula, reused directly rather than re-derived — no formula
    duplication:

    - `"flat"`: neither term present (hardware fully compensated both).
    - `"spreading_only"`: only `20*log10(R)` present (hardware already
      removed absorption loss, spreading loss remains).
    - `"absorption_only"`: only `2*absorption_db_per_m*R` present
      (hardware already removed spreading loss, absorption remains).
    - `"full_physics"`: both terms present, i.e. hardware applied no
      correction at all — the raw signal decays exactly per the
      physically-grounded model `apply_tvg()` assumes.

    Each candidate is fit with **one free scale parameter,
    independently** — this deliberately does NOT fit a single joint
    2-parameter regression over spreading and absorption simultaneously.
    A joint fit was tried during the investigation that motivated this
    function and found poorly conditioned: over realistic range spans
    the two terms are too collinear, producing wild, sign-flipped, or
    physically uninterpretable coefficients even when the combined curve
    visually fit well. Do not "improve" this into a joint fit without
    re-deriving why that failed (see
    `docs/planning/v0.13.0-tvg-term-selection-and-hardware-curve-fitting.md`).

    Parameters
    ----------
    binned : pd.DataFrame
        `mean_backscatter_vs_range()`'s own output, computed by the
        caller on **raw** (uncorrected) backscatter — e.g.
        `mean_backscatter_vs_range(ds_raw_or_reconstructed,
        value_var="backscatter_db_raw", ...)` from
        `before_after_backscatter()`'s output. This function does no
        binning of its own; it only fits candidate curves to bins
        already computed.
    spreading_loss_db : float, default 20.0
        Physical constant (spherical spreading) — rarely needs
        overriding, unlike `absorption_db_per_m`.
    absorption_db_per_m : float
        No default: the `absorption_only`/`full_physics` candidates are
        only meaningful if built from *this survey's actual*
        frequency-appropriate coefficient. Callers should pass
        `absorption_for_frequency(ds.attrs["freq_hz_start"])` explicitly
        rather than rely on this function to guess it.
    r_ref_m : float, optional
        Reference range bin — `observed_loss(r) = mean[r_ref] - mean[r]`
        is what every candidate is actually fit against, since
        `tvg_gain_db()`'s absolute level has no meaning without an
        unknown source-level/target-strength constant. Auto-selected
        when `None` as the bin with the **maximum mean** among
        `min_count_for_r_ref`-eligible bins that also fall within the
        near-field search window (`r_ref_search_fraction`, below) — the
        physically-motivated reference is the strongest observed return
        (the near-nadir peak), not simply the first/nearest bin. Real
        data has a blanking/ring-down artifact immediately at the start
        of every ping (near-zero-range bins read anomalously *low*, not
        high — see Known Issue #5) that a "first bin past some count"
        rule would otherwise anchor on, inverting the sign of every fit.
        Restricting the search to the near-field half (rather than the
        full range) matters separately: on a survey where hardware has
        already flattened the raw curve almost entirely (peak-to-trough
        under ~1dB — pure measurement noise at that point, not a real
        peak), an unrestricted global-max search can land on a
        far-range bin purely by chance, which then collapses
        `r_fit_max_m`'s window to a single bin (see `fit_fraction`) and
        makes every candidate's fit degenerate (RMSE=0 for all of
        them). Found on a real, genuinely-flat Klein survey.
    r_fit_min_m : float, optional
        Start of the fit window (inclusive). `None` (default) reproduces
        this function's original one-sided behavior: the fit window
        starts at `r_ref_m` itself, so only bins at or beyond the
        reference are fit. `r_ref_m` still always serves as the loss
        reference (`observed_loss_db = means[i_ref] - means`) regardless
        of `r_fit_min_m` — this parameter only changes which bins the
        candidate models are actually scored against, not what "zero
        loss" means.

        Set explicitly (e.g. `0.0`, or just past a known near-boundary
        artifact) to fit **both sides** of `r_ref_m` — matters once
        `r_ref_m` is deliberately pinned well into the swath (e.g. the
        middle) rather than left at its near-field auto-pick: with the
        old one-sided window, moving `r_ref_m` out that far would throw
        away everything nearer than it, discarding most of the usable
        near-field data for no reason. Real motivation (2026-09-02): a
        bright artifact sits immediately at the first-bottom-return
        boundary (see `stage4_decay`'s `exclude_first_n_bins`) — the
        fix for that is dropping the handful of bins nearest the
        boundary, not the entire near half of the swath.
    r_fit_max_m : float, optional
        End of the fit window (inclusive). Auto-selected when `None`
        (see below). Bins beyond this are ignored by the fit (though
        still returned in `observed_range_m`/`observed_loss_db` for
        plotting the full curve).
    min_count_for_r_ref : int, default 30
        When auto-selecting `r_ref_m`, only consider bins with at least
        this many samples — avoids anchoring the loss reference on a
        thin, noisy bin whose mean is an unreliable estimate. Conservative
        floor; lower only for very short/sparse ping ranges.
    fit_fraction : float, default 0.6
        When auto-selecting `r_fit_max_m`: `r_ref_m + fit_fraction *
        (max_observed_range_m - r_ref_m)` — restricts the fit to a
        contiguous near-to-mid window, deliberately excluding the
        far-range tail, where Known Issue #6's noise-amplification
        artifacts live and per-bin sample counts thin out. This is a
        reasonable, inspectable heuristic, not a validated one —
        callers investigating a real survey should still look at the
        full curve (via the returned `observed_range_m`/
        `observed_loss_db`, or `viz.tvg_diagnostics.plot_candidate_fit`)
        and override `r_fit_max_m` directly if the auto-selected window
        looks wrong for that survey.
    r_ref_search_fraction : float, default 0.5
        When auto-selecting `r_ref_m`, only consider bins within this
        fraction of the observed range span from its near edge (e.g.
        0.5 = the near-field half). Keeps the auto-selected reference
        physically near-field even when the true peak is only a
        fraction of a dB above the noise floor everywhere else — see
        `r_ref_m`'s docstring above for the failure mode this prevents.
        Falls back to the full range if no eligible bin falls in that
        window (shouldn't happen with realistic range coverage).
    value_col : str, default "mean"
        Which of `binned`'s per-bin aggregate columns to fit against —
        `"mean"` (the original behavior, unchanged for every existing
        caller) or `"median"`. Added for pairing with aggressive
        per-bin outlier rejection upstream (e.g.
        `analysis.investigation.reject_range_outliers`), on the theory
        that a plain arithmetic mean isn't distribution-neutral on a
        right-skewed population the way a median is. **That theory
        didn't hold up cleanly on real data** — see
        `analysis.investigation.stage4_decay`'s `fit_value_col`
        docstring for the actual (mixed, dataset-dependent) result.
        Internal variable names below still say `means` for a minimal
        diff; they hold whichever column `value_col` selects.

    Returns
    -------
    TVGCandidateFitResult

    Raises
    ------
    ValueError
        If `binned` is empty, or no bin has `count >= min_count_for_r_ref`.
    """
    if binned.empty:
        raise ValueError("fit_hardware_tvg_curve: `binned` has no rows")

    range_centers = ((binned["range_min_m"] + binned["range_max_m"]) / 2.0).to_numpy()
    means = binned[value_col].to_numpy()
    counts = binned["count"].to_numpy()

    order = np.argsort(range_centers)
    range_centers, means, counts = range_centers[order], means[order], counts[order]

    if r_ref_m is None:
        eligible = np.where(counts >= min_count_for_r_ref)[0]
        if eligible.size == 0:
            raise ValueError(
                f"fit_hardware_tvg_curve: no bin has count >= {min_count_for_r_ref}; "
                "lower min_count_for_r_ref or pass r_ref_m explicitly"
            )
        # Restrict to the near-field half (or r_ref_search_fraction) of
        # the range span -- see r_ref_search_fraction's docstring for
        # why an unrestricted global-max search can fail on real,
        # near-flat data. Falls back to all eligible bins if none fall
        # in that window.
        near_field_cutoff = range_centers.min() + r_ref_search_fraction * (
            range_centers.max() - range_centers.min()
        )
        near_field_eligible = eligible[range_centers[eligible] <= near_field_cutoff]
        search_pool = near_field_eligible if near_field_eligible.size > 0 else eligible
        # Strongest observed return among eligible bins, not merely the
        # first/nearest one -- see the "Real data has a blanking/
        # ring-down artifact" note in the docstring for why "first bin
        # past some count" silently anchors on an artifact instead.
        best_eligible = search_pool[np.argmax(means[search_pool])]
        r_ref_m = float(range_centers[best_eligible])

    if r_fit_max_m is None:
        r_fit_max_m = r_ref_m + fit_fraction * (float(range_centers.max()) - r_ref_m)
    if r_fit_min_m is None:
        r_fit_min_m = r_ref_m  # prior one-sided behavior, unchanged default

    i_ref = int(np.argmin(np.abs(range_centers - r_ref_m)))
    observed_loss_db = means[i_ref] - means

    fit_mask = (range_centers >= r_fit_min_m) & (range_centers <= r_fit_max_m)
    n_bins_used = int(fit_mask.sum())

    param_sets = {
        "flat": TVGParams(spreading_loss_db=0.0, absorption_db_per_m=0.0),
        "spreading_only": TVGParams(spreading_loss_db=spreading_loss_db, absorption_db_per_m=0.0),
        "absorption_only": TVGParams(spreading_loss_db=0.0, absorption_db_per_m=absorption_db_per_m),
        "full_physics": TVGParams(spreading_loss_db=spreading_loss_db, absorption_db_per_m=absorption_db_per_m),
    }

    # tvg_gain_db() takes range_m >= 0 with no singularity at r=0 for
    # log10 only when r>0 -- clip at the same min_range_m floor
    # apply_tvg()/apply_tvg_to_amplitude() already use, purely to avoid
    # log10(0) on a possible zero-range bin center; this has no bearing
    # on the fit itself (both model and reference are clipped identically).
    range_for_model = np.clip(range_centers, TVGParams().min_range_m, None)

    candidates = []
    for name in _TVG_CANDIDATE_NAMES:
        params = param_sets[name]
        model_curve = tvg_gain_db(range_for_model, params)
        # tvg_gain_db is a *gain* (grows with range, meant to compensate
        # loss); a raw signal that still carries this term uncompensated
        # is correspondingly *weaker* at longer range by the same
        # amount. observed_loss_db is defined ref-minus-r (positive =
        # weaker at r than at r_ref), so the model must match that sign:
        # r-minus-ref, the opposite subtraction order from the gain
        # curve's own natural (increasing) direction.
        model_loss = model_curve - model_curve[i_ref]

        if name == "flat":
            # Model is identically zero everywhere -- no free scale to
            # fit (scale*0 == 0 for any scale), so report scale=0.0 and
            # RMSE directly against the observed loss itself. Fixed-scale
            # residual is identical -- pinning a nonexistent scale to 1.0
            # changes nothing (1.0*0 == 0.0*0).
            scale = 0.0
            residual = observed_loss_db[fit_mask]
            fixed_residual = residual
        else:
            m = model_loss[fit_mask]
            o = observed_loss_db[fit_mask]
            denom = np.dot(m, m)
            scale = float(np.dot(m, o) / denom) if denom > 0 else 0.0
            residual = scale * m - o
            # scale pinned to exactly 1.0 -- the literal, uncalibrated
            # physics model. See TVGCandidateFit's docstring for why
            # stage4_decay ranks candidates by this, not `rmse_db`.
            fixed_residual = m - o

        rmse_db = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
        fixed_scale_rmse_db = float(np.sqrt(np.mean(fixed_residual**2))) if fixed_residual.size else float("nan")

        candidates.append(
            TVGCandidateFit(
                name=name,
                params=params,
                scale=scale,
                rmse_db=rmse_db,
                fixed_scale_rmse_db=fixed_scale_rmse_db,
                r_ref_m=r_ref_m,
                r_fit_min_m=r_fit_min_m,
                r_fit_max_m=r_fit_max_m,
                n_bins_used=n_bins_used,
            )
        )

    # Ranked by fixed_scale_rmse_db (literal, uncalibrated model), not
    # the free-fit rmse_db -- matches stage4_decay's own ranking
    # criterion (2026-08-24 change; see TVGCandidateFit's docstring).
    best = min(candidates, key=lambda c: c.fixed_scale_rmse_db)

    return TVGCandidateFitResult(
        candidates=candidates,
        best=best,
        observed_range_m=range_centers,
        observed_loss_db=observed_loss_db,
    )
