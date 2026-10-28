"""
TVG analysis: backscatter-vs-range profiles and hardware-TVG detection.

Profiles and comparisons:

- :func:`backscatter_vs_range`, :func:`backscatter_vs_sample` — binned
  statistics (re-exported from :mod:`sidescanpy.processing.profiles`).
- :func:`compare_before_after` — raw vs. TVG-corrected dB and the
  applied gain, side by side.
- :func:`bin_by_altitude` — split pings into equal-count altitude bins.

Detecting what TVG the data already carries:

- :func:`fit_tvg_curve` — fit four single-term candidate loss models
  (``flat``, ``spreading_only``, ``absorption_only``, ``full_physics``)
  to an observed loss-vs-range curve (:class:`TVGCandidateFitResult`).
- :func:`fit_range_decay` — bin a dataset, fit the candidates and
  classify which terms hardware has already removed
  (:class:`RangeDecayResult`).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from sidescanpy.analysis._common import channel_mask, resolve_channel_idx, water_column_var
from sidescanpy.analysis._result import ResultMixin, array_field
from sidescanpy.core.units import MIN_LINEAR, linear_to_db
from sidescanpy.processing.bottom import mask_before_sample
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.outliers import reject_range_outliers
from sidescanpy.processing.profiles import backscatter_vs_range, backscatter_vs_sample
from sidescanpy.processing.radiometric import (
    DEFAULT_SPREADING_LOSS_DB,
    TVGParams,
    absorption_for_frequency,
    apply_tvg,
    center_frequency_hz,
    tvg_gain_db,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RangeDecayResult",
    "TVGCandidateFit",
    "TVGCandidateFitResult",
    "backscatter_vs_range",
    "backscatter_vs_sample",
    "bin_by_altitude",
    "compare_before_after",
    "fit_range_decay",
    "fit_tvg_curve",
]


def compare_before_after(ds_raw: xr.Dataset, params: TVGParams | None = None) -> xr.Dataset:
    """Raw and TVG-corrected backscatter side by side, with ground range.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Uncorrected channel dataset (see
        :func:`sidescanpy.processing.invert`).
    params : TVGParams, optional
        Passed to :func:`~sidescanpy.processing.radiometric.apply_tvg`.

    Returns
    -------
    xr.Dataset
        ``ds_raw`` with geometric ranges plus ``backscatter_db_raw``
        (linear → dB, no gain), ``backscatter_db_corrected`` (``apply_tvg``
        output) and ``tvg_gain_db`` (their difference).
    """
    ds_geom = compute_geometric_ranges(ds_raw)
    ds_tvg = apply_tvg(ds_raw, params)

    backscatter_db_raw = linear_to_db(ds_raw["backscatter_intensity"])
    backscatter_db_corrected = ds_tvg["backscatter_intensity"]
    gain_db = backscatter_db_corrected - backscatter_db_raw

    out = ds_geom.copy(deep=False)
    out.attrs = dict(ds_geom.attrs)
    out = out.assign(
        backscatter_db_raw=backscatter_db_raw,
        backscatter_db_corrected=backscatter_db_corrected,
        tvg_gain_db=gain_db,
    )
    return out


def bin_by_altitude(
    ds: xr.Dataset,
    n_bins: int = 3,
    altitude_var: str = "altitude_m",
) -> dict[str, xr.Dataset]:
    """Split pings into ``n_bins`` equal-count (quantile) altitude bins.

    Comparing a range profile across altitude bins separates
    geometry-dependent effects (e.g. grazing angle) from range-only ones.

    Returns
    -------
    dict[str, xr.Dataset]
        Keys ``"altitude_{lo:.2f}_{hi:.2f}m"`` in increasing altitude.
        Pings with non-finite altitude are in no bin.
    """
    altitude = np.asarray(ds[altitude_var].values, dtype=np.float64)
    valid = np.isfinite(altitude)

    edges = np.percentile(altitude[valid], np.linspace(0, 100, n_bins + 1))
    # A constant altitude collapses every edge to one value; widen it.
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

_TVG_CANDIDATE_NAMES = ("flat", "spreading_only", "absorption_only", "full_physics")


@dataclass(frozen=True)
class TVGCandidateFit:
    """One candidate loss model fit against an observed loss-vs-range curve.

    ``scale`` is the least-squares multiplier on the candidate's model
    (near 1: the data decays like that model as written; near 0: that
    term is essentially absent) and ``rmse_db`` its residual.
    ``fixed_scale_rmse_db`` is the residual with the scale pinned to 1 —
    the literal physics model. Candidates are ranked by the fixed-scale
    value, because a free scale on one term can absorb unrelated
    range-dependent effects and win on RMSE alone. For ``flat`` (model
    identically zero) both values are equal.
    """

    name: str  # one of _TVG_CANDIDATE_NAMES
    params: TVGParams  # TVGParams used to generate this candidate's model curve
    scale: float
    rmse_db: float
    fixed_scale_rmse_db: float
    r_ref_m: float
    r_fit_max_m: float
    n_bins_used: int
    # Near edge of the fit window; None means "same as r_ref_m"
    # (a one-sided window).
    r_fit_min_m: float | None = None


@dataclass(frozen=True)
class TVGCandidateFitResult:
    """All four candidate fits for one curve, plus the best by fixed-scale RMSE."""

    candidates: list[TVGCandidateFit]  # fixed order: _TVG_CANDIDATE_NAMES
    best: TVGCandidateFit
    observed_range_m: np.ndarray  # bin centers (for plotting)
    observed_loss_db: np.ndarray  # observed loss at those centers (for plotting)


def fit_tvg_curve(
    binned: pd.DataFrame,
    *,
    spreading_loss_db: float = DEFAULT_SPREADING_LOSS_DB,
    absorption_db_per_m: float,
    r_ref_m: float | None = None,
    r_fit_min_m: float | None = None,
    r_fit_max_m: float | None = None,
    min_count_for_r_ref: int = 30,
    fit_fraction: float = 0.6,
    r_ref_search_fraction: float = 0.5,
    value_col: str = "mean",
) -> TVGCandidateFitResult:
    """Fit four candidate hardware-TVG models to a binned raw dB-vs-range curve.

    Each candidate is a term (or both) of
    :func:`~sidescanpy.processing.radiometric.tvg_gain_db` still present
    in the data:

    - ``flat`` — neither term (hardware compensated both)
    - ``spreading_only`` — only ``spreading_loss_db * log10(R)``
    - ``absorption_only`` — only ``2 * absorption_db_per_m * R``
    - ``full_physics`` — both (no hardware compensation)

    The fitted quantity is ``observed_loss(r) = value[r_ref] - value[r]``
    (absolute level is unknown). Each candidate gets one free scale,
    fitted independently: a joint two-term regression is poorly
    conditioned because the terms are nearly collinear over realistic
    range spans.

    Parameters
    ----------
    binned : pd.DataFrame
        :func:`backscatter_vs_range` output computed on uncorrected dB
        backscatter.
    spreading_loss_db : float, default 30.0
        Spreading coefficient of the ``spreading_only`` and ``full_physics``
        candidates (:data:`~sidescanpy.processing.radiometric.DEFAULT_SPREADING_LOSS_DB`).
    absorption_db_per_m : float
        Required; pass the survey's own frequency-appropriate value.
    r_ref_m : float, optional
        Reference range. Default: the bin with the highest value among
        bins with ``count >= min_count_for_r_ref`` in the near
        ``r_ref_search_fraction`` of the range span (falls back to all
        eligible bins). Not simply the nearest bin, which often reads low
        from ring-down/blanking; not the global max, which on flat data
        can land on a far-range noise bump.
    r_fit_min_m : float, optional
        Near edge of the fit window. Default ``r_ref_m`` (one-sided).
        Set it lower to fit both sides of the reference.
    r_fit_max_m : float, optional
        Far edge of the fit window. Default
        ``r_ref_m + fit_fraction * (max_range - r_ref_m)``.
    min_count_for_r_ref : int, default 30
    fit_fraction : float, default 0.6
    r_ref_search_fraction : float, default 0.5
    value_col : {"mean", "median"}, default "mean"
        Per-bin statistic to fit.

    Returns
    -------
    TVGCandidateFitResult

    Raises
    ------
    ValueError
        If ``binned`` is empty, or no bin has ``count >= min_count_for_r_ref``.
    """
    if binned.empty:
        raise ValueError("fit_tvg_curve: `binned` has no rows")

    range_centers = ((binned["range_min_m"] + binned["range_max_m"]) / 2.0).to_numpy()
    means = binned[value_col].to_numpy()
    counts = binned["count"].to_numpy()

    order = np.argsort(range_centers)
    range_centers, means, counts = range_centers[order], means[order], counts[order]

    if r_ref_m is None:
        eligible = np.where(counts >= min_count_for_r_ref)[0]
        if eligible.size == 0:
            raise ValueError(
                f"fit_tvg_curve: no bin has count >= {min_count_for_r_ref}; "
                "lower min_count_for_r_ref or pass r_ref_m explicitly"
            )
        near_field_cutoff = range_centers.min() + r_ref_search_fraction * (
            range_centers.max() - range_centers.min()
        )
        near_field_eligible = eligible[range_centers[eligible] <= near_field_cutoff]
        search_pool = near_field_eligible if near_field_eligible.size > 0 else eligible
        best_eligible = search_pool[np.argmax(means[search_pool])]
        r_ref_m = float(range_centers[best_eligible])

    if r_fit_max_m is None:
        r_fit_max_m = r_ref_m + fit_fraction * (float(range_centers.max()) - r_ref_m)
    if r_fit_min_m is None:
        r_fit_min_m = r_ref_m

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

    # Clip at apply_tvg's own min_range_m floor to avoid log10(0); model
    # and reference are clipped identically.
    range_for_model = np.clip(range_centers, TVGParams().min_range_m, None)

    candidates = []
    for name in _TVG_CANDIDATE_NAMES:
        params = param_sets[name]
        model_curve = tvg_gain_db(range_for_model, params)
        # A gain curve grows with range; the matching loss (ref minus r,
        # positive = weaker at r) is r minus ref of that gain.
        model_loss = model_curve - model_curve[i_ref]

        if name == "flat":
            # Model is identically zero: no scale to fit.
            scale = 0.0
            residual = observed_loss_db[fit_mask]
            fixed_residual = residual
        else:
            m = model_loss[fit_mask]
            o = observed_loss_db[fit_mask]
            denom = np.dot(m, m)
            scale = float(np.dot(m, o) / denom) if denom > 0 else 0.0
            residual = scale * m - o
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

    best = min(candidates, key=lambda c: c.fixed_scale_rmse_db)

    return TVGCandidateFitResult(
        candidates=candidates,
        best=best,
        observed_range_m=range_centers,
        observed_loss_db=observed_loss_db,
    )


# ============================================================
# RANGE-DECAY CLASSIFICATION
# ============================================================

# Candidate names say which term is still PRESENT in the data; the TVG
# status says what hardware has already REMOVED. They are inverses.
_TVG_CANDIDATE_TO_STATUS = {
    "flat": "full",
    "spreading_only": "absorption_only",
    "absorption_only": "spreading_only",
    "full_physics": "none",
}

# Empirical margin past the raw near-field peak for r_fit_min_m="auto";
# the corrected curve's transition zone runs further than the raw peak.
_AUTO_NEAR_EDGE_MARGIN_M = 5.0


def _auto_near_edge_m(
    ds: xr.Dataset, value_var: str, range_var: str, bin_width_m: float,
    search_max_m: float, margin_m: float,
) -> float | None:
    """Far edge of the near-field peak bin (max mean of raw ``value_var``
    within ``[0, search_max_m]``) plus ``margin_m``; ``None`` if nothing
    to search."""
    binned_raw = backscatter_vs_range(ds, value_var=value_var, range_var=range_var, bin_width_m=bin_width_m)
    if binned_raw.empty:
        return None
    centers = (binned_raw["range_min_m"] + binned_raw["range_max_m"]) / 2.0
    near = binned_raw[centers <= search_max_m]
    if near.empty:
        return None
    peak_idx = near["mean"].idxmax()
    peak_far_edge = float(near.loc[peak_idx, "range_max_m"])
    return peak_far_edge + margin_m


@dataclass
class RangeDecayResult(ResultMixin):
    """TVG-status classification from a range-decay fit.

    ``tvg_status`` names what hardware has already removed: ``"none"``,
    ``"spreading_only"``, ``"absorption_only"``, ``"full"``, or
    ``"ambiguous"`` (winner not clearly ahead, or too little data).
    ``best_rmse_db`` is the winner's fixed-scale RMSE (the ranking
    criterion); ``best_free_fit_*`` are its free-scale values.
    ``r_ref_m_override`` / ``r_fit_min_m_override`` record what was
    requested (e.g. ``"auto"``); ``r_ref_m`` / ``r_fit_min_m`` what was
    used. With ``r_fit_min_m="auto"`` resolved, ``r_fit_min_auto_search_max_m``
    is the bound of the near-field peak search (bin centers up to it) and
    ``r_fit_min_auto_peak_edge_m`` the peak bin's far edge, so the resolved
    value is ``r_fit_min_auto_peak_edge_m + r_fit_min_auto_margin_m`` before
    any clamp to ``r_ref_m``; it can lie past the search bound. ``fit`` (the
    full :class:`TVGCandidateFitResult`) is left out of ``to_dict()``;
    ``candidates`` summarizes it.
    """

    tvg_status: str | None = None
    freq_hz: float | None = None
    db_reference: dict | None = None
    absorption_db_per_m: float | None = None
    bin_width_m: float | None = None
    bin_width_m_requested: float | None = None
    exclude_first_n_bins: int = 0
    n_bins_excluded_after_bottom: int = 0
    r_ref_m: float | None = None
    r_ref_m_override: float | str | None = None
    r_fit_min_m: float | None = None
    r_fit_min_m_override: float | str | None = None
    r_fit_min_auto_margin_m: float | None = None
    r_fit_min_auto_search_max_m: float | None = None
    r_fit_min_auto_peak_edge_m: float | None = None
    r_fit_max_m: float | None = None
    n_bins_used: int | None = None
    best_candidate: str | None = None
    best_scale: float | None = None
    best_rmse_db: float | None = None
    best_free_fit_scale: float | None = None
    best_free_fit_rmse_db: float | None = None
    ambiguous_rel_tol: float | None = None
    candidates: dict[str, dict[str, float]] = field(default_factory=dict)
    range_var: str | None = None
    channel: int | None = None
    water_column_n_excluded: int = 0
    nadir_gap_n_excluded: int = 0
    reject_outliers: bool = False
    outlier_mad_k: float | None = None
    outlier_n_rejected: int = 0
    fit_value_col: str | None = None
    notes: list[str] = field(default_factory=list)
    binned: pd.DataFrame | None = None
    fit: TVGCandidateFitResult | None = array_field()


def fit_range_decay(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
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
    r_ref_m: float | str | None = None,
    r_fit_min_m: float | str | None = None,
    r_fit_min_auto_margin_m: float = _AUTO_NEAR_EDGE_MARGIN_M,
    r_ref_search_fraction: float = 0.5,
    r_ref_window_fraction: float = 1.0 / 3.0,
    channel: int | str | None = None,
    spreading_loss_db: float = DEFAULT_SPREADING_LOSS_DB,
) -> RangeDecayResult:
    """Classify which TVG terms ``value_var`` already carries.

    Converts to dB, bins by ``range_var``, fits the four
    :func:`fit_tvg_curve` candidates and maps the winner (lowest
    fixed-scale RMSE) to a TVG status. If the runner-up's RMSE is within
    ``ambiguous_rel_tol`` (relative) of the winner's, the status is
    ``"ambiguous"``.

    Water-column samples are always excluded via ``ground_range_m``'s NaN
    pattern (when present), independent of ``range_var``.

    dB reference
    ------------
    Without calibration an absolute dB level is arbitrary, so linear
    values are converted relative to the data's own
    ``db_reference_percentile`` (or ``db_reference_value``). Every fit
    quantity is a dB difference, so this choice does not change the
    classification.

    Parameters
    ----------
    range_var : str, default "slant_range_m"
        Range axis; slant range matches ``apply_tvg``'s gain model.
    already_db : bool, default False
        ``value_var`` is already dB; skip the conversion.
    db_reference_percentile : float, default 99.0
    db_reference_value : float, optional
        Exact linear reference instead of the percentile (e.g. ``1.0``
        for "dB re 1").
    freq_hz : float, optional
        Default: the channel's center frequency
        (:func:`~sidescanpy.processing.radiometric.center_frequency_hz`);
        sets the absorption term.
    bin_width_m : float, default 5.0
        Requested bin width; shrunk (never widened) so that
        ``usable_range_m`` spans at least ``min_bins`` bins.
    usable_range_m : float, optional
        Far edge of the fit window.
    min_bins : int, default 6
    min_total_count : int, default 30
        Fewer binned samples than this → ``"ambiguous"`` without a fit.
    exclude_before_sample : np.ndarray, optional
        ``(ping, channel)`` sample index; samples before it are dropped
        (in dB space) before binning. For aggregate fits pass a uniform
        boundary (:func:`~sidescanpy.processing.bottom.uniform_exclusion_boundary`):
        a per-ping one biases the near-range bins and the reference pick.
    exclude_first_n_bins : int, default 0
        Drop this many nearest bins after binning, before the reference
        pick and fit — for a bright leading-edge artifact at the bottom
        return.
    reject_outliers : bool, default False
        Run :func:`~sidescanpy.processing.outliers.reject_range_outliers`
        on the dB values first (``outlier_mad_k``, ``outlier_min_bin_count``).
    fit_value_col : {"mean", "median"}, default "mean"
        Passed to :func:`fit_tvg_curve` as ``value_col``.
    r_ref_m : float, "auto", "mid_swath" or None
        Reference range. ``None``: :func:`fit_tvg_curve`'s near-field
        auto-pick. ``"auto"`` (recommended with ``r_fit_min_m="auto"``):
        placed ``r_ref_window_fraction`` of the way from the resolved fit
        window's near edge to ``usable_range_m``, so it always lies inside
        the window. ``"mid_swath"`` (legacy): ``usable_range_m / 2``. Both
        string modes fall back to the auto-pick, with a note, when
        ``usable_range_m`` is unavailable.
    r_fit_min_m : float, "auto" or None
        Near edge of the fit window (resolved before ``r_ref_m``).
        ``None``: one-sided from ``r_ref_m``. ``"auto"``: the raw
        near-field peak bin's far edge plus ``r_fit_min_auto_margin_m``,
        searched up to a numeric ``r_ref_m`` or
        ``r_ref_search_fraction * usable_range_m``; falls back to
        ``None`` with a note when it can't be resolved. Only the peak is
        searched for within that bound: the margin, added afterwards, can
        carry the value past it, and the result reports both
        (``r_fit_min_auto_search_max_m``, ``r_fit_min_auto_peak_edge_m``).
        If it ends up beyond a numeric/"mid_swath" ``r_ref_m`` it is
        clamped to it.
    r_fit_min_auto_margin_m : float, default 5.0
    r_ref_search_fraction : float, default 0.5
    r_ref_window_fraction : float, default 1/3
    channel : int or str, optional
        Restrict to one physical side (index or ``channel_id`` value).
        ``None`` pools both. Applied by masking.
    spreading_loss_db : float, default 30.0
        Spreading coefficient of the ``spreading_only`` and ``full_physics``
        candidates; pass the value the survey's correction uses.
    """
    if range_var not in ds:
        raise ValueError(f"fit_range_decay: {range_var!r} not found — compute geometric ranges first")

    channel_idx = resolve_channel_idx(ds, channel, caller="fit_range_decay") if channel is not None else None

    wc_var = water_column_var(ds, range_var)
    water_column_finite = np.isfinite(np.asarray(ds[wc_var].values))
    n_water_column_excluded = int(np.sum(
        np.isfinite(np.asarray(ds[value_var].values)) & ~water_column_finite
    ))

    # Shrink (never widen) the bin width so the range up to
    # usable_range_m holds at least min_bins bins.
    bin_width_m_requested = bin_width_m
    if usable_range_m is not None and usable_range_m > 0 and min_bins > 0:
        min_width_for_bins = usable_range_m / min_bins
        if min_width_for_bins < bin_width_m:
            bin_width_m = min_width_for_bins

    freq_hz = float(freq_hz) if freq_hz is not None else center_frequency_hz(ds)

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
                x_ref = MIN_LINEAR  # degenerate (all-zero/empty) fallback
            db_reference = {"method": "percentile", "percentile": db_reference_percentile, "value": x_ref}
        db_da = 20.0 * np.log10((np.abs(arr) + MIN_LINEAR) / x_ref)

    db_da = xr.DataArray(
        np.where(water_column_finite, db_da.values, np.nan), dims=db_da.dims, coords=db_da.coords,
    )

    if channel_idx is not None:
        db_da = xr.DataArray(
            np.where(channel_mask(ds, db_da, channel_idx), db_da.values, np.nan),
            dims=db_da.dims, coords=db_da.coords,
        )

    n_nadir_gap_excluded = 0
    if exclude_before_sample is not None:
        db_before = np.asarray(db_da.values)
        db_masked = mask_before_sample(db_before, exclude_before_sample)
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

    work = ds[[range_var]].assign(_decay_db=db_da)
    binned = backscatter_vs_range(work, value_var="_decay_db", range_var=range_var, bin_width_m=bin_width_m)

    n_bins_excluded_after_bottom = 0
    if exclude_first_n_bins > 0 and not binned.empty:
        binned = binned.sort_values("range_min_m").reset_index(drop=True)
        n_bins_excluded_after_bottom = min(exclude_first_n_bins, len(binned))
        binned = binned.iloc[n_bins_excluded_after_bottom:].reset_index(drop=True)

    result = RangeDecayResult(
        freq_hz=freq_hz,
        db_reference=db_reference,
        bin_width_m=bin_width_m,
        bin_width_m_requested=bin_width_m_requested,
        exclude_first_n_bins=exclude_first_n_bins,
        n_bins_excluded_after_bottom=n_bins_excluded_after_bottom,
        range_var=range_var,
        channel=channel_idx,
        water_column_n_excluded=n_water_column_excluded,
        nadir_gap_n_excluded=n_nadir_gap_excluded,
        reject_outliers=reject_outliers,
        outlier_mad_k=outlier_mad_k if reject_outliers else None,
        outlier_n_rejected=n_outliers_rejected,
        fit_value_col=fit_value_col,
        binned=binned,
    )

    if binned.empty or int(binned["count"].sum()) < min_total_count:
        result.tvg_status = "ambiguous"
        result.notes.append("Not enough binned samples to fit a decay model.")
        return result

    # Resolve the fit window's near edge FIRST, then (for r_ref_m="auto")
    # place the reference inside it. Picking the reference independently
    # and clamping the window to it can put the reference on the window's
    # noisy boundary, and a bad reference shifts every bin's loss.
    r_ref_m_requested = r_ref_m
    r_fit_min_m_requested = r_fit_min_m

    r_fit_min_m_auto_note: str | None = None
    auto_search_max_m: float | None = None
    auto_peak_edge_m: float | None = None
    if isinstance(r_fit_min_m, str):
        if r_fit_min_m != "auto":
            raise ValueError(f"fit_range_decay: r_fit_min_m string values must be 'auto', got {r_fit_min_m!r}")
        # Only a numeric r_ref_m can bound the search here; a string
        # r_ref_m is resolved afterwards.
        near_edge_search_max_m = r_ref_m if isinstance(r_ref_m, (int, float)) else (
            r_ref_search_fraction * usable_range_m if usable_range_m else None
        )
        auto_value = None
        if near_edge_search_max_m is None:
            r_fit_min_m_auto_note = (
                "r_fit_min_m='auto' requested but neither a numeric r_ref_m nor usable_range_m "
                "is available to bound its near-field search -- falling back to the prior "
                "one-sided fit window (starts at r_ref_m)."
            )
        else:
            auto_peak_edge_m = _auto_near_edge_m(
                ds, value_var, range_var, bin_width_m,
                search_max_m=near_edge_search_max_m, margin_m=0.0,
            )
            if auto_peak_edge_m is None:
                r_fit_min_m_auto_note = (
                    f"r_fit_min_m='auto' found no bins within [0, {near_edge_search_max_m:.2f}m] to "
                    "search -- falling back to the prior one-sided fit window (starts at r_ref_m)."
                )
            else:
                auto_search_max_m = float(near_edge_search_max_m)
                auto_value = auto_peak_edge_m + r_fit_min_auto_margin_m
                past = (
                    f"; the margin puts it {auto_value - auto_search_max_m:.2f}m past the search bound"
                    if auto_value > auto_search_max_m else ""
                )
                r_fit_min_m_auto_note = (
                    f"r_fit_min_m='auto' resolved to {auto_value:.2f}m (near-field peak bin's far edge "
                    f"{auto_peak_edge_m:.2f}m, peak searched among bin centers within "
                    f"[0, {auto_search_max_m:.2f}m], + {r_fit_min_auto_margin_m:.1f}m margin{past})."
                )
        r_fit_min_m = auto_value

    r_ref_m_resolution_note: str | None = None
    if isinstance(r_ref_m, str):
        if r_ref_m == "mid_swath":
            if not usable_range_m:
                r_ref_m = None
                r_ref_m_resolution_note = (
                    "r_ref_m='mid_swath' requested but usable_range_m is unavailable -- "
                    "falling back to the near-field auto-pick instead."
                )
            else:
                r_ref_m = usable_range_m / 2.0
                r_ref_m_resolution_note = f"r_ref_m='mid_swath' resolved to {r_ref_m:.2f}m (usable_range_m / 2)."
        elif r_ref_m == "auto":
            near = r_fit_min_m if r_fit_min_m is not None else 0.0
            far = usable_range_m
            if far is None or far <= near:
                r_ref_m = None
                r_ref_m_resolution_note = (
                    "r_ref_m='auto' needs usable_range_m (and it must exceed the resolved "
                    "r_fit_min_m) to place the reference inside the fit window -- falling back "
                    "to the near-field auto-pick instead."
                )
            else:
                r_ref_m = near + r_ref_window_fraction * (far - near)
                r_ref_m_resolution_note = (
                    f"r_ref_m='auto' resolved to {r_ref_m:.2f}m ({r_ref_window_fraction:.2f} of the "
                    f"way from the fit window's near edge {near:.2f}m to its far edge {far:.2f}m)."
                )
        else:
            raise ValueError(f"fit_range_decay: r_ref_m string values must be 'auto' or 'mid_swath', got {r_ref_m!r}")

    # Safety net for a reference not derived from the window (numeric or
    # "mid_swath"); r_ref_m="auto" is inside the window by construction.
    if r_fit_min_m is not None and r_ref_m is not None and r_fit_min_m > r_ref_m:
        r_fit_min_m_auto_note = (
            f"{r_fit_min_m_auto_note} r_fit_min_m ({r_fit_min_m:.2f}m) exceeded r_ref_m "
            f"({r_ref_m:.2f}m) -- clamped down to r_ref_m so the reference stays inside the fit "
            "window (locally one-sided)."
        )
        r_fit_min_m = r_ref_m

    absorption_db_per_m = absorption_for_frequency(freq_hz)
    fit_result = fit_tvg_curve(
        binned, spreading_loss_db=spreading_loss_db, absorption_db_per_m=absorption_db_per_m,
        r_ref_m=r_ref_m, r_fit_min_m=r_fit_min_m, r_fit_max_m=usable_range_m, r_ref_search_fraction=r_ref_search_fraction, value_col=fit_value_col,
    )

    ranked = sorted(fit_result.candidates, key=lambda c: c.fixed_scale_rmse_db)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    notes = result.notes
    if r_fit_min_m_auto_note is not None:
        notes.append(r_fit_min_m_auto_note)
    if r_ref_m_resolution_note is not None:
        notes.append(r_ref_m_resolution_note)
    if second is not None and best.fixed_scale_rmse_db > 0 and (
        second.fixed_scale_rmse_db - best.fixed_scale_rmse_db
    ) / best.fixed_scale_rmse_db < ambiguous_rel_tol:
        tvg_status = "ambiguous"
        notes.append(
            f"Best ({best.name}, RMSE={best.fixed_scale_rmse_db:.2f}dB) not clearly separated from "
            f"runner-up ({second.name}, RMSE={second.fixed_scale_rmse_db:.2f}dB)."
        )
    else:
        tvg_status = _TVG_CANDIDATE_TO_STATUS[best.name]
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

    result.tvg_status = tvg_status
    result.absorption_db_per_m = absorption_db_per_m
    result.r_ref_m = best.r_ref_m
    result.r_ref_m_override = r_ref_m_requested
    result.r_fit_min_m = best.r_fit_min_m
    result.r_fit_min_m_override = r_fit_min_m_requested
    result.r_fit_min_auto_margin_m = r_fit_min_auto_margin_m if r_fit_min_m_requested == "auto" else None
    result.r_fit_min_auto_search_max_m = auto_search_max_m
    result.r_fit_min_auto_peak_edge_m = auto_peak_edge_m
    result.r_fit_max_m = best.r_fit_max_m
    result.n_bins_used = best.n_bins_used
    result.best_candidate = best.name
    result.best_scale = best.scale
    result.best_rmse_db = best.fixed_scale_rmse_db
    result.best_free_fit_scale = best.scale
    result.best_free_fit_rmse_db = best.rmse_db
    result.ambiguous_rel_tol = ambiguous_rel_tol
    result.candidates = {
        c.name: {"scale": c.scale, "rmse_db": c.rmse_db, "fixed_scale_rmse_db": c.fixed_scale_rmse_db}
        for c in fit_result.candidates
    }
    result.fit = fit_result
    return result
