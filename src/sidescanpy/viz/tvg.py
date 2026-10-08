"""
Plots for TVG and range-decay analysis (:mod:`sidescanpy.analysis.tvg`).

Each function draws into an optional ``ax``, returns ``(fig, ax)`` and
never saves a file.

2026 rshomberg@uri.edu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sidescanpy.analysis.tvg import RangeDecayResult, TVGCandidateFit
from sidescanpy.processing.radiometric import TVGParams, tvg_gain_db

logger = logging.getLogger(__name__)


_CANDIDATE_COLORS = {"flat": "C0", "spreading_only": "C1", "absorption_only": "C2", "full_physics": "C3"}


def _new_fig_ax(ax: Axes | None, **subplots_kwargs) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(**subplots_kwargs)
    else:
        fig = ax.figure
    return fig, ax


def plot_mean_vs_range(
    binned: pd.DataFrame,
    ax: Axes | None = None,
    *,
    show_band: bool = True,
    band: tuple[str, str] = ("p10", "p90"),
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    zorder: float = 3.0,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Line plot of mean backscatter vs. range, with an optional shaded
    percentile band.

    Call twice on the same `ax` (raw vs. corrected) for a direct
    before/after overlay.

    Parameters
    ----------
    binned : pd.DataFrame
        Output of `backscatter_vs_range` (or
        `backscatter_vs_sample` — any DataFrame with a
        "mean" column and matching min/max range columns).
    show_band : bool, default True
        Shade the region between `band`'s two percentile columns.
    band : (str, str), default ("p10", "p90")
    linewidth : float, default 1.5
        Matplotlib's own default is 1.5, so this default is a no-op for
        existing callers -- pass a larger value (e.g. when overlaying
        on top of `plot_per_ping_profiles`' dense low-alpha spaghetti,
        which can otherwise visually bury a thin mean line even with a
        distinct color) to make the mean line read clearly on top.
    zorder : float, default 3.0
        Matplotlib draws in call order by default (zorder 1, 2, 3, ...
        as each `ax.plot`/`fill_between` call happens) -- 3.0 is chosen
        to sit above a typical single prior `plot_per_ping_profiles`
        call (which doesn't set its own zorder, so defaults to 1) even
        if draw order alone wouldn't guarantee it. Raise further if
        overlaying on top of more layers.
    """
    fig, ax = _new_fig_ax(ax)

    range_col = "range_min_m" if "range_min_m" in binned.columns else "sample_min"
    max_col = "range_max_m" if "range_max_m" in binned.columns else "sample_max"
    x = (binned[range_col] + binned[max_col]) / 2.0

    line, = ax.plot(
        x, binned["mean"], label=label, color=color, marker="o", markersize=3,
        linewidth=linewidth, zorder=zorder,
    )
    if show_band and band[0] in binned.columns and band[1] in binned.columns:
        ax.fill_between(
            x, binned[band[0]], binned[band[1]], alpha=0.2, color=line.get_color(), zorder=zorder - 0.5,
        )

    ax.set_xlabel("Range (m)" if range_col == "range_min_m" else "Sample index")
    ax.set_ylabel("Backscatter (dB)")
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    return fig, ax


def plot_tvg_gain_curve(
    ds: xr.Dataset,
    ax: Axes | None = None,
    *,
    range_var: str = "ground_range_m",
    title: str | None = "TVG gain vs. range",
) -> tuple[Figure, Axes]:
    """
    Plot `tvg_gain_db` vs. range from `compare_before_after`'s output.

    Shows how much gain TVG actually added at each range — useful
    alongside `plot_mean_vs_range` to compare against the observed
    residual's shape.
    """
    fig, ax = _new_fig_ax(ax)

    ranges = np.asarray(ds[range_var].values, dtype=np.float64).ravel()
    gain = np.asarray(ds["tvg_gain_db"].values, dtype=np.float64).ravel()
    valid = np.isfinite(ranges) & np.isfinite(gain)
    order = np.argsort(ranges[valid])

    ax.plot(ranges[valid][order], gain[valid][order], ".", markersize=1, alpha=0.3)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("TVG gain (dB)")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_per_ping_profiles(
    ds: xr.Dataset,
    value_var: str,
    range_var: str = "ground_range_m",
    n_pings: int = 50,
    ax: Axes | None = None,
    alpha: float = 0.08,
    *,
    channel_idx: int = 0,
    seed: int = 0,
    title: str | None = None,
    x_sign: float = 1.0,
) -> tuple[Figure, Axes]:
    """
    "Spaghetti plot": overlay `n_pings` individual ping profiles
    (`value_var` vs. `range_var`, one channel) at low alpha.

    Visual density shows ping-to-ping consistency/variance directly —
    complementary to `plot_mean_vs_range`'s percentile band, which
    summarizes spread but averages away periodic per-ping artifacts.

    Parameters
    ----------
    channel_idx : int, default 0
        Which channel to plot (0 = port, 1 = starboard per this
        schema's convention) — call twice with `channel_idx=0` and
        `channel_idx=1` on separate axes for a side-by-side port/
        starboard comparison. Previously hardcoded to 0; the default
        preserves that exact behavior for existing callers.
    x_sign : float, default 1.0
        Multiplies every plotted `range_var` value before drawing —
        pass `-1.0` (typically for `channel_idx=0`/port) to mirror one
        channel onto the negative side of a shared axis, so calling
        this twice (once per channel, opposite signs) on the same `ax`
        draws one continuous full-swath profile instead of two
        independent one-sided ones. `range_var` itself (e.g.
        `slant_range_m`) is always unsigned magnitude — this is a
        display-only sign flip, not a different range computation.
        1.0 is a no-op, preserving every existing caller's output
        exactly.
    """
    fig, ax = _new_fig_ax(ax)

    n_total = ds.sizes["ping"]
    rng = np.random.default_rng(seed)
    n_pick = min(n_pings, n_total)
    ping_idx = rng.choice(n_total, size=n_pick, replace=False)

    for p in ping_idx:
        sel = dict(ping=p, channel=channel_idx)
        r = np.asarray(ds[range_var].isel(**sel).values, dtype=np.float64) * x_sign
        v = np.asarray(ds[value_var].isel(**sel).values, dtype=np.float64)
        valid = np.isfinite(r) & np.isfinite(v)
        order = np.argsort(r[valid])
        ax.plot(r[valid][order], v[valid][order], color="C0", alpha=alpha)

    ax.set_xlabel("Range (m)")
    ax.set_ylabel(value_var)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_multi_frequency_comparison(
    binned_by_freq: dict[str, pd.DataFrame],
    ax: Axes | None = None,
    *,
    title: str | None = "Mean backscatter vs. range, by frequency",
) -> tuple[Figure, Axes]:
    """
    Overlay `backscatter_vs_range` curves for multiple frequencies,
    color-coded by frequency.

    If multiple frequencies (different `absorption_for_frequency()`
    coefficients) show a similarly-shaped residual, that's evidence
    against "just retune absorption" (candidate 1).
    """
    fig, ax = _new_fig_ax(ax)
    for freq_label, binned in binned_by_freq.items():
        plot_mean_vs_range(binned, ax=ax, show_band=False, label=freq_label)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_altitude_bin_comparison(
    binned_by_altitude: dict[str, pd.DataFrame],
    ax: Axes | None = None,
    *,
    title: str | None = "Mean backscatter vs. range, by altitude bin",
) -> tuple[Figure, Axes]:
    """
    Overlay `backscatter_vs_range` curves for multiple altitude
    bins (from `bin_by_altitude`), color-coded by bin.

    If the residual's shape changes with altitude (different near-nadir
    geometry), that's evidence for a missing angular correction
    (candidate 2); roughly altitude-invariant leans toward candidate 1.
    """
    fig, ax = _new_fig_ax(ax)
    for bin_label, binned in binned_by_altitude.items():
        plot_mean_vs_range(binned, ax=ax, show_band=False, label=bin_label)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax


def _params_label(params: TVGParams) -> str:
    """Human-readable summary of a candidate's literal, uncalibrated
    coefficients -- e.g. "spreading=40.0dB, absorption=0.20dB/m" -- for
    `plot_candidate_fit`'s legend. Only non-zero terms are listed
    (`flat` has neither, so this returns an empty string for it)."""
    parts = []
    if params.spreading_loss_db:
        parts.append(f"spreading={params.spreading_loss_db:.1f}dB")
    if params.absorption_db_per_m:
        parts.append(f"absorption={params.absorption_db_per_m:.2f}dB/m")
    return ", ".join(parts)


def plot_candidate_fit(
    fit: TVGCandidateFit,
    observed_range_m: np.ndarray,
    observed_loss_db: np.ndarray,
    ax: Axes | None = None,
    *,
    color: str | None = None,
    show_observed: bool = True,
    title: str | None = None,
    use_fixed_scale: bool = True,
    model_n_points: int = 200,
) -> tuple[Figure, Axes]:
    """
    Overlay one `analysis.tvg.fit_tvg_curve()`
    candidate's fitted model against the observed raw-loss curve it was
    fit against.

    Call once per candidate on the same `ax` (`result.candidates` from
    `TVGCandidateFitResult`) — `show_observed=True` on the first call
    only, `False` on the rest, so the observed curve is drawn once and
    each candidate's fitted line overlays it. Mirrors
    `plot_mean_vs_range`'s existing "call twice for before/after" idiom,
    generalized to "call once per candidate."

    Parameters
    ----------
    fit : TVGCandidateFit
        One candidate from `TVGCandidateFitResult.candidates` (or `.best`).
    observed_range_m, observed_loss_db : np.ndarray
        `TVGCandidateFitResult.observed_range_m`/`.observed_loss_db` —
        the same arrays for every candidate fit from one
        `fit_tvg_curve()` call, so passing a different
        candidate on repeated calls only changes the overlaid line.
    show_observed : bool, default True
        Plot the observed curve itself. Set `False` on repeat calls
        against the same `ax` to avoid redrawing it once per candidate.
    use_fixed_scale : bool, default True
        Draw the literal, uncalibrated model (scale pinned to 1.0) --
        matches what `fit_range_decay` actually ranks candidates by
        (2026-08-24 change; see `TVGCandidateFit`'s docstring). `False`
        draws the old free-fit-scaled line instead, for diagnostic
        comparison against the fixed-scale one -- e.g. to see by how
        much a candidate needed to cheat to look competitive.
    model_n_points : int, default 200
        The model line is drawn over its own dense, evenly-spaced range
        array from 0 to `observed_range_m`'s max — not just at the bin
        centers `observed_range_m` happens to have — so the curve's
        shape reads smoothly all the way back to near-zero range for
        reference, including where no observed data exists (e.g. inside
        the excluded nadir gap). Purely a display domain; the fit
        itself was never evaluated there and this draws no new
        conclusion about that span.
    """
    fig, ax = _new_fig_ax(ax)

    # x-axis is absolute range -- the y-axis (loss relative to r_ref)
    # already carries the "relative" framing; putting distance on an
    # r_ref-centered axis too made it hard to read the fit's actual
    # extent in real range, and hid where each candidate's line crosses
    # into negative relative loss (nearer than r_ref, where the model
    # predicts *less* loss than at the reference) at a glance.
    if show_observed:
        ax.plot(
            observed_range_m, observed_loss_db,
            "o", color="black", markersize=7, markerfacecolor="none",
            markeredgewidth=1.3, alpha=0.7, label="observed",
        )
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.3)
        ax.axvline(
            fit.r_ref_m, color="k", linewidth=0.8, alpha=0.4, linestyle=":",
            label=f"reference (r_ref={fit.r_ref_m:.1f}m)",
        )

    model_range_m = np.linspace(0.0, float(np.max(observed_range_m)), model_n_points)
    model_curve = tvg_gain_db(np.clip(model_range_m, fit.params.min_range_m, None), fit.params)
    ref_gain = float(tvg_gain_db(np.clip(np.array([fit.r_ref_m]), fit.params.min_range_m, None), fit.params)[0])
    plotted_scale = 1.0 if use_fixed_scale else fit.scale
    model_loss = plotted_scale * (model_curve - ref_gain)
    # Every candidate's model_loss is exactly 0 at r_ref by construction
    # (evaluated directly at r_ref_m, not snapped to a bin) and negative
    # for r < r_ref (nearer than the reference implies less loss than at
    # the reference) -- the model line now spans 0 to the full observed
    # range regardless of where real bins start, so both that crossing
    # and the curve's near-zero-range shape are visible for reference.
    rmse_shown = fit.fixed_scale_rmse_db if use_fixed_scale else fit.rmse_db
    params_label = _params_label(fit.params)
    label = f"{fit.name} (RMSE={rmse_shown:.2f}dB{', ' + params_label if params_label else ''})"
    ax.plot(model_range_m, model_loss, "--", color=color, label=label)
    # r_fit_min_m is None on a TVGCandidateFit built before that field
    # existed -- falls back to r_ref_m, the prior one-sided behavior.
    fit_min = fit.r_fit_min_m if fit.r_fit_min_m is not None else fit.r_ref_m
    fit_range_label = f"fit range ({fit_min:.1f}–{fit.r_fit_max_m:.1f}m)" if show_observed else None
    ax.axvspan(fit_min, fit.r_fit_max_m, alpha=0.05, color=color, label=fit_range_label)

    ax.set_xlabel("Range (m)")
    ax.set_ylabel(f"Loss relative to {fit.r_ref_m:.1f}m (dB)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig, ax


def plot_range_decay_fit(result: RangeDecayResult, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Observed loss-vs-range curve with every candidate model overlaid
    (one :func:`plot_candidate_fit` call per candidate on a shared axis)."""
    fit_result = result.fit
    fig, ax = _new_fig_ax(ax, figsize=(9, 6))
    if fit_result is None:
        ax.set_title("No fit available")
        return fig, ax

    for i, candidate in enumerate(fit_result.candidates):
        plot_candidate_fit(
            candidate,
            fit_result.observed_range_m,
            fit_result.observed_loss_db,
            ax=ax,
            color=_CANDIDATE_COLORS.get(candidate.name),
            show_observed=(i == 0),
            title=f"Decay model fit (best: {result.best_candidate})",
        )
    return fig, ax


def set_fit_window_limits(
    ax: Axes, result: RangeDecayResult, *, x_pad_fraction: float = 0.12,
    y_pad_fraction: float = 0.25, y_pad_min_db: float = 1.0, x_pad_min_m: float = 1.0,
) -> None:
    """Set robust axis limits on a decay-fit axis from the fit window.

    y-limits come from the observed values inside the fit window
    (``result.r_fit_min_m``..``r_fit_max_m``) only, so an out-of-window
    outlier bin cannot compress the fit curves. x-limits deliberately
    span 0 to the far edge (padded), not the fit window, so a misplaced
    reference or window stays visible. With a shared x-axis, call this on
    one panel only. No-op if there is no fit or no recorded window.
    """
    fit_result = result.fit
    if fit_result is None:
        return
    r_fit_min_m = result.r_fit_min_m
    r_fit_max_m = result.r_fit_max_m
    if r_fit_min_m is None or r_fit_max_m is None:
        return

    ranges = fit_result.observed_range_m
    losses = fit_result.observed_loss_db
    window_mask = (ranges >= r_fit_min_m) & (ranges <= r_fit_max_m)
    in_window_loss = losses[window_mask]
    if in_window_loss.size:
        y_lo, y_hi = float(in_window_loss.min()), float(in_window_loss.max())
        y_pad = max(y_pad_fraction * (y_hi - y_lo), y_pad_min_db)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    x_span = r_fit_max_m - r_fit_min_m
    # x_pad_min_m keeps a zero-width (single-bin) window from producing
    # set_xlim(x, x) and matplotlib's "identical low and high xlims" warning.
    x_pad = max(x_pad_fraction * x_span, x_pad_min_m)
    x_far_edge = max(r_fit_max_m, float(np.nanmax(ranges))) if ranges.size else r_fit_max_m
    ax.set_xlim(0.0, x_far_edge + x_pad)


def plot_range_decay_residuals(
    result: RangeDecayResult, *, use_fixed_scale: bool = True, ax: Axes | None = None, relative_x: bool = True,
) -> tuple[Figure, Axes]:
    """Residual (model - observed) vs. range within each candidate's fit window.

    ``use_fixed_scale=True`` (default) draws the scale-1.0 residual and
    labels it with ``fixed_scale_rmse_db``, the number candidates are
    ranked by; ``False`` draws the free-fit-scaled residual for
    comparison. The reference uses the nearest bin to ``r_ref_m``,
    matching ``fit_tvg_curve``'s own RMSE computation exactly.
    ``relative_x=True`` plots distance from each candidate's ``r_ref_m``;
    pass ``False`` for absolute range (to stack under
    :func:`plot_candidate_fit` on a shared x-axis).
    """
    fit_result = result.fit
    fig, ax = _new_fig_ax(ax, figsize=(9, 5))
    if fit_result is None:
        ax.set_title("No fit available")
        return fig, ax

    ranges = fit_result.observed_range_m
    observed = fit_result.observed_loss_db

    for candidate in fit_result.candidates:
        model_curve = tvg_gain_db(np.clip(ranges, candidate.params.min_range_m, None), candidate.params)
        i_ref = int(np.argmin(np.abs(ranges - candidate.r_ref_m)))
        plotted_scale = 1.0 if use_fixed_scale else candidate.scale
        model_loss = plotted_scale * (model_curve - model_curve[i_ref])
        residual = model_loss - observed
        # r_fit_min_m is None on fits recorded before that field existed.
        fit_min = candidate.r_fit_min_m if candidate.r_fit_min_m is not None else candidate.r_ref_m
        in_window = (ranges >= fit_min) & (ranges <= candidate.r_fit_max_m)
        rmse_shown = candidate.fixed_scale_rmse_db if use_fixed_scale else candidate.rmse_db
        x = ranges - candidate.r_ref_m if relative_x else ranges
        ax.plot(
            x[in_window], residual[in_window],
            color=_CANDIDATE_COLORS.get(candidate.name), marker=".", markersize=4,
            label=f"{candidate.name} (RMSE={rmse_shown:.2f}dB)",
        )

    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Distance from reference range (m)" if relative_x else "Range (m)")
    ax.set_ylabel("Residual, model - observed (dB)")
    ax.set_title("Fit residuals")
    ax.legend(fontsize=8)
    return fig, ax
