"""
Plotting functions for the investigation pipeline
(`sidescanpy.analysis.investigation`).

One function per stage's qualitative output. Consumes each stage's
`StageResult` (metrics + the `data` scratch dict stage 3/4 populate)
rather than recomputing anything — follows `viz/tvg_diagnostics.py`'s
"pure presentation over pre-computed analysis output" split, and reuses
`viz/waterfall.py: plot_waterfall` / `viz/tvg_diagnostics.py:
plot_candidate_fit` directly for the two stages whose plots already
exist in this library.

2026 rshomberg@uri.edu
"""

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from sidescanpy.analysis.investigation import StageResult
from sidescanpy.processing.radiometric import tvg_gain_db
from sidescanpy.viz.waterfall import plot_waterfall


def _new_fig_ax(ax: Axes | None, **subplots_kwargs) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(**subplots_kwargs)
    else:
        fig = ax.figure
    return fig, ax


# ============================================================
# STAGE 1 — BASIC CHARACTERIZATION
# ============================================================


def plot_stage1_histograms(
    ds: xr.Dataset, value_var: str = "backscatter_raw", *, bins: int = 100
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Dual-panel histogram: linear scale (left) and log10(|x|) scale (right)."""
    arr = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4))
    ax_lin.hist(finite, bins=bins, color="C0")
    ax_lin.set_xlabel(value_var)
    ax_lin.set_ylabel("Count")
    ax_lin.set_title("Linear scale")

    nonzero = finite[finite != 0]
    log_vals = np.log10(np.abs(nonzero)) if nonzero.size else np.array([])
    ax_log.hist(log_vals, bins=bins, color="C1")
    ax_log.set_xlabel(f"log10(|{value_var}|), zeros excluded")
    ax_log.set_ylabel("Count")
    ax_log.set_title("Log scale")

    fig.suptitle(f"Stage 1 — value distribution ({value_var})")
    fig.tight_layout()
    return fig, (ax_lin, ax_log)


def plot_stage1_sample_ping(
    ds: xr.Dataset, value_var: str = "backscatter_raw", *, ping_idx: int | None = None
) -> tuple[Figure, tuple[Axes, Axes]]:
    """One representative ping, port and starboard side by side."""
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2

    fig, (ax_port, ax_stbd) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ch, label in ((ax_port, 0, "port"), (ax_stbd, 1, "starboard")):
        y = np.asarray(ds[value_var].isel(ping=ping_idx, channel=ch).values, dtype=np.float64)
        ax.plot(np.arange(y.size), y, linewidth=0.7)
        ax.set_xlabel("Sample index")
        ax.set_title(label)
    ax_port.set_ylabel(value_var)
    fig.suptitle(f"Stage 1 — sample ping {ping_idx}")
    fig.tight_layout()
    return fig, (ax_port, ax_stbd)


# ============================================================
# STAGE 1.5 — OUTLIER COMPARISON
# ============================================================


def plot_stage1_5_before_after(
    ds: xr.Dataset,
    stage1_5: StageResult,
    value_var: str = "backscatter_raw",
    *,
    bins: int = 100,
    linear_xlim_percentile: float = 99.5,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Overlaid before/after histograms (linear + log scale) showing
    exactly what `reject_range_outliers` discarded from `value_var`.

    Both panels use one shared set of bin edges per panel (computed
    once, reused for both the "before" and "after" call) — matplotlib's
    `hist()` picks its own edges per call when given an integer bin
    count, so two independent calls compare different-width bins
    against each other, which silently misrepresents the overlay.

    The linear panel's x-axis is deliberately capped at the `before`
    population's `linear_xlim_percentile` (default 99.5th) rather than
    its true max: a handful of extreme values (exactly what this
    comparison is about) can otherwise stretch the axis wide enough
    that the entire bulk distribution — before and after — collapses
    into an unreadable sliver at the left edge. Values beyond the cap
    are still counted (not dropped from the data), just off-screen;
    the plot title reports how many.
    """
    filtered_arr = stage1_5.data.get("filtered_array")

    original = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
    original_finite = original[np.isfinite(original)]

    filtered_finite = np.array([])
    if filtered_arr is not None:
        filtered_finite = np.asarray(filtered_arr, dtype=np.float64).ravel()
        filtered_finite = filtered_finite[np.isfinite(filtered_finite)]

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4))

    # ----- linear panel: shared edges, capped view -----
    xmax = float(np.percentile(original_finite, linear_xlim_percentile)) if original_finite.size else 1.0
    xmax = max(xmax, 1e-9)
    edges_lin = np.linspace(0.0, xmax, bins + 1)
    n_offscale = int(np.sum(original_finite > xmax))

    ax_lin.hist(original_finite, bins=edges_lin, alpha=0.55, color="C0", label="before")
    if filtered_finite.size:
        ax_lin.hist(filtered_finite, bins=edges_lin, alpha=0.55, color="C3", label="after (outliers removed)")
    ax_lin.set_xlim(0.0, xmax)
    if n_offscale:
        ax_lin.text(
            0.98, 0.95, f"{n_offscale:,} before-values above view\n(up to {original_finite.max():.0f})",
            transform=ax_lin.transAxes, ha="right", va="top", fontsize=7.5, color="C0",
        )

    # ----- log panel: shared edges over the full (wider) before range -----
    nonzero_before = original_finite[original_finite != 0]
    log_before = np.log10(np.abs(nonzero_before)) if nonzero_before.size else np.array([])
    nonzero_after = filtered_finite[filtered_finite != 0]
    log_after = np.log10(np.abs(nonzero_after)) if nonzero_after.size else np.array([])

    if log_before.size:
        lo, hi = log_before.min(), log_before.max()
        if log_after.size:
            lo, hi = min(lo, log_after.min()), max(hi, log_after.max())
        edges_log = np.linspace(lo, hi, bins + 1)
        ax_log.hist(log_before, bins=edges_log, alpha=0.55, color="C0", label="before")
        if log_after.size:
            ax_log.hist(log_after, bins=edges_log, alpha=0.55, color="C3", label="after (outliers removed)")

    ax_lin.set_xlabel(value_var)
    ax_lin.set_ylabel("Count")
    ax_lin.set_title(f"Linear scale (capped at p{linear_xlim_percentile:g})")
    ax_lin.legend(fontsize=8)

    ax_log.set_xlabel(f"log10(|{value_var}|), zeros excluded")
    ax_log.set_ylabel("Count")
    ax_log.set_title("Log scale")
    ax_log.legend(fontsize=8)

    n_rejected = stage1_5.metrics.get("outlier_n_rejected")
    pct_rejected = stage1_5.metrics.get("outlier_pct_rejected")
    suffix = f" — {n_rejected} rejected ({pct_rejected:.2f}%)" if n_rejected is not None else ""
    fig.suptitle(f"Stage 1.5 — distribution before/after outlier rejection ({value_var}){suffix}")
    fig.tight_layout()
    return fig, (ax_lin, ax_log)


# ============================================================
# STAGE 2 — NADIR ANALYSIS
# ============================================================


def _cross_track_profile(ds: xr.Dataset, value_var: str, ping_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed sample-index x, value y: port mirrored negative, starboard
    positive, matching `viz.waterfall`'s "sample" x-axis convention."""
    port = np.asarray(ds[value_var].isel(ping=ping_idx, channel=0).values, dtype=np.float64)
    stbd = np.asarray(ds[value_var].isel(ping=ping_idx, channel=1).values, dtype=np.float64)
    x_port = -np.arange(port.size)[::-1]
    x_stbd = np.arange(stbd.size)
    x = np.concatenate([x_port, x_stbd])
    y = np.concatenate([port[::-1], stbd])
    return x, y


def plot_stage2_cross_track_profile(
    ds: xr.Dataset, value_var: str = "backscatter_raw", *, ping_idx: int | None = None, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2
    fig, ax = _new_fig_ax(ax, figsize=(10, 4))

    x, y = _cross_track_profile(ds, value_var, ping_idx)
    ax.plot(x, y, linewidth=0.7)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Sample index (port ← | → starboard)")
    ax.set_ylabel(value_var)
    ax.set_title(f"Stage 2 — cross-track profile, ping {ping_idx}")
    return fig, ax


def plot_stage2_nadir_zoom(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    *,
    stage2: StageResult | None = None,
    nadir_n_samples: int = 50,
    ping_idx: int | None = None,
    zoom_factor: int = 2,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Cross-track profile around nadir, zoomed to actually include the
    water-column-to-bottom transition, not just the narrow acquisition
    blanking gap.

    Without `stage2` (a `stage2_nadir()` result), this only has
    `nadir_n_samples` to go on and zooms to `nadir_n_samples *
    zoom_factor` — usually far too narrow to reach the real bottom
    return. Pass `stage2` so the window widens to actually include the
    detected boundaries and both get marked: the **geometric** boundary
    (dotted, thin — where the beam mathematically first reaches
    flat-bottom range) and the **strong-return** boundary (solid, bold
    — where backscatter actually rises to a real echo; real-data check
    found these can differ by 2x+, see `stage2_nadir`'s docstring). The
    region between them is shaded separately from the true water column
    — real, physically genuine near-grazing-incidence signal, just too
    weak to treat as flat-bottom for curve fitting.
    """
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2
    fig, ax = _new_fig_ax(ax, figsize=(9, 4.5))

    x, y = _cross_track_profile(ds, value_var, ping_idx)

    window = nadir_n_samples * zoom_factor
    geo_p90 = None
    strong_p90 = None
    if stage2 is not None:
        geo_p90 = stage2.metrics.get("geometric_bottom_contact_sample_p90")
        strong_p90 = stage2.metrics.get("first_strong_return_sample_p90")
        widen_to = strong_p90 if strong_p90 is not None else geo_p90
        if widen_to is not None:
            window = max(window, int(widen_to * 1.3))

    mask = np.abs(x) <= window
    ax.plot(x[mask], y[mask], marker="o", markersize=2.5, linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.axvspan(-nadir_n_samples, nadir_n_samples, color="orange", alpha=0.15, label="nadir gap window")

    if geo_p90 is not None or strong_p90 is not None:
        first_bottom = stage2.data.get("first_bottom_sample")
        first_strong = stage2.data.get("first_strong_return_sample")
        geo_this = first_bottom[ping_idx, 0] if first_bottom is not None and ping_idx < first_bottom.shape[0] else np.nan
        strong_this = first_strong[ping_idx, 0] if first_strong is not None and ping_idx < first_strong.shape[0] else np.nan
        geo_b = geo_this if np.isfinite(geo_this) else geo_p90
        strong_b = strong_this if np.isfinite(strong_this) else strong_p90

        if geo_b is not None:
            for sign, label in ((-1, None), (1, "geometric bottom contact")):
                ax.axvline(sign * geo_b, color="#8B8B3A", linestyle=":", linewidth=1.3, label=label)
        if strong_b is not None:
            for sign, label in ((-1, None), (1, "first strong return")):
                ax.axvline(sign * strong_b, color="#2E7D74", linestyle="--", linewidth=1.6, label=label)
        if geo_b is not None and strong_b is not None and strong_b > geo_b:
            ax.axvspan(-strong_b, -geo_b, color="#B5482E", alpha=0.08)
            ax.axvspan(geo_b, strong_b, color="#B5482E", alpha=0.08, label="weak near-grazing zone")

    ax.set_xlabel("Sample index (port ← | → starboard)")
    ax.set_ylabel(value_var)
    ax.set_title(f"Stage 2 — nadir zoom, ping {ping_idx}")
    ax.legend(fontsize=7.5, loc="upper right")
    return fig, ax


def plot_stage2_waterfall(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    *,
    stage2: StageResult | None = None,
    ax: Axes | None = None,
    **waterfall_kwargs,
) -> tuple[Figure, Axes]:
    """Full-survey cross-track-vs-ping waterfall of `value_var`.

    Reuses `viz.waterfall.plot_waterfall` directly rather than
    reimplementing a pcolormesh — that function reads a fixed
    `backscatter_intensity` variable name (schema convention), so
    `value_var` is aliased onto that name on a shallow copy first.

    Pass `stage2` (a `stage2_nadir()` result) to overlay both detected
    boundaries as traced lines on both sides — the most direct way to
    see whether either tracks the real signal structure across the
    whole survey, not just one ping. The **strong-return** boundary
    (solid) is the one that matters for excluding weak/water-column
    samples from curve fitting; the **geometric** boundary (dotted) is
    shown too since real data can show a large, informative gap
    between the two (see `stage2_nadir`'s docstring).
    """
    fig, ax = _new_fig_ax(ax, figsize=(12, 6))
    aliased = ds.assign(backscatter_intensity=ds[value_var])
    plot_waterfall(aliased, x_axis="sample", ax=ax, title=f"Stage 2 — waterfall ({value_var})", **waterfall_kwargs)

    if stage2 is not None:
        first_bottom = stage2.data.get("first_bottom_sample")
        first_strong = stage2.data.get("first_strong_return_sample")

        def _trace(boundary, color, style, label, lw):
            n_pings = boundary.shape[0]
            y = np.arange(n_pings)
            port_b, stbd_b = boundary[:, 0], boundary[:, 1] if boundary.shape[1] > 1 else boundary[:, 0]
            valid_port, valid_stbd = np.isfinite(port_b), np.isfinite(stbd_b)
            ax.plot(-port_b[valid_port], y[valid_port], color=color, linestyle=style, linewidth=lw, alpha=0.85)
            ax.plot(stbd_b[valid_stbd], y[valid_stbd], color=color, linestyle=style, linewidth=lw, alpha=0.85, label=label)

        if first_bottom is not None:
            _trace(first_bottom, "#8B8B3A", ":", "geometric bottom contact", 1.0)
        if first_strong is not None:
            _trace(first_strong, "#2E7D74", "-", "first strong return", 1.3)
        if first_bottom is not None or first_strong is not None:
            ax.legend(fontsize=8, loc="upper right")

    return fig, ax


# ============================================================
# STAGE 2.5 — BEAM GEOMETRY
# ============================================================

_CANDIDATE_COLORS = ("#7A7A7A", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860")


def plot_stage2_5_beam_geometry(
    ds: xr.Dataset,
    stage2_5: StageResult,
    value_var: str = "backscatter_raw",
    range_var: str = "across_track_m",
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Axial (side-view) beam-geometry diagram directly above a
    ground-range single-ping profile, sharing one x-axis, so candidate
    beam-tilt angles can be checked by eye against where the real
    signal actually starts.

    Top panel: towfish, flat seafloor at the ping's own altitude, one
    straight ray per candidate angle (from `stage2_5_beam_geometry`),
    and the angle implied by the data itself (thicker, dashed).
    Bottom panel: the actual ping (`value_var` vs. `range_var`), with
    the same candidate positions as vertical lines and the empirically
    observed first-strong-return point marked, so a candidate that
    lines up with where the signal actually starts is directly visible
    rather than argued from a table of numbers.
    """
    ping_idx = stage2_5.metrics.get("ping_idx", 0)
    altitude = stage2_5.metrics.get("altitude_m")
    candidates = stage2_5.metrics.get("candidates", [])
    implied = stage2_5.metrics.get("implied", {})

    fig, (ax_geo, ax_ping) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )

    if altitude is None or not candidates:
        ax_geo.set_title(f"Stage 2.5 — beam geometry unavailable (ping {ping_idx})")
        return fig, (ax_geo, ax_ping)

    max_reach = max(c["predicted_across_track_m"] for c in candidates)
    for entry in implied.values():
        across = entry.get("observed_across_track_m")
        if across is not None:
            max_reach = max(max_reach, abs(across))
    max_reach = max(max_reach, altitude) * 1.15

    # ----- top panel: axial geometry -----
    ax_geo.plot(0, 0, marker="v", markersize=14, color="#142029", zorder=5)
    ax_geo.text(0, 0.35, "towfish", ha="center", fontsize=8, color="#142029")
    ax_geo.axhline(-altitude, color="#8B5A2B", linewidth=1.6)
    ax_geo.text(max_reach * 0.98, -altitude, f"seafloor (altitude={altitude:.2f}m)",
                ha="right", va="bottom", fontsize=8, color="#8B5A2B")
    ax_geo.axvline(0, color="k", linewidth=0.5, alpha=0.4, linestyle=":")

    for i, c in enumerate(candidates):
        color = _CANDIDATE_COLORS[i % len(_CANDIDATE_COLORS)]
        x = c["predicted_across_track_m"]
        label = f"{c['angle_from_vertical_deg']:.0f}° from vertical ({c['angle_from_horizontal_deg']:.0f}° from horiz.)"
        for sign in (-1, 1):
            ax_geo.plot([0, sign * x], [0, -altitude], color=color, linewidth=1.3, alpha=0.85,
                        label=label if sign == -1 else None)

    for label, entry in implied.items():
        angle = entry.get("angle_from_vertical_deg")
        across = entry.get("observed_across_track_m")
        if angle is not None and across is not None:
            ax_geo.plot([0, across], [0, -altitude], color="#B5482E", linewidth=2.4, linestyle="--",
                        alpha=0.9, label=f"implied by data ({label}): {angle:.0f}° from vertical" if label == "port" else None)
            ax_geo.plot(across, -altitude, marker="o", markersize=6, color="#B5482E", zorder=6)

    ax_geo.set_ylim(-altitude * 1.3, altitude * 0.6)
    ax_geo.set_xlim(-max_reach, max_reach)
    ax_geo.set_ylabel("Depth below towfish (m)")
    ax_geo.set_title(f"Stage 2.5 — beam geometry, ping {ping_idx} (straight-angle candidates)")
    ax_geo.legend(fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.55))

    # ----- bottom panel: real ping profile on the same x-axis -----
    port = np.asarray(ds[value_var].isel(ping=ping_idx, channel=0).values, dtype=np.float64)
    stbd = np.asarray(ds[value_var].isel(ping=ping_idx, channel=1).values, dtype=np.float64)
    x_port = np.asarray(ds[range_var].isel(ping=ping_idx, channel=0).values, dtype=np.float64)
    x_stbd = np.asarray(ds[range_var].isel(ping=ping_idx, channel=1).values, dtype=np.float64)

    for x, y in ((x_port, port), (x_stbd, stbd)):
        valid = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[valid])
        ax_ping.plot(x[valid][order], y[valid][order], color="#2E4057", linewidth=0.7, alpha=0.9)

    for i, c in enumerate(candidates):
        color = _CANDIDATE_COLORS[i % len(_CANDIDATE_COLORS)]
        for sign in (-1, 1):
            ax_ping.axvline(sign * c["predicted_across_track_m"], color=color, linewidth=1.0, alpha=0.7)

    for entry in implied.values():
        across = entry.get("observed_across_track_m")
        if across is not None:
            ax_ping.axvline(across, color="#B5482E", linewidth=1.8, linestyle="--", alpha=0.9)

    ax_ping.set_xlabel("Across-track distance (m)  (port ← | → starboard)")
    ax_ping.set_ylabel(value_var)
    ax_ping.set_title("Same ping, real backscatter (candidate positions marked)")

    fig.tight_layout()
    return fig, (ax_geo, ax_ping)


# ============================================================
# STAGE 3 — USABLE RANGE
# ============================================================


def plot_stage3_usable_range(stage3: StageResult) -> tuple[Figure, tuple[Axes, Axes]]:
    """Median and std-dev vs. range, with the detected usable-range
    boundary marked on both panels."""
    binned = stage3.data.get("binned")
    fig, (ax_med, ax_std) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if binned is None or binned.empty:
        ax_med.set_title("Stage 3 — no binned data")
        return fig, (ax_med, ax_std)

    x = (binned["range_min_m"] + binned["range_max_m"]) / 2.0
    ax_med.plot(x, binned["median"], marker="o", markersize=3)
    ax_med.fill_between(x, binned["p10"], binned["p90"], alpha=0.2)
    ax_med.set_ylabel("Median backscatter")
    ax_med.set_title("Stage 3 — usable range")

    ax_std.plot(x, binned["std"], marker="o", markersize=3, color="C1")
    ax_std.set_ylabel("Std dev")
    ax_std.set_xlabel("Range (m)")

    usable_range_m = stage3.metrics.get("usable_range_m")
    if usable_range_m is not None:
        for a in (ax_med, ax_std):
            a.axvline(usable_range_m, color="red", linestyle="--", linewidth=1, label=f"usable range = {usable_range_m:.1f}m")
            a.legend(fontsize=8)

    noise_floor = stage3.metrics.get("noise_floor")
    if noise_floor is not None:
        ax_med.axhline(noise_floor, color="gray", linestyle=":", linewidth=1)

    fig.tight_layout()
    return fig, (ax_med, ax_std)


# ============================================================
# STAGE 4 — DECAY ANALYSIS
# ============================================================


def plot_stage4_decay_fit(stage4: StageResult) -> tuple[Figure, Axes]:
    """Observed loss-vs-range curve with all four candidate models overlaid.

    Thin wrapper around `viz.tvg_diagnostics.plot_candidate_fit`, called
    once per candidate on a shared axis, per that function's own
    documented calling convention.
    """
    from sidescanpy.viz.tvg_diagnostics import plot_candidate_fit

    fit_result = stage4.data.get("fit_result")
    fig, ax = plt.subplots(figsize=(9, 6))
    if fit_result is None:
        ax.set_title("Stage 4 — no fit available")
        return fig, ax

    colors = {"flat": "C0", "spreading_only": "C1", "absorption_only": "C2", "full_physics": "C3"}
    for i, candidate in enumerate(fit_result.candidates):
        plot_candidate_fit(
            candidate,
            fit_result.observed_range_m,
            fit_result.observed_loss_db,
            ax=ax,
            color=colors.get(candidate.name),
            show_observed=(i == 0),
            title=f"Stage 4 — decay model fit (best: {stage4.metrics.get('best_candidate')})",
        )
    return fig, ax


def plot_stage4_residuals(stage4: StageResult) -> tuple[Figure, Axes]:
    """Residual (model - observed) vs. range for each candidate's fit
    window, using the same model-curve construction as
    `plot_candidate_fit` (recomputed here since `TVGCandidateFit` only
    stores the summary RMSE, not the per-bin residual array)."""
    fit_result = stage4.data.get("fit_result")
    fig, ax = plt.subplots(figsize=(9, 5))
    if fit_result is None:
        ax.set_title("Stage 4 — no fit available")
        return fig, ax

    ranges = fit_result.observed_range_m
    observed = fit_result.observed_loss_db
    colors = {"flat": "C0", "spreading_only": "C1", "absorption_only": "C2", "full_physics": "C3"}

    for candidate in fit_result.candidates:
        model_curve = tvg_gain_db(np.clip(ranges, candidate.params.min_range_m, None), candidate.params)
        i_ref = int(np.argmin(np.abs(ranges - candidate.r_ref_m)))
        model_loss = candidate.scale * (model_curve - model_curve[i_ref])
        residual = model_loss - observed
        in_window = (ranges >= candidate.r_ref_m) & (ranges <= candidate.r_fit_max_m)
        # Same reference-relative x-axis as plot_stage4_decay_fit — see
        # that function's (plot_candidate_fit's) comment for why.
        rel_range_m = ranges - candidate.r_ref_m
        ax.plot(
            rel_range_m[in_window], residual[in_window],
            color=colors.get(candidate.name), marker=".", markersize=4,
            label=f"{candidate.name} (RMSE={candidate.rmse_db:.2f}dB)",
        )

    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Distance from reference range (m)")
    ax.set_ylabel("Residual, model - observed (dB)")
    ax.set_title("Stage 4 — fit residuals")
    ax.legend(fontsize=8)
    return fig, ax


# ============================================================
# STAGE 5 — UNIT IDENTIFICATION
# ============================================================


def plot_stage5_distribution(
    ds: xr.Dataset, value_var: str = "backscatter_raw", *, bins: int = 100
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Same distribution-shape view as Stage 1 (linear + log histogram) —
    shared rather than duplicated, since Stage 5's unit call is read
    directly off Stage 1's value-range signature."""
    fig, axes = plot_stage1_histograms(ds, value_var, bins=bins)
    fig.suptitle(f"Stage 5 — distribution shape ({value_var})")
    return fig, axes
