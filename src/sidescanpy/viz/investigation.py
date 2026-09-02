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

    fig.suptitle(f"Value distribution ({value_var})")
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
    fig.suptitle(f"Sample ping {ping_idx}")
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
    fig.suptitle(f"Distribution before/after outlier rejection ({value_var}){suffix}")
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
    ax.set_title(f"Cross-track profile, ping {ping_idx}")
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
    full nadir gap through to the real bottom return, not just the
    narrow acquisition blanking window.

    Without `stage2` (a `stage2_nadir()` result), this only has
    `nadir_n_samples` to go on and zooms to `nadir_n_samples *
    zoom_factor` — usually far too narrow to reach the real bottom
    return. Pass `stage2` so the window widens to actually include the
    detected boundaries and both get marked: **altitude** (dotted,
    thin — the vehicle's own altitude re-expressed in sample units, not
    a real boundary) and the **first bottom return** (solid, bold —
    where backscatter actually rises to a real echo; real-data check
    found these can differ by 2x+, see `stage2_nadir`'s docstring).

    Everything from nadir (x=0) out to the first bottom return — the
    acquisition-blanking burst *and* the beam-geometry null zone beyond
    it — is shaded as one continuous span labeled "water column
    returns": the beam's near edge is never at 0 deg from vertical, so
    nothing in this span is a real bottom echo, whether the sample was
    electronically blanked or simply never illuminated. Not two
    separately-colored regions — that distinction mattered while this
    boundary was still being debugged, but both are the same thing from
    a "should this be treated as bottom signal" standpoint.
    """
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2
    fig, ax = _new_fig_ax(ax, figsize=(9, 4.5))

    x, y = _cross_track_profile(ds, value_var, ping_idx)

    window = nadir_n_samples * zoom_factor
    altitude_p90 = None
    bottom_p90 = None
    if stage2 is not None:
        altitude_p90 = stage2.metrics.get("altitude_sample_p90")
        bottom_p90 = stage2.metrics.get("first_bottom_return_sample_p90")
        widen_to = bottom_p90 if bottom_p90 is not None else altitude_p90
        if widen_to is not None:
            window = max(window, int(widen_to * 1.3))

    mask = np.abs(x) <= window
    ax.plot(x[mask], y[mask], marker="o", markersize=2.5, linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)

    altitude_b = None
    bottom_b = None
    if altitude_p90 is not None or bottom_p90 is not None:
        altitude_arr = stage2.data.get("altitude_sample")
        bottom_arr = stage2.data.get("first_bottom_return_sample")
        altitude_this = altitude_arr[ping_idx, 0] if altitude_arr is not None and ping_idx < altitude_arr.shape[0] else np.nan
        bottom_this = bottom_arr[ping_idx, 0] if bottom_arr is not None and ping_idx < bottom_arr.shape[0] else np.nan
        altitude_b = altitude_this if np.isfinite(altitude_this) else altitude_p90
        bottom_b = bottom_this if np.isfinite(bottom_this) else bottom_p90

    if bottom_b is not None:
        ax.axvspan(-bottom_b, bottom_b, color="#B5482E", alpha=0.08, label="water column returns")
    else:
        # No stage2 boundary available -- fall back to the raw blanking
        # window alone, the only nadir-gap information available.
        ax.axvspan(-nadir_n_samples, nadir_n_samples, color="#B5482E", alpha=0.08, label="water column returns")

    if altitude_b is not None:
        for sign, label in ((-1, None), (1, "altitude")):
            ax.axvline(sign * altitude_b, color="#8B8B3A", linestyle=":", linewidth=1.3, label=label)
    if bottom_b is not None:
        for sign, label in ((-1, None), (1, "first bottom return")):
            ax.axvline(sign * bottom_b, color="#2E7D74", linestyle="--", linewidth=1.6, label=label)

    ax.set_xlabel("Sample index (port ← | → starboard)")
    ax.set_ylabel(value_var)
    ax.set_title(f"Nadir zoom, ping {ping_idx}")
    ax.legend(fontsize=7.5, loc="upper right")
    return fig, ax


def plot_stage2_waterfall(
    ds: xr.Dataset,
    value_var: str = "backscatter_raw",
    *,
    stage2: StageResult | None = None,
    ax: Axes | None = None,
    ping_offset: int = 0,
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
    whole survey, not just one ping. The **first bottom return** (solid)
    is the one that matters for excluding the nadir gap from curve
    fitting; **altitude** (dotted) is shown too since real data can show
    a large, informative gap between the two (see `stage2_nadir`'s
    docstring).

    ping_offset : int, default 0
        Added to both the waterfall's own y-axis (forwarded to
        `plot_waterfall`) and the traced boundary lines' y-values, so a
        caller plotting an already-sliced window (e.g. one section of
        a larger survey) can show real, absolute ping numbers instead
        of a 0-based index local to the slice. 0 is a no-op, preserving
        every existing caller's exact output -- before this parameter
        existed, the traced lines were always drawn at a 0-based
        `np.arange(n_pings)` regardless of what `ping_offset` the
        underlying waterfall image itself might already have been given
        via `waterfall_kwargs`, which would have silently misaligned
        the trace against the image the moment a caller tried that.
    """
    fig, ax = _new_fig_ax(ax, figsize=(12, 6))
    aliased = ds.assign(backscatter_intensity=ds[value_var])
    plot_waterfall(
        aliased, x_axis="sample", ax=ax, title=f"Waterfall ({value_var})",
        ping_offset=ping_offset, **waterfall_kwargs,
    )

    if stage2 is not None:
        altitude_arr = stage2.data.get("altitude_sample")
        bottom_arr = stage2.data.get("first_bottom_return_sample")

        def _trace(boundary, color, style, label, lw):
            n_pings = boundary.shape[0]
            y = np.arange(n_pings) + ping_offset
            port_b, stbd_b = boundary[:, 0], boundary[:, 1] if boundary.shape[1] > 1 else boundary[:, 0]
            valid_port, valid_stbd = np.isfinite(port_b), np.isfinite(stbd_b)
            ax.plot(-port_b[valid_port], y[valid_port], color=color, linestyle=style, linewidth=lw, alpha=0.85)
            ax.plot(stbd_b[valid_stbd], y[valid_stbd], color=color, linestyle=style, linewidth=lw, alpha=0.85, label=label)

        if altitude_arr is not None:
            _trace(altitude_arr, "#8B8B3A", ":", "altitude", 1.0)
        if bottom_arr is not None:
            _trace(bottom_arr, "#2E7D74", "-", "first bottom return", 1.3)
        if altitude_arr is not None or bottom_arr is not None:
            ax.legend(fontsize=8, loc="upper right")

    return fig, ax


# ============================================================
# STAGE 2.5 — BEAM GEOMETRY
# ============================================================

def plot_stage2_5_beam_geometry(
    ds: xr.Dataset,
    stage2_5: StageResult,
    value_var: str = "backscatter_raw",
    range_var: str = "across_track_m",
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Axial (side-view) beam-geometry diagram directly above a
    ground-range single-ping profile, sharing one x-axis: the beam's
    real near edge (from the first bottom return) and far edge (from
    Stage 3's usable range), triangulated against altitude, not a set
    of arbitrary candidate angles.

    Top panel: towfish, flat seafloor at the ping's own altitude, the
    near ray (solid — the beam's own lower edge, empirically measured)
    and the far ray (dashed — where Stage 3 found the signal becomes
    unusable), with the wedge between them shaded as the beam's actual
    illuminated span. Bottom panel: the same ping's real backscatter
    (`value_var` vs. `range_var`), with the same two boundaries marked
    as vertical lines, so the geometry can be checked directly against
    where the real signal starts and ends.
    """
    ping_idx = stage2_5.metrics.get("ping_idx", 0)
    altitude = stage2_5.metrics.get("altitude_m")
    near_ray = stage2_5.metrics.get("near_ray", {})
    far_ray = stage2_5.metrics.get("far_ray", {})
    beam_center = stage2_5.metrics.get("beam_center_angle_deg")
    beam_width = stage2_5.metrics.get("beam_width_deg")

    fig, (ax_geo, ax_ping) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )

    if altitude is None:
        ax_geo.set_title(f"Beam geometry unavailable (ping {ping_idx})")
        return fig, (ax_geo, ax_ping)

    max_reach = altitude
    for ray in (near_ray, far_ray):
        for entry in ray.values():
            for key in ("observed_across_track_m", "predicted_across_track_m"):
                v = entry.get(key)
                if v is not None:
                    max_reach = max(max_reach, abs(v))
    max_reach *= 1.15

    # ----- top panel: axial geometry -----
    ax_geo.plot(0, 0, marker="v", markersize=14, color="#142029", zorder=5)
    ax_geo.text(0, 0.35, "towfish", ha="center", fontsize=8, color="#142029")
    ax_geo.axhline(-altitude, color="#8B5A2B", linewidth=1.6)
    ax_geo.text(max_reach * 0.98, -altitude, f"seafloor (altitude={altitude:.2f}m)",
                ha="right", va="bottom", fontsize=8, color="#8B5A2B")
    ax_geo.axvline(0, color="k", linewidth=0.5, alpha=0.4, linestyle=":")

    for label, entry in near_ray.items():
        angle = entry.get("angle_from_vertical_deg")
        across = entry.get("observed_across_track_m")
        if angle is not None and across is not None:
            ax_geo.plot([0, across], [0, -altitude], color="#2E7D74", linewidth=2.2,
                        alpha=0.9, label=f"near edge ({label}): {angle:.0f}° from vertical" if label == "port" else None)
            ax_geo.plot(across, -altitude, marker="o", markersize=6, color="#2E7D74", zorder=6)

    for label, entry in far_ray.items():
        angle = entry.get("angle_from_vertical_deg")
        across = entry.get("predicted_across_track_m")
        if angle is not None and across is not None:
            ax_geo.plot([0, across], [0, -altitude], color="#B5482E", linewidth=2.2, linestyle="--",
                        alpha=0.9, label=f"far edge (usable range): {angle:.0f}° from vertical" if label == "port" else None)

    # Shade the beam's actual illuminated span between near and far edges,
    # per side, using each side's own (already-signed) coordinates --
    # not a magnitude re-signed by an external multiplier, which
    # previously double-applied the sign whenever the picked near_x was
    # already negative (port) and inverted the fill onto the wrong side.
    for label in ("port", "starboard"):
        near_x = near_ray.get(label, {}).get("observed_across_track_m")
        far_x = far_ray.get(label, {}).get("predicted_across_track_m")
        if near_x is not None and far_x is not None:
            ax_geo.fill(
                [0, near_x, far_x], [0, -altitude, -altitude],
                color="#2E7D74", alpha=0.08,
            )

    ax_geo.set_ylim(-altitude * 1.3, altitude * 0.6)
    ax_geo.set_xlim(-max_reach, max_reach)
    ax_geo.set_ylabel("Depth below towfish (m)")
    title = f"Beam geometry, ping {ping_idx}"
    if beam_center is not None and beam_width is not None:
        title += f" (center {beam_center:.0f}°, width {beam_width:.0f}° from vertical)"
    ax_geo.set_title(title)
    ax_geo.legend(fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.5))

    # ----- bottom panel: real ping profile on the same x-axis -----
    port = np.asarray(ds[value_var].isel(ping=ping_idx, channel=0).values, dtype=np.float64)
    stbd = np.asarray(ds[value_var].isel(ping=ping_idx, channel=1).values, dtype=np.float64)
    x_port = np.asarray(ds[range_var].isel(ping=ping_idx, channel=0).values, dtype=np.float64)
    x_stbd = np.asarray(ds[range_var].isel(ping=ping_idx, channel=1).values, dtype=np.float64)

    for x, y in ((x_port, port), (x_stbd, stbd)):
        valid = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[valid])
        ax_ping.plot(x[valid][order], y[valid][order], color="#2E4057", linewidth=0.7, alpha=0.9)

    for entry in near_ray.values():
        across = entry.get("observed_across_track_m")
        if across is not None:
            ax_ping.axvline(across, color="#2E7D74", linewidth=1.8, alpha=0.9)
    for entry in far_ray.values():
        across = entry.get("predicted_across_track_m")
        if across is not None:
            ax_ping.axvline(across, color="#B5482E", linewidth=1.8, linestyle="--", alpha=0.9)

    ax_ping.set_xlabel("Across-track distance (m)  (port ← | → starboard)")
    ax_ping.set_ylabel(value_var)
    ax_ping.set_title("Same ping, real backscatter (near/far beam edges marked)")

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
        ax_med.set_title("No binned data")
        return fig, (ax_med, ax_std)

    x = (binned["range_min_m"] + binned["range_max_m"]) / 2.0
    ax_med.plot(x, binned["median"], marker="o", markersize=3)
    ax_med.fill_between(x, binned["p10"], binned["p90"], alpha=0.2)
    ax_med.set_ylabel("Median backscatter")
    ax_med.set_title("Usable range")

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
        ax.set_title("No fit available")
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
            title=f"Decay model fit (best: {stage4.metrics.get('best_candidate')})",
        )
    return fig, ax


def plot_stage4_residuals(
    stage4: StageResult, *, use_fixed_scale: bool = True, ax: Axes | None = None, relative_x: bool = True,
) -> tuple[Figure, Axes]:
    """Residual (model - observed) vs. range for each candidate's fit
    window, using the same model-curve construction as
    `plot_candidate_fit` (recomputed here since `TVGCandidateFit` only
    stores the summary RMSE, not the per-bin residual array).

    `use_fixed_scale=True` (default, 2026-08-25 fix): draws the fixed-
    scale (1.0) residual and labels it with `fixed_scale_rmse_db` — the
    exact numbers `fit_hardware_tvg_curve()`/`stage4_decay` actually rank
    candidates by. **This function had silently kept using the old
    free-fit-scaled residual/RMSE after `plot_candidate_fit` (the
    decay_fit.png chart) was fixed to fixed-scale ranking on 2026-08-24
    — caught by a user reviewing the two charts side by side and noticing
    they no longer told the same story.** `use_fixed_scale=False` draws
    the old free-fit-scaled residual instead, for explicit diagnostic
    comparison (e.g. to see how much a candidate's scale had to move
    from 1.0 to look competitive) — same purpose as `plot_candidate_fit`'s
    own `use_fixed_scale` parameter, kept for symmetry.

    The nearest-bin `i_ref` (rather than an exact `tvg_gain_db`
    evaluation at `r_ref_m`, which `plot_candidate_fit`'s own *model
    line* uses for a different reason — a denser display domain) is
    deliberately kept here: it's not a discretization shortcut, it's the
    literal computation `fit_hardware_tvg_curve()` itself uses to
    produce `fixed_scale_rmse_db` — this plot's whole purpose is to show
    the actual per-bin residuals that produced that number, so it must
    match that function's math exactly, not a more "precise" variant of
    it.

    `ax`, if given, draws into it instead of creating a new figure —
    e.g. to combine this with `plot_candidate_fit` in one stacked figure
    (fit on top, residuals below, sharing an x-axis).

    `relative_x=True` (default): x-axis is distance from each
    candidate's own `r_ref_m` — matches this function's original
    convention. `relative_x=False`: x-axis is absolute range instead,
    matching `plot_candidate_fit`'s *current* convention (it used to
    also be reference-relative; that changed when the model line was
    extended back to zero range for reference) — pass `False` when
    stacking this under `plot_candidate_fit` on a shared x-axis, or the
    two panels' x=0 won't mean the same thing.
    """
    fit_result = stage4.data.get("fit_result")
    fig, ax = _new_fig_ax(ax, figsize=(9, 5))
    if fit_result is None:
        ax.set_title("No fit available")
        return fig, ax

    ranges = fit_result.observed_range_m
    observed = fit_result.observed_loss_db
    colors = {"flat": "C0", "spreading_only": "C1", "absorption_only": "C2", "full_physics": "C3"}

    for candidate in fit_result.candidates:
        model_curve = tvg_gain_db(np.clip(ranges, candidate.params.min_range_m, None), candidate.params)
        i_ref = int(np.argmin(np.abs(ranges - candidate.r_ref_m)))
        plotted_scale = 1.0 if use_fixed_scale else candidate.scale
        model_loss = plotted_scale * (model_curve - model_curve[i_ref])
        residual = model_loss - observed
        # r_fit_min_m is None on a TVGCandidateFit built before that field
        # existed -- falls back to r_ref_m, the prior one-sided behavior.
        fit_min = candidate.r_fit_min_m if candidate.r_fit_min_m is not None else candidate.r_ref_m
        in_window = (ranges >= fit_min) & (ranges <= candidate.r_fit_max_m)
        rmse_shown = candidate.fixed_scale_rmse_db if use_fixed_scale else candidate.rmse_db
        x = ranges - candidate.r_ref_m if relative_x else ranges
        ax.plot(
            x[in_window], residual[in_window],
            color=colors.get(candidate.name), marker=".", markersize=4,
            label=f"{candidate.name} (RMSE={rmse_shown:.2f}dB)",
        )

    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Distance from reference range (m)" if relative_x else "Range (m)")
    ax.set_ylabel("Residual, model - observed (dB)")
    ax.set_title("Fit residuals")
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
    fig.suptitle(f"Distribution shape ({value_var})")
    return fig, axes
