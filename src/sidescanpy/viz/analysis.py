"""
Plots for the analysis results in :mod:`sidescanpy.analysis`.

Each function renders one result (or a dataset, where the plot is of the
data itself), draws into caller-supplied axes when given, returns
``(fig, ax)`` or ``(fig, axes)`` and never saves a file. Figure-level
decorations (suptitle, ``tight_layout``) are only applied when the
function created the figure. TVG / range-decay plots live in
:mod:`sidescanpy.viz.tvg`.

2026 rshomberg@uri.edu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sidescanpy.analysis.beam import BeamGeometryResult
from sidescanpy.analysis.nadir import NadirResult
from sidescanpy.analysis.navigation import NavigationSummary
from sidescanpy.analysis.outliers import OutlierComparisonResult
from sidescanpy.analysis.usable_range import UsableRangeResult
from sidescanpy.viz.waterfall import plot_waterfall

logger = logging.getLogger(__name__)


def _new_fig_ax(ax: Axes | None, **subplots_kwargs) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(**subplots_kwargs)
    else:
        fig = ax.figure
    return fig, ax


def _new_fig_axes(axes, nrows: int, ncols: int, **subplots_kwargs) -> tuple[Figure, tuple, bool]:
    """``(fig, axes, created)`` -- a new grid of subplots unless ``axes`` is given."""
    if axes is None:
        fig, grid = plt.subplots(nrows, ncols, **subplots_kwargs)
        return fig, tuple(np.ravel(grid)), True
    axes = tuple(axes)
    return axes[0].figure, axes, False


def _finish(fig: Figure, created: bool, suptitle: str | None = None) -> None:
    if not created:
        return
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()


# ============================================================
# Value characterization
# ============================================================


def plot_value_histograms(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    *,
    bins: int = 100,
    axes: tuple[Axes, Axes] | None = None,
    title: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Dual-panel histogram: linear scale (left) and log10(|x|) scale (right)."""
    arr = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]

    fig, (ax_lin, ax_log), created = _new_fig_axes(axes, 1, 2, figsize=(11, 4))
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

    _finish(fig, created, title if title is not None else f"Value distribution ({value_var})")
    return fig, (ax_lin, ax_log)


def plot_units_distribution(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    *,
    bins: int = 100,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """The :func:`plot_value_histograms` view titled for a units call
    (``infer_units`` reads the value-range signature this shows)."""
    return plot_value_histograms(ds, value_var, bins=bins, axes=axes, title=f"Distribution shape ({value_var})")


def plot_sample_ping(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    *,
    ping_idx: int | None = None,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """One representative ping (default: the middle one), port and starboard side by side."""
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2

    fig, (ax_port, ax_stbd), created = _new_fig_axes(axes, 1, 2, figsize=(11, 4), sharey=True)
    for ax, ch, label in ((ax_port, 0, "port"), (ax_stbd, 1, "starboard")):
        y = np.asarray(ds[value_var].isel(ping=ping_idx, channel=ch).values, dtype=np.float64)
        ax.plot(np.arange(y.size), y, linewidth=0.7)
        ax.set_xlabel("Sample index")
        ax.set_title(label)
    ax_port.set_ylabel(value_var)
    _finish(fig, created, f"Sample ping {ping_idx}")
    return fig, (ax_port, ax_stbd)


# ============================================================
# Outlier rejection
# ============================================================


def plot_outlier_comparison(
    ds: xr.Dataset,
    result: OutlierComparisonResult,
    value_var: str = "backscatter_intensity",
    *,
    bins: int = 100,
    linear_xlim_percentile: float = 99.5,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Overlaid before/after histograms (linear + log scale) of what
    ``reject_range_outliers`` discarded from ``value_var``.

    Each panel uses one shared set of bin edges for both populations. The
    linear panel is capped at the "before" population's
    ``linear_xlim_percentile`` so a few extreme values cannot squash the
    bulk; the count beyond the cap is annotated.
    """
    filtered_arr = result.filtered_values

    original = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
    original_finite = original[np.isfinite(original)]

    filtered_finite = np.array([])
    if filtered_arr is not None:
        filtered_finite = np.asarray(filtered_arr, dtype=np.float64).ravel()
        filtered_finite = filtered_finite[np.isfinite(filtered_finite)]

    fig, (ax_lin, ax_log), created = _new_fig_axes(axes, 1, 2, figsize=(11, 4))

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

    n_rejected = result.n_rejected
    pct_rejected = result.pct_rejected
    suffix = f" — {n_rejected} rejected ({pct_rejected:.2f}%)" if n_rejected is not None else ""
    _finish(fig, created, f"Distribution before/after outlier rejection ({value_var}){suffix}")
    return fig, (ax_lin, ax_log)


# ============================================================
# Nadir
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


def plot_cross_track_profile(
    ds: xr.Dataset, value_var: str = "backscatter_intensity", *, ping_idx: int | None = None, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """One ping across the full swath (port mirrored to negative sample index)."""
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


def plot_nadir_zoom(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    *,
    nadir: NadirResult | None = None,
    nadir_n_samples: int = 50,
    ping_idx: int | None = None,
    zoom_factor: int = 2,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Cross-track profile zoomed around nadir.

    Without ``nadir`` (an ``analyze_nadir()`` result) the window is
    ``nadir_n_samples * zoom_factor``. With it, the window widens to the
    detected boundaries, which are marked: altitude (dotted, the
    vehicle altitude in sample units) and first bottom return (dashed).
    Everything from nadir out to the first bottom return is shaded as
    "water column returns".
    """
    n_pings = ds.sizes["ping"]
    ping_idx = ping_idx if ping_idx is not None else n_pings // 2
    fig, ax = _new_fig_ax(ax, figsize=(9, 4.5))

    x, y = _cross_track_profile(ds, value_var, ping_idx)

    window = nadir_n_samples * zoom_factor
    altitude_p90 = None
    bottom_p90 = None
    if nadir is not None:
        altitude_p90 = nadir.altitude_sample_p90
        bottom_p90 = nadir.first_bottom_return_sample_p90
        widen_to = bottom_p90 if bottom_p90 is not None else altitude_p90
        if widen_to is not None:
            window = max(window, int(widen_to * 1.3))

    mask = np.abs(x) <= window
    ax.plot(x[mask], y[mask], marker="o", markersize=2.5, linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)

    altitude_b = None
    bottom_b = None
    if altitude_p90 is not None or bottom_p90 is not None:
        altitude_arr = nadir.altitude_sample
        bottom_arr = nadir.first_bottom_return_sample
        altitude_this = altitude_arr[ping_idx, 0] if altitude_arr is not None and ping_idx < altitude_arr.shape[0] else np.nan
        bottom_this = bottom_arr[ping_idx, 0] if bottom_arr is not None and ping_idx < bottom_arr.shape[0] else np.nan
        altitude_b = altitude_this if np.isfinite(altitude_this) else altitude_p90
        bottom_b = bottom_this if np.isfinite(bottom_this) else bottom_p90

    if bottom_b is not None:
        ax.axvspan(-bottom_b, bottom_b, color="#B5482E", alpha=0.08, label="water column returns")
    else:
        # No detected boundary: fall back to the raw blanking window.
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


def plot_nadir_waterfall(
    ds: xr.Dataset,
    value_var: str = "backscatter_intensity",
    *,
    nadir: NadirResult | None = None,
    ax: Axes | None = None,
    ping_offset: int = 0,
    **waterfall_kwargs,
) -> tuple[Figure, Axes]:
    """Cross-track-vs-ping waterfall of ``value_var`` (via
    :func:`plot_waterfall`), with ``nadir``'s altitude (dotted) and first
    bottom return (solid) traced on both sides when given.

    ``ping_offset`` shifts both the image and the traced boundaries, so a
    slice of a larger survey can show absolute ping numbers.
    """
    fig, ax = _new_fig_ax(ax, figsize=(12, 6))
    aliased = ds.assign(backscatter_intensity=ds[value_var])
    plot_waterfall(
        aliased, x_axis="sample", ax=ax, title=f"Waterfall ({value_var})",
        ping_offset=ping_offset, **waterfall_kwargs,
    )

    if nadir is not None:
        altitude_arr = nadir.altitude_sample
        bottom_arr = nadir.first_bottom_return_sample

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
# Beam geometry
# ============================================================


def plot_beam_geometry(
    ds: xr.Dataset,
    result: BeamGeometryResult,
    value_var: str = "backscatter_intensity",
    range_var: str = "across_track_m",
    *,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Side-view beam-geometry diagram above the same ping's backscatter
    profile, sharing one x-axis.

    Top: towfish, seafloor at the ping's altitude, the near edge (solid,
    from the first bottom return) and far edge (dashed, from the usable
    range), with the illuminated wedge shaded. Bottom: ``value_var`` vs.
    ``range_var`` with the same edges marked.
    """
    ping_idx = result.ping_idx if result.ping_idx is not None else 0
    altitude = result.altitude_m
    near_ray = result.near_ray
    far_ray = result.far_ray
    beam_center = result.beam_center_angle_deg
    beam_width = result.beam_width_deg

    fig, (ax_geo, ax_ping), created = _new_fig_axes(
        axes, 2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
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

    # Shade each side's illuminated span from its own signed coordinates.
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

    _finish(fig, created)
    return fig, (ax_geo, ax_ping)


# ============================================================
# Usable range
# ============================================================


def plot_usable_range(
    result: UsableRangeResult, *, axes: tuple[Axes, Axes] | None = None
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Median and std-dev vs. range, with the usable-range boundary marked."""
    binned = result.binned
    fig, (ax_med, ax_std), created = _new_fig_axes(axes, 2, 1, figsize=(9, 7), sharex=True)

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

    usable_range_m = result.usable_range_m
    if usable_range_m is not None:
        for a in (ax_med, ax_std):
            a.axvline(usable_range_m, color="red", linestyle="--", linewidth=1, label=f"usable range = {usable_range_m:.1f}m")
            a.legend(fontsize=8)

    noise_floor = result.noise_floor
    if noise_floor is not None:
        ax_med.axhline(noise_floor, color="gray", linestyle=":", linewidth=1)

    _finish(fig, created)
    return fig, (ax_med, ax_std)


# ============================================================
# Segments
# ============================================================


def plot_segments_overview(
    ds: xr.Dataset, segments, *, axes: tuple[Axes, Axes] | None = None
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Altitude and pitch vs. ping index with each segment boundary marked
    and excluded spans shaded, for reviewing ``find_segments()`` output."""
    n_pings = ds.sizes["ping"]
    ping_idx = np.arange(n_pings)
    fig, (ax_alt, ax_pitch), created = _new_fig_axes(axes, 2, 1, figsize=(11, 6), sharex=True)

    if "altitude_m" in ds:
        ax_alt.plot(ping_idx, ds["altitude_m"].values, color="C0", linewidth=0.8)
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Segmentation overview")

    if "pitch_deg" in ds:
        ax_pitch.plot(ping_idx, ds["pitch_deg"].values, color="C1", linewidth=0.8)
    ax_pitch.set_ylabel("Pitch (deg)")
    ax_pitch.set_xlabel("Ping index")

    for ax in (ax_alt, ax_pitch):
        for seg in segments:
            if seg.status == "excluded":
                ax.axvspan(seg.start, seg.stop, color="red", alpha=0.15)
            ax.axvline(seg.start, color="k", linewidth=0.6, alpha=0.5)
        if segments:
            ax.axvline(segments[-1].stop, color="k", linewidth=0.6, alpha=0.5)

    _finish(fig, created)
    return fig, (ax_alt, ax_pitch)


# ============================================================
# Navigation / attitude
# ============================================================


def plot_navigation_summary(
    ds: xr.Dataset,
    *,
    ping_offset: int = 0,
    result: NavigationSummary | None = None,
    axes: tuple[Axes, Axes, Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes, Axes]]:
    """Heading, pitch/roll, altitude/depth/heave vs. ping number, plus the
    lon/lat track colored by ping number.

    The heading panel shades the [0,360) and [-180,180) conventions and,
    given ``result`` (``summarize_navigation()``), marks points outside
    both. ``ping_offset`` makes the x-axis read absolute ping numbers.
    ``axes`` is ``(heading, attitude, vertical, track)``.
    """
    n_pings = ds.sizes["ping"]
    ping_axis = ping_offset + np.arange(n_pings)

    if axes is None:
        fig, axd = plt.subplot_mosaic(
            [["heading", "track"], ["attitude", "track"], ["vertical", "track"]],
            figsize=(13, 9), width_ratios=[2.0, 1.2],
        )
        ax_heading, ax_attitude, ax_vertical, ax_track = axd["heading"], axd["attitude"], axd["vertical"], axd["track"]
        ax_attitude.sharex(ax_heading)
        ax_vertical.sharex(ax_heading)
        created = True
    else:
        ax_heading, ax_attitude, ax_vertical, ax_track = axes
        fig = ax_heading.figure
        created = False

    if "heading_deg" in ds:
        heading = np.asarray(ds["heading_deg"].values, dtype=np.float64)
        ax_heading.axhspan(0, 360, color="tab:green", alpha=0.05, zorder=0)
        ax_heading.axhspan(-180, 180, color="tab:blue", alpha=0.05, zorder=0)
        for y in (0, 360, -180, 180):
            ax_heading.axhline(y, color="0.75", lw=0.6, ls=":")
        ax_heading.plot(ping_axis, heading, color="0.2", lw=0.8)
        if result is not None and (result.heading_frac_out_of_range or 0.0) > 0.0:
            out_mask = ~(((heading >= 0.0) & (heading < 360.0)) | ((heading >= -180.0) & (heading < 180.0)))
            ax_heading.scatter(
                ping_axis[out_mask], heading[out_mask], color="tab:red", s=10, zorder=5,
                label="outside both conventions",
            )
            ax_heading.legend(loc="upper right", fontsize=8)
        title = "heading_deg"
        if result is not None and result.heading_convention:
            title += f"  (classification: {result.heading_convention})"
        ax_heading.set_title(title)
        ax_heading.set_ylabel("degrees")

    for var, color in (("pitch_deg", "tab:orange"), ("roll_deg", "tab:purple")):
        if var in ds:
            ax_attitude.plot(ping_axis, ds[var].values, color=color, lw=0.8, label=var)
    ax_attitude.axhline(0, color="0.8", lw=0.6)
    ax_attitude.set_ylabel("degrees")
    ax_attitude.set_title("pitch / roll")
    if ax_attitude.get_legend_handles_labels()[0]:
        ax_attitude.legend(loc="upper right", fontsize=8)

    for var, color in (("altitude_m", "tab:green"), ("depth_m", "tab:brown"), ("heave_m", "tab:gray")):
        if var in ds:
            ax_vertical.plot(ping_axis, ds[var].values, color=color, lw=0.8, label=var)
    ax_vertical.set_ylabel("meters")
    ax_vertical.set_xlabel("ping number")
    ax_vertical.set_title("altitude / depth / heave")
    if ax_vertical.get_legend_handles_labels()[0]:
        ax_vertical.legend(loc="upper right", fontsize=8)

    if "latitude" in ds and "longitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        ax_track.plot(lon, lat, color="0.6", lw=0.4, zorder=0)
        sc = ax_track.scatter(lon, lat, c=ping_axis, cmap="viridis", s=6, zorder=1)
        ax_track.set_aspect("equal", adjustable="datalim")
        ax_track.set_xlabel("longitude")
        ax_track.set_ylabel("latitude")
        ax_track.set_title("track (color = ping number)")
        fig.colorbar(sc, ax=ax_track, label="ping number", fraction=0.046, pad=0.04)
    else:
        ax_track.set_axis_off()
        ax_track.text(0.5, 0.5, "no latitude/longitude", ha="center", va="center", transform=ax_track.transAxes)

    _finish(fig, created, f"pings {ping_axis[0]}:{ping_axis[-1] + 1}" if n_pings else "pings (empty)")
    return fig, (ax_heading, ax_attitude, ax_vertical, ax_track)
