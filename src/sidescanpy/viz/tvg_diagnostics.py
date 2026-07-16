"""
Plotting functions for TVG diagnostics.

Consumes `sidescanpy.analysis.tvg_diagnostics`'s outputs. Follows
`viz/gridded.py`'s `(fig, ax)`-return convention and the project-wide
1.0/99.0 percentile-band defaults.

2026 rshomberg@uri.edu
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def _new_fig_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots()
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
        Output of `mean_backscatter_vs_range` (or
        `mean_backscatter_vs_sample_index` — any DataFrame with a
        "mean" column and matching min/max range columns).
    show_band : bool, default True
        Shade the region between `band`'s two percentile columns.
    band : (str, str), default ("p10", "p90")
    """
    fig, ax = _new_fig_ax(ax)

    range_col = "range_min_m" if "range_min_m" in binned.columns else "sample_min"
    max_col = "range_max_m" if "range_max_m" in binned.columns else "sample_max"
    x = (binned[range_col] + binned[max_col]) / 2.0

    line, = ax.plot(x, binned["mean"], label=label, color=color, marker="o", markersize=3)
    if show_band and band[0] in binned.columns and band[1] in binned.columns:
        ax.fill_between(
            x, binned[band[0]], binned[band[1]], alpha=0.2, color=line.get_color()
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
    Plot `tvg_gain_db` vs. range from `before_after_backscatter`'s output.

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
    seed: int = 0,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    "Spaghetti plot": overlay `n_pings` individual ping profiles
    (`value_var` vs. `range_var`, one channel) at low alpha.

    Visual density shows ping-to-ping consistency/variance directly —
    complementary to `plot_mean_vs_range`'s percentile band, which
    summarizes spread but averages away periodic per-ping artifacts.
    """
    fig, ax = _new_fig_ax(ax)

    n_total = ds.sizes["ping"]
    rng = np.random.default_rng(seed)
    n_pick = min(n_pings, n_total)
    ping_idx = rng.choice(n_total, size=n_pick, replace=False)

    channel_idx = 0
    for p in ping_idx:
        sel = dict(ping=p, channel=channel_idx)
        r = np.asarray(ds[range_var].isel(**sel).values, dtype=np.float64)
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
    Overlay `mean_backscatter_vs_range` curves for multiple frequencies,
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
    Overlay `mean_backscatter_vs_range` curves for multiple altitude
    bins (from `bin_pings_by_altitude`), color-coded by bin.

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
