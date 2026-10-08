"""
CLI-private figure layouts and PNG writers.

The ``sidescan-*`` commands are a quick-run convenience, so the figures
they save are composed and written here rather than in ``sidescanpy.viz``
(whose helpers only draw and return Axes/Figure). Importing this module
needs matplotlib (the ``viz`` extra); CLI entry points import it lazily
after :func:`sidescanpy.cli._common.require_matplotlib` has checked for it.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: never probe for a GUI backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pyproj import CRS

from sidescanpy.core.target import Target
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.processing.track import SurveyTrack
from sidescanpy.viz.gridded import plot_gridded_raster
from sidescanpy.viz.targets import plot_target_position_scatter, plot_track_with_targets
from sidescanpy.viz.track import plot_track_and_scanlines
from sidescanpy.viz.tvg import plot_candidate_fit, plot_range_decay_residuals, set_fit_window_limits
from sidescanpy.viz.waterfall import plot_survey_overview, plot_waterfall

logger = logging.getLogger(__name__)


def save_figure(fig: Figure, path, *, dpi: int, tight: bool = False) -> None:
    """Save ``fig`` to ``path`` and close it."""
    try:
        if tight:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


# ============================================================
# sidescan-export grid / track
# ============================================================

def save_gridded_png(
    raster: SideScanGriddedRaster,
    output_path: str,
    dpi: int = 150,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    gamma: float = 1.0,
    show_colorbar: bool = True,
    show_land: bool = False,
    show_locator: bool = False,
    show_scale_bar: bool = True,
) -> None:
    """Plot a gridded raster with ``plot_gridded_raster`` and save it,
    cropped to the drawn content (``bbox_inches="tight"``)."""
    fig, _ax = plot_gridded_raster(
        raster,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmin_pct=vmin_pct,
        vmax_pct=vmax_pct,
        gamma=gamma,
        show_colorbar=show_colorbar,
        show_land=show_land,
        show_locator=show_locator,
        show_scale_bar=show_scale_bar,
    )
    save_figure(fig, output_path, dpi=dpi, tight=True)


def save_track_png(
    track: SurveyTrack,
    output_path: str,
    *,
    crs: Optional[CRS] = None,
    dpi: int = 150,
    show_land: bool = False,
    show_locator: bool = False,
    show_scale_bar: bool = True,
) -> None:
    """Plot a survey track + scan lines and save it.

    Saves at the figure's own fixed size (no ``bbox_inches="tight"``):
    ``plot_track_and_scanlines`` uses a fixed figsize with an equal
    aspect, and cropping to the drawn content would make the PNG size
    depend on the survey's shape. Every track.png has the same pixel
    dimensions; letterbox padding is fine.
    """
    fig, _ax = plot_track_and_scanlines(
        track.track_lat, track.track_lon,
        track.scan_port_lat, track.scan_port_lon,
        track.scan_stbd_lat, track.scan_stbd_lon,
        crs=crs,
        show_land=show_land,
        show_scale_bar=show_scale_bar,
        show_locator=show_locator,
    )
    save_figure(fig, output_path, dpi=dpi)


def save_track_with_targets_png(
    track: SurveyTrack,
    targets: list[Target],
    output_path: str,
    *,
    crs: Optional[CRS] = None,
    dpi: int = 150,
    show_land: bool = False,
    show_locator: bool = False,
) -> None:
    """Plot a survey track + scan lines + target overlay and save it at the
    figure's own fixed size (see :func:`save_track_png` for why)."""
    fig, _ax = plot_track_with_targets(track, targets, crs=crs, show_land=show_land, show_locator=show_locator)
    save_figure(fig, output_path, dpi=dpi)


# ============================================================
# sidescan-convert targets
# ============================================================

def save_target_position_scatter_png(
    targets: list[Target], output_path: str, *, title: Optional[str] = None, dpi: int = 150,
) -> bool:
    """Save ``plot_target_position_scatter``'s figure. Returns False (no
    file written) if no target had a resolvable position."""
    fig = plot_target_position_scatter(targets, title=title)
    if fig is None:
        return False
    save_figure(fig, output_path, dpi=dpi)
    return True


# ============================================================
# sidescan-export waterfall
# ============================================================

def save_waterfall_png(
    ds, output_path: Path, *, x_axis, y_axis, ping_offset, cmap, vmin, vmax, vmin_pct, vmax_pct, gamma, dpi,
    value_var="backscatter_intensity", box_rects=(),
) -> None:
    """One waterfall window, with an optional red box + label per entry of
    ``box_rects`` (``(label, x0, y0, width, height)`` in the plot's axis
    units)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        plot_waterfall(
            ds, x_axis=x_axis, y_axis=y_axis, ping_offset=ping_offset,
            cmap=cmap, vmin=vmin, vmax=vmax, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
            gamma=gamma, ax=ax, value_var=value_var,
        )
        for label, x0, y0, width, height in box_rects:
            ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="red", linewidth=1.2))
            ax.annotate(label, (x0, y0), color="red", fontsize=11, fontweight="bold", xytext=(2, 2), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)


def save_survey_overview_png(ds, output_path: Path, *, dpi: int, **overview_kwargs) -> None:
    """Whole-survey compressed waterfall (``plot_survey_overview``) on a
    14x8 inch figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    try:
        plot_survey_overview(ds, ax=ax, **overview_kwargs)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)


# ============================================================
# sidescan-analyze tvg-fit
# ============================================================

def save_tvg_fit_png(decay, output_path: Path, *, suptitle: str, residual_xlabel: str, dpi: int = 140) -> None:
    """Two-panel decay-fit figure: every candidate curve over the observed
    loss (top) and the best fit's residuals (bottom)."""
    fit_result = decay.fit
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for i, c in enumerate(fit_result.candidates):
        plot_candidate_fit(c, fit_result.observed_range_m, fit_result.observed_loss_db, ax=ax_fit, show_observed=(i == 0))
    ax_fit.set_title(f"Decay fit — best: {decay.best_candidate}, RMSE={decay.best_rmse_db:.2f}dB")
    ax_fit.set_xlabel("")

    plot_range_decay_residuals(decay, ax=ax_res, relative_x=False)
    ax_res.set_title("Residuals")
    ax_res.set_xlabel(residual_xlabel)

    set_fit_window_limits(ax_fit, decay)

    fig.suptitle(suptitle)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi)

