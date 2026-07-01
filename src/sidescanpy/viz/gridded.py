"""
Visualization for gridded raster data.

Functions for plotting SideScanGriddedRaster objects with matplotlib.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from sidescanpy.grids.raster import SideScanGriddedRaster

logger = logging.getLogger(__name__)


def plot_gridded_raster(
    raster: SideScanGriddedRaster,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    show_grid: bool = False,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a gridded raster as a 2D heatmap.

    Parameters
    ----------
    raster : SideScanGriddedRaster
        Gridded backscatter raster.
    fig : matplotlib.figure.Figure, optional
        Existing figure. If None, create new.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, create new.
    cmap : str, default "viridis"
        Colormap name.
    vmin, vmax : float, optional
        Colorbar min/max. If None, use percentile-based auto-scaling.
    show_colorbar : bool, default True
        Include colorbar.
    show_grid : bool, default False
        Show grid lines.
    title : str, optional
        Plot title. If None, auto-generate from raster metadata.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Axes with plotted raster.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Extract backscatter and coordinates
    backscatter = raster.backscatter
    lats = raster.dataset["latitude"].values
    lons = raster.dataset["longitude"].values

    # Auto-scale if not provided
    if vmin is None or vmax is None:
        finite_vals = backscatter[np.isfinite(backscatter)]
        if len(finite_vals) > 0:
            if vmin is None:
                vmin = np.percentile(finite_vals, 2)
            if vmax is None:
                vmax = np.percentile(finite_vals, 98)
        else:
            vmin, vmax = 0, 1

    # Create masked array for NaN handling
    backscatter_masked = np.ma.masked_invalid(backscatter)

    # Plot
    im = ax.pcolormesh(
        lons,
        lats,
        backscatter_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    # Labels and formatting
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    if title is None:
        title = (
            f"{raster.channel_name} — "
            f"{raster.dataset.attrs.get('resampling_method', 'unknown')} "
            f"(resolution: {raster.resolution_m:.0f} m)"
        )
    ax.set_title(title)

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_aspect("equal")

    if show_colorbar:
        units = raster.dataset["backscatter"].attrs.get("units", "unknown units")
        cbar = plt.colorbar(im, ax=ax, label=f"Backscatter ({units})")
        logger.debug(f"Colorbar range: {vmin:.3f} — {vmax:.3f}")

    return fig, ax


def plot_gridded_with_coverage(
    raster: SideScanGriddedRaster,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot gridded raster with n_contributing_samples as transparency.

    Cells with few contributing samples appear faded; cells with many
    contributing samples appear opaque.

    Parameters
    ----------
    raster : SideScanGriddedRaster
        Gridded backscatter raster.
    fig : matplotlib.figure.Figure, optional
        Existing figure. If None, create new.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, create new.
    cmap : str, default "viridis"
        Colormap name.
    vmin, vmax : float, optional
        Backscatter colorbar min/max.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    backscatter = raster.backscatter
    lats = raster.dataset["latitude"].values
    lons = raster.dataset["longitude"].values

    # Get n_contributing_samples if available, otherwise use uniform alpha
    if "n_contributing_samples" in raster.dataset.data_vars:
        n_contrib = raster.dataset["n_contributing_samples"].values
    else:
        # Fallback: use backscatter validity as alpha (finite=opaque, NaN=transparent)
        n_contrib = np.isfinite(backscatter).astype(int)
        logger.debug("n_contributing_samples not found; using backscatter validity for alpha")

    # Auto-scale backscatter
    if vmin is None or vmax is None:
        finite_vals = backscatter[np.isfinite(backscatter)]
        if len(finite_vals) > 0:
            if vmin is None:
                vmin = np.percentile(finite_vals, 2)
            if vmax is None:
                vmax = np.percentile(finite_vals, 98)
        else:
            vmin, vmax = 0, 1

    # Normalize n_contrib to [0, 1] for alpha channel
    max_contrib = np.max(n_contrib)
    if max_contrib > 0:
        alpha = n_contrib / max_contrib
    else:
        alpha = np.ones_like(n_contrib)

    # Create RGBA image
    backscatter_masked = np.ma.masked_invalid(backscatter)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # Map backscatter to colors, then apply alpha
    colors = cmap_obj(norm(backscatter))
    colors[..., 3] = alpha  # Set alpha channel

    # Plot as image
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    im = ax.imshow(
        colors,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    if title is None:
        title = (
            f"{raster.channel_name} — "
            f"{raster.dataset.attrs.get('resampling_method', 'unknown')} "
            f"(coverage: {raster.data_coverage:.1%})"
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.2, linestyle="--")

    return fig, ax

def plot_gridded_multiple_methods(
    rasters: dict,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot gridded rasters from multiple resampling methods side-by-side.

    Parameters
    ----------
    rasters : dict
        Dict mapping method name (str) to SideScanGriddedRaster.
        Example: {"mean": raster_mean, "max": raster_max, ...}
    figsize : tuple of (width, height)
        Figure size in inches.
    cmap : str, default "viridis"
        Colormap name.
    vmin, vmax : float, optional
        Shared colorbar range. If None, compute from all rasters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
        Array of axes (one per method).
    """
    # Compute shared min/max across all rasters
    if vmin is None or vmax is None:
        all_finite = []
        for raster in rasters.values():
            finite_vals = raster.backscatter[np.isfinite(raster.backscatter)]
            all_finite.extend(finite_vals)

        all_finite = np.array(all_finite)
        if len(all_finite) > 0:
            if vmin is None:
                vmin = np.percentile(all_finite, 2)
            if vmax is None:
                vmax = np.percentile(all_finite, 98)
        else:
            vmin, vmax = 0, 1

    # Create subplots
    n_methods = len(rasters)
    n_cols = min(3, n_methods)  # Up to 3 columns
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Plot each method
    for idx, (method_name, raster) in enumerate(rasters.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Plot
        backscatter_masked = np.ma.masked_invalid(raster.backscatter)
        lats = raster.dataset["latitude"].values
        lons = raster.dataset["longitude"].values

        im = ax.pcolormesh(
            lons,
            lats,
            backscatter_masked,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.set_title(f"{method_name} (cov: {raster.data_coverage:.1%})", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_aspect("equal")

        units = raster.dataset["backscatter"].attrs.get("units", "unknown units")
        plt.colorbar(im, ax=ax, label=f"Backscatter ({units})")

    # Hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.tight_layout()

    return fig, axes


def export_gridded_png(
    raster: SideScanGriddedRaster,
    output_path: str,
    dpi: int = 150,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Export gridded raster to PNG file.

    Parameters
    ----------
    raster : SideScanGriddedRaster
        Gridded backscatter raster.
    output_path : str
        Output PNG file path.
    dpi : int, default 150
        Dots per inch.
    cmap : str, default "viridis"
        Colormap name.
    vmin, vmax : float, optional
        Colorbar range.
    """
    fig, ax = plot_gridded_raster(
        raster,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=True,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Exported gridded raster to {output_path}")
    plt.close(fig)