"""
Plot a gridded raster (:class:`~sidescanpy.grids.SideScanGriddedRaster`).

:func:`plot_gridded_raster` draws one raster on one Axes and returns
``(fig, ax)``. Multi-raster comparisons and coverage views are recipes in
``examples/figures/``.

2026 rshomberg@uri.edu
"""

import logging
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy.core.units import mask_floor_clip
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.viz._cmap import resolve_cmap

logger = logging.getLogger(__name__)


# ============================================================
# CRS-AWARE LABELS
# ============================================================

def _axis_labels(raster: SideScanGriddedRaster) -> Tuple[str, str]:
    """(xlabel, ylabel) for a raster's CRS — degrees for geographic, meters for projected."""
    if raster.is_geographic:
        return "Longitude (°E)", "Latitude (°N)"
    return "Easting (m)", "Northing (m)"


def _crs_label(raster: SideScanGriddedRaster) -> str:
    """Short human-readable CRS identifier, e.g. 'WGS 84 / UTM zone 19N (EPSG:32619)'."""
    crs = raster.crs
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"{crs.name} (EPSG:{epsg})"
    return crs.name


def plot_gridded_raster(
    raster: SideScanGriddedRaster,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    gamma: float = 1.0,
    show_colorbar: bool = True,
    show_grid: bool = False,
    show_land: bool = False,
    show_locator: bool = False,
    show_scale_bar: bool = True,
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
        Colorbar min/max. If None, use percentile-based auto-scaling
        (``vmin_pct``/``vmax_pct``).
    vmin_pct, vmax_pct : float, default 1.0 and 99.0
        Percentiles for auto color scaling when ``vmin``/``vmax`` aren't
        given. Matches ``plot_waterfall``'s defaults — use the same
        values for both when comparing a waterfall and a gridded export
        of the same data, otherwise the two can show very different
        apparent contrast/normalization purely from differing percentile
        cutoffs, independent of any real difference in the underlying
        values (backscatter distributions with a sharp low-end density
        cliff, e.g. from floor-clipped no-return samples, are
        especially sensitive to this — a 1-2 percentile difference can
        shift the effective vmin by tens of dB).

        **Matching these two values is necessary but not sufficient**
        for a waterfall and a gridded export to look comparably
        normalized: percentiles are computed over different underlying
        populations (every raw sample vs. every gridded cell *mean*),
        so even identical ``vmin_pct``/``vmax_pct`` can yield different
        absolute bounds — see ``resample_channel_to_grid()``'s "Notes"
        section.
    gamma : float, default 1.0
        Display-only power-law contrast curve, applied via
        ``matplotlib.colors.PowerNorm(gamma, vmin, vmax)`` instead of a
        plain linear scale. The plotted array is never renormalized —
        only the value-to-color mapping is warped — so ``raster``
        stays in its real units (e.g. dB) and the colorbar (if shown)
        keeps honest, correctly-labeled tick values, just non-uniformly
        spaced. ``gamma=1.0`` (default) is linear, identical to this
        function's behavior before this parameter existed.
    show_colorbar : bool, default True
        Include colorbar.
    show_grid : bool, default False
        Show grid lines.
    show_land : bool, default False
        Draw land/water context (vendored 1:50m Natural Earth data)
        behind the raster, in cells NaN masks out (see
        ``sidescanpy.viz.basemap``) -- water-colored fill plus land
        polygons, drawn first so the raster paints over it wherever
        there's real data. Requires the optional ``maps`` extra
        (cartopy); raises ``ImportError`` with an install hint if it
        isn't installed, rather than silently skipping the background.
    show_locator : bool, default False
        Add a small inset (bottom-left) showing this raster's location
        zoomed way out, with a marker box — "where is this, relative to
        the coast." Requires the optional ``maps`` extra.
    show_scale_bar : bool, default True
        Draw a scale bar (see ``sidescanpy.viz.scalebar``). Only drawn
        for a projected raster (``raster.is_geographic`` False, the
        common case — auto-UTM-selected grids) -- a geographic raster
        is skipped, not drawn wrong, since meters-per-degree isn't a
        single constant this function can infer.
    title : str, optional
        Plot title. If None, auto-generate from raster metadata,
        including the CRS (see ``_crs_label``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Axes with plotted raster.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Extract backscatter and coordinates (CRS-aware: lat/lon degrees
    # for a geographic raster, northing/easting meters for a projected
    # one — see SideScanGriddedRaster.y_values/x_values)
    backscatter = mask_floor_clip(raster.backscatter)
    ys = raster.y_values
    xs = raster.x_values

    # Auto-scale if not provided
    if vmin is None or vmax is None:
        finite_vals = backscatter[np.isfinite(backscatter)]
        if len(finite_vals) > 0:
            if vmin is None:
                vmin = np.percentile(finite_vals, vmin_pct)
            if vmax is None:
                vmax = np.percentile(finite_vals, vmax_pct)
        else:
            vmin, vmax = 0, 1

    if show_land:
        from sidescanpy.viz.basemap import add_land_background

        # margin_frac=0.0: unlike track.py's tight-to-the-data bbox,
        # raster.bbox already has its own ~10% margin baked in by
        # bbox_from_channel_dataset() (the grid-construction step, run
        # before this raster ever existed) -- add_land_background's own
        # default 8% margin would stack on top of that, nearly doubling
        # the padding around the swath. That compounding was the direct
        # cause of a real user-reported "way more padding than
        # necessary" bug (2026-09-18).
        add_land_background(ax, raster.bbox, raster.crs, zorder=0, margin_frac=0.0)

    # Create masked array for NaN handling
    backscatter_masked = np.ma.masked_invalid(backscatter)

    # gamma != 1.0: PowerNorm warps the color mapping only, so the plotted
    # array and colorbar stay in the raster's real units (never renormalized
    # to [0, 1]) — matches viz/waterfall.py's plot_waterfall().
    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax) if gamma != 1.0 else None
    scale_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}

    # Plot
    im = ax.pcolormesh(
        xs,
        ys,
        backscatter_masked,
        cmap=resolve_cmap(cmap),
        shading="auto",
        zorder=1,
        **scale_kwargs,
    )

    # Labels and formatting
    xlabel, ylabel = _axis_labels(raster)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = (
            f"{raster.channel_name} — "
            f"{raster.dataset.attrs.get('resampling_method', 'unknown')} "
            f"(resolution: {raster.resolution_m:g} m)\n"
            f"CRS: {_crs_label(raster)}"
        )
    ax.set_title(title)

    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_aspect("equal")

    if show_colorbar:
        units = raster.dataset["backscatter"].attrs.get("units", "unknown units")
        plt.colorbar(im, ax=ax, label=f"Backscatter ({units})")
        logger.debug(f"Colorbar range: {vmin:.3f} -- {vmax:.3f}")

    if show_locator:
        from sidescanpy.viz.basemap import plot_locator_thumbnail

        south, west, north, east = raster.bbox
        if raster.is_geographic:
            bbox_wgs84 = (west, south, east, north)
        else:
            from pyproj import Transformer

            to_wgs84 = Transformer.from_crs(raster.crs, "EPSG:4326", always_xy=True)
            lons, lats = to_wgs84.transform([west, east], [south, north])
            bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
        center_lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2.0
        center_lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2.0
        inset_ax = ax.inset_axes([0.02, 0.02, 0.22, 0.22])
        plot_locator_thumbnail(center_lon, center_lat, mark_bbox_wgs84=bbox_wgs84, ax=inset_ax, fig=fig)

    if show_scale_bar:
        if raster.is_geographic:
            logger.debug("add_scale_bar: skipped, raster is geographic (meters-per-degree isn't a single constant)")
        else:
            from sidescanpy.viz.scalebar import add_scale_bar
            add_scale_bar(ax, meters_per_data_unit=1.0)

    return fig, ax
