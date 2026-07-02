"""
Survey track + scan-line plotting.

Full-survey overview: the vessel's nadir track plus periodic "rungs"
showing the port/starboard scan-line (swath edge) positions, in a
projected coordinate system (UTM by default). Distinct from
plot_waterfall (per-ping backscatter imagery) and plot_gridded_raster
(resampled backscatter raster) — this shows geometry/coverage only, no
backscatter values.

2026 rshomberg@uri.edu
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)


def _project(lat: np.ndarray, lon: np.ndarray, crs: CRS) -> Tuple[np.ndarray, np.ndarray]:
    """WGS84 (lat, lon) -> (x, y) in crs. Identity (lon, lat) if crs is geographic."""
    if crs.is_geographic:
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    transformer = Transformer.from_crs(_WGS84, crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def plot_track_and_scanlines(
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    scan_port_lat: np.ndarray,
    scan_port_lon: np.ndarray,
    scan_stbd_lat: np.ndarray,
    scan_stbd_lon: np.ndarray,
    *,
    crs: Optional[CRS] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a survey's nadir track plus periodic port/starboard scan-line
    "rungs", in a projected coordinate system.

    Parameters
    ----------
    track_lat, track_lon : array
        Per-ping nadir position (WGS84 degrees) — one point per ping,
        drawn as a continuous line.
    scan_port_lat, scan_port_lon, scan_stbd_lat, scan_stbd_lon : array
        Port/starboard swath-edge position (WGS84 degrees), same
        length, index-aligned — typically a strided subset of pings
        (drawing a rung per ping is unreadable/slow for a full
        survey). Each index draws one port-to-starboard line segment.
    crs : pyproj.CRS, optional
        Target CRS to project into. Defaults to an auto-selected UTM
        zone from the track's centroid (see
        ``sidescanpy.grids.utm_crs_for_lonlat``).
    fig, ax : optional
        Existing figure/axes to draw into.
    title : str, optional
        Plot title. Default includes point/rung counts and the CRS.

    Returns
    -------
    fig, ax
    """
    if crs is None:
        from sidescanpy.grids.definition import utm_crs_for_lonlat
        crs = utm_crs_for_lonlat(
            float(np.nanmean(track_lon)), float(np.nanmean(track_lat))
        )

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    track_x, track_y = _project(track_lat, track_lon, crs)
    port_x, port_y = _project(scan_port_lat, scan_port_lon, crs)
    stbd_x, stbd_y = _project(scan_stbd_lat, scan_stbd_lon, crs)

    n_rungs = len(port_x)
    if n_rungs > 0:
        segments = np.stack(
            [np.column_stack([port_x, port_y]), np.column_stack([stbd_x, stbd_y])],
            axis=1,
        )
        lc = LineCollection(
            segments, colors="steelblue", linewidths=0.6, alpha=0.6,
            zorder=1, label="Scan line (port–starboard)",
        )
        ax.add_collection(lc)
        # LineCollection doesn't update axis data limits on its own.
        ax.plot(port_x, port_y, alpha=0, zorder=1)
        ax.plot(stbd_x, stbd_y, alpha=0, zorder=1)

    ax.plot(track_x, track_y, color="black", linewidth=1.0, label="Track (nadir)", zorder=2)

    is_geo = crs.is_geographic
    ax.set_xlabel("Longitude (°E)" if is_geo else "Easting (m)")
    ax.set_ylabel("Latitude (°N)" if is_geo else "Northing (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    if title is None:
        epsg = crs.to_epsg()
        crs_label = f"{crs.name} (EPSG:{epsg})" if epsg is not None else crs.name
        title = (
            f"Survey track ({len(track_x):,} pings) + "
            f"{n_rungs:,} scan line(s)\nCRS: {crs_label}"
        )
    ax.set_title(title)

    return fig, ax


def export_track_and_scanlines_png(
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    scan_port_lat: np.ndarray,
    scan_port_lon: np.ndarray,
    scan_stbd_lat: np.ndarray,
    scan_stbd_lon: np.ndarray,
    output_path: str,
    *,
    crs: Optional[CRS] = None,
    dpi: int = 150,
) -> None:
    """Plot and save a survey track + scan lines PNG. See plot_track_and_scanlines."""
    fig, ax = plot_track_and_scanlines(
        track_lat, track_lon,
        scan_port_lat, scan_port_lon,
        scan_stbd_lat, scan_stbd_lon,
        crs=crs,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Exported track + scan lines to {output_path}")
    plt.close(fig)
