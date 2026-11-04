"""
Survey track + scan-line plotting.

Full-survey overview: the vessel's nadir track plus periodic "rungs"
showing the port/starboard scan-line (swath edge) positions, in a
projected coordinate system (UTM by default). Each rung is split at
nadir and its two halves are colored by side (port red, starboard
green by default, the navigation-light convention). Distinct from
plot_waterfall (per-ping backscatter imagery) and plot_gridded_raster
(resampled backscatter raster) — this shows geometry/coverage only, no
backscatter values.

``sidescanpy.processing.build_survey_track`` builds the lat/lon arrays
this plot needs from a multi-file survey dict.

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
    show_land: bool = False,
    show_locator: bool = False,
    show_scale_bar: bool = True,
    scan_nadir_lat: Optional[np.ndarray] = None,
    scan_nadir_lon: Optional[np.ndarray] = None,
    port_color: str = "tab:red",
    starboard_color: str = "tab:green",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a survey's nadir track plus periodic port/starboard scan-line
    "rungs", in a projected coordinate system. Each rung is drawn as two
    halves, nadir to port edge and nadir to starboard edge, colored by
    side.

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
    show_land : bool, default False
        Draw land/water context (vendored 1:50m Natural Earth data)
        behind the track — see ``sidescanpy.viz.basemap``. Requires the
        optional ``maps`` extra (cartopy); raises ``ImportError`` with
        an install hint if it isn't installed.
    show_locator : bool, default False
        Add a small inset (bottom-left) showing this survey's location
        zoomed way out, with a marker box. Requires the optional
        ``maps`` extra.
    show_scale_bar : bool, default True
        Draw a scale bar (see ``sidescanpy.viz.scalebar``). Only drawn
        for a projected ``crs`` (the default) -- a caller-supplied
        geographic CRS is skipped, not drawn wrong, since meters-per-
        degree isn't a single constant this function can infer.
    scan_nadir_lat, scan_nadir_lon : array, optional
        Nadir position at each rung (WGS84 degrees), index-aligned with
        the scan arrays (``SurveyTrack.scan_nadir_*``). Where the rung
        splits into its two halves. Defaults to the midpoint of each
        rung, which is right when both sides have the same range.
    port_color, starboard_color : str, default "tab:red", "tab:green"
        Matplotlib colors for the port and starboard halves.

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

    if show_land:
        from sidescanpy.viz.basemap import add_land_background

        all_x = np.concatenate([track_x, port_x, stbd_x])
        all_y = np.concatenate([track_y, port_y, stbd_y])
        finite = np.isfinite(all_x) & np.isfinite(all_y)
        bbox = (float(np.min(all_y[finite])), float(np.min(all_x[finite])), float(np.max(all_y[finite])), float(np.max(all_x[finite])))
        add_land_background(ax, bbox, crs, zorder=0)

    n_rungs = len(port_x)
    if n_rungs > 0:
        if scan_nadir_lat is not None and scan_nadir_lon is not None:
            nadir_x, nadir_y = _project(scan_nadir_lat, scan_nadir_lon, crs)
        else:
            nadir_x, nadir_y = (port_x + stbd_x) / 2.0, (port_y + stbd_y) / 2.0
        nadir = np.column_stack([nadir_x, nadir_y])
        for edge_x, edge_y, color, label in (
            (port_x, port_y, port_color, "Port swath"),
            (stbd_x, stbd_y, starboard_color, "Starboard swath"),
        ):
            halves = np.stack([nadir, np.column_stack([edge_x, edge_y])], axis=1)
            ax.add_collection(LineCollection(
                halves, colors=color, linewidths=0.6, alpha=0.6, zorder=1, label=label,
            ))
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

    if show_locator:
        from sidescanpy.viz.basemap import plot_locator_thumbnail

        all_lon = np.concatenate([track_lon, scan_port_lon, scan_stbd_lon])
        all_lat = np.concatenate([track_lat, scan_port_lat, scan_stbd_lat])
        finite = np.isfinite(all_lon) & np.isfinite(all_lat)
        bbox_wgs84 = (float(np.min(all_lon[finite])), float(np.min(all_lat[finite])), float(np.max(all_lon[finite])), float(np.max(all_lat[finite])))
        center_lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2.0
        center_lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2.0
        inset_ax = ax.inset_axes([0.02, 0.02, 0.22, 0.22])
        plot_locator_thumbnail(center_lon, center_lat, mark_bbox_wgs84=bbox_wgs84, ax=inset_ax, fig=fig)

    if show_scale_bar:
        if is_geo:
            logger.debug("add_scale_bar: skipped, crs is geographic (meters-per-degree isn't a single constant)")
        else:
            from sidescanpy.viz.scalebar import add_scale_bar
            add_scale_bar(ax, meters_per_data_unit=1.0)

    return fig, ax
