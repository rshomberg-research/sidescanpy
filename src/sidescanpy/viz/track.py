"""
Survey track + scan-line plotting.

Full-survey overview: the vessel's nadir track plus periodic "rungs"
showing the port/starboard scan-line (swath edge) positions, in a
projected coordinate system (UTM by default). Distinct from
plot_waterfall (per-ping backscatter imagery) and plot_gridded_raster
(resampled backscatter raster) — this shows geometry/coverage only, no
backscatter values.

``build_survey_track`` builds the lat/lon arrays this module's plot
functions need from a multi-file survey dict — the loading/geometry
side of a "track" output; pair it with ``plot_track_and_scanlines``/
``export_track_and_scanlines_png`` above.

2026 rshomberg@uri.edu
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from pyproj import CRS, Transformer

from sidescanpy.processing import apply_correction_steps, interpolate_aux_onto_pings

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
    show_land: bool = False,
    show_locator: bool = False,
    show_scale_bar: bool = True,
) -> None:
    """Plot and save a survey track + scan lines PNG. See plot_track_and_scanlines.

    Deliberately saves at the figure's own full, fixed size (no
    ``bbox_inches="tight"``) -- ``plot_track_and_scanlines`` already
    creates a fixed-size figure (``figsize=(12, 10)``) with
    ``ax.set_aspect("equal")`` (correctly preserving real-world
    geographic proportions -- a long/skinny survey's plotted content is
    legitimately shrunk to fit, not distorted). ``bbox_inches="tight"``
    used to crop the *saved* PNG down to just the drawn content's own
    bounding box, discarding those letterboxed margins -- so a
    wide/square survey (which already filled the canvas) saved close to
    the full size while a long/skinny one got cropped down to a thin
    strip, making saved file dimensions inconsistent across surveys
    despite the figure itself always being created at one fixed size.
    Real user report: every saved track.png should be the same
    dimensions regardless of survey shape, padding is fine.
    """
    fig, ax = plot_track_and_scanlines(
        track_lat, track_lon,
        scan_port_lat, scan_port_lon,
        scan_stbd_lat, scan_stbd_lon,
        crs=crs,
        show_land=show_land,
        show_scale_bar=show_scale_bar,
        show_locator=show_locator,
    )
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Exported track + scan lines to {output_path}")
    plt.close(fig)


def build_survey_track(
    surveys: dict,
    freq_key: str,
    *,
    scanline_stride: int,
) -> Optional[dict]:
    """Nadir track + strided port/starboard scan-line endpoints for one
    frequency, across every file that has it, in time order.

    Cheap by design, but *not* by slicing the sample axis down to one
    sample before georeferencing — that's tempting (we only want the
    farthest-range sample per ping) but wrong:
    ``compute_geometric_ranges`` computes each sample's position from
    ``np.arange(ds.sizes["sample"])``, a *local* index into whatever
    sample axis it's handed, not the sample's true position in the
    original ping. Pre-slicing to ``sample=[-1]`` collapses the sample
    axis to length 1, so the function sees "sample 0" (point-blank
    range) instead of the true far-range edge — silently producing NaN
    via the water-column mask instead of a real position.

    The safe optimization instead pre-selects the *strided pings*
    (slicing the ping axis doesn't touch ``ds.sizes["sample"]``, so it
    doesn't affect the sample-index math) while keeping each selected
    ping's full sample axis intact through georeferencing, and only
    slices to the last sample *afterward*. This still avoids
    georeferencing every ping — just the ones actually needed for a
    rung — while staying correct.

    
    Parameters
    ----------
    surveys : dict[Path, SideScanSurvey]
    freq_key : str
        e.g. ``"hz_1495000"``.
    scanline_stride : int
        Draw a scan-line rung every Nth ping.

    Returns
    -------
    dict or None
        None if no file has ``freq_key``. Otherwise
        ``{"track_lat", "track_lon", "scan_port_lat", "scan_port_lon",
        "scan_stbd_lat", "scan_stbd_lon"}`` — track arrays cover every
        ping; scan arrays cover every ``scanline_stride``-th ping.
    """
    entries = [(p, s) for p, s in surveys.items() if freq_key in s.frequencies]
    if not entries:
        return None

    entries = [(p, interpolate_aux_onto_pings(s)) for p, s in entries]
    entries.sort(
        key=lambda item: item[1].get_frequency(freq_key).dataset["time"].values[0]
    )

    track_lat_parts, track_lon_parts = [], []
    edge_datasets = []
    for _p, survey in entries:
        ds = survey.get_frequency(freq_key).dataset
        track_lat_parts.append(ds["latitude"].values)
        track_lon_parts.append(ds["longitude"].values)

        strided_idx = np.arange(0, ds.sizes["ping"], scanline_stride)
        edge_ds = ds.isel(ping=strided_idx).load()
        edge_ds = apply_correction_steps(edge_ds, {"geometric", "georef"})
        # Each file's sample axis may be a different length (e.g. one
        # file's channel was cropped further than another's by the
        # far-range noise crop stage), so "sample"'s own index label
        # for the retained last sample (e.g. 557 vs. 663) differs
        # across files. Left in place, concatenating along "ping" below
        # aligns on that mismatched "sample" label and NaNs out
        # whichever file doesn't have that particular index value —
        # drop the label since only the value at "the last sample" is
        # wanted here, not which index it came from.
        last_sample = edge_ds.isel(sample=[-1])
        if "sample" in last_sample.coords:
            last_sample = last_sample.drop_vars("sample")
        edge_datasets.append(last_sample)

    track_lat = np.concatenate(track_lat_parts)
    track_lon = np.concatenate(track_lon_parts)

    # Each entry in edge_datasets is already the strided-ping subset for
    # its file (selected before georeferencing, above) — no further
    # striding needed here.
    edge_concat = xr.concat(edge_datasets, dim="ping")
    lat_edge = edge_concat["latitude_per_sample"].values[:, :, 0]   # (ping, channel)
    lon_edge = edge_concat["longitude_per_sample"].values[:, :, 0]

    return {
        "track_lat": track_lat,
        "track_lon": track_lon,
        "scan_port_lat": lat_edge[:, 0],
        "scan_port_lon": lon_edge[:, 0],
        "scan_stbd_lat": lat_edge[:, 1],
        "scan_stbd_lon": lon_edge[:, 1],
    }
