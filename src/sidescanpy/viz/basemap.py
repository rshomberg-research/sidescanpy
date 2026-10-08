"""
Land/water context for georeferenced plots.

Two things live here: a background layer (land/water fill) to draw
*behind* an already-projected plot (a gridded raster, a survey track),
and a small standalone locator thumbnail (zoomed way out, with a marker
showing where the main plot sits) — "where is this, relative to the
coast." Neither one interprets or corrects backscatter; this is
orientation context only.

Land geometry comes from a vendored 1:50m Natural Earth shapefile
(``assets/natural_earth/``, public domain) — coarse by design (a
coastline resolves as a smooth simplified outline, not a survey-scale
shoreline). Good enough to tell "near land" from "open water" and to
place a survey on a regional map; not a substitute for real chart data
at sidescan survey scales (tens to thousands of meters). See that
directory's own README for provenance/regeneration.

Requires the optional ``maps`` extra (cartopy) — every function here
raises ``ImportError`` with a clear message if it isn't installed,
rather than silently skipping the background (which could be mistaken
for "there's no land nearby").

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
import math
from importlib.resources import files
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Rectangle
from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)
_LAND_COLOR = "#ddd2ba"
_WATER_COLOR = "#cfe3ec"
_LAND_SHAPEFILE = "ne_50m_land.shp"

try:
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.path import shapely_to_path
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


def _require_cartopy() -> None:
    if not _HAS_CARTOPY:
        raise ImportError(
            "sidescanpy.viz.basemap requires the optional 'maps' extra (cartopy) -- "
            "install with `pip install sidescanpy[maps]` or `uv sync --extra maps`."
        )


def _land_geometries_wgs84(bbox_wgs84: Tuple[float, float, float, float]) -> list:
    """``bbox_wgs84``: (west, south, east, north) degrees. Coarse
    bounding-box pre-filter (not an exact clip -- matplotlib clips to
    the axes' own limits regardless) over the vendored land polygons."""
    _require_cartopy()
    path = files("sidescanpy.assets.natural_earth") / _LAND_SHAPEFILE
    reader = shpreader.Reader(str(path))
    west, south, east, north = bbox_wgs84
    out = []
    for geom in reader.geometries():
        gw, gs, ge, gn = geom.bounds
        if ge < west or gw > east or gn < south or gs > north:
            continue
        out.append(geom)
    return out


def _draw_geometries(ax: plt.Axes, geoms: list, *, color: str, zorder: int) -> None:
    if not geoms:
        return
    patches = [PathPatch(shapely_to_path(geom)) for geom in geoms]
    ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor="none", zorder=zorder))


def add_land_background(
    ax: plt.Axes,
    bbox: Tuple[float, float, float, float],
    crs: CRS,
    *,
    land_color: str = _LAND_COLOR,
    water_color: str = _WATER_COLOR,
    zorder: int = 0,
    margin_frac: float = 0.08,
) -> None:
    """Fill ``ax`` with a water color, draw land polygons on top of it,
    and set the axes' data limits to ``bbox`` (plus a margin) -- meant to
    be called *before* the caller's own real data is plotted at a higher
    ``zorder``, so land/water sits behind it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bbox : (south, west, north, east)
        In ``crs``'s own units -- matches
        ``sidescanpy.grids.raster.SideScanGriddedRaster.bbox``'s own
        convention directly. Degrees for a geographic CRS, meters for a
        projected one (e.g. UTM).
    crs : pyproj.CRS
        The CRS ``ax``'s existing data (if any) is already plotted in.
    margin_frac : float, default 0.08
        Extra margin around ``bbox``, as a fraction of its own
        height/width, so land right at the requested edge doesn't look
        abruptly clipped.
    """
    _require_cartopy()
    south, west, north, east = bbox
    dy = (north - south) * margin_frac
    dx = (east - west) * margin_frac
    south, north = south - dy, north + dy
    west, east = west - dx, east + dx

    ax.set_facecolor(water_color)

    if crs.is_geographic:
        geoms = _land_geometries_wgs84((west, south, east, north))
        _draw_geometries(ax, geoms, color=land_color, zorder=zorder)
    else:
        to_wgs84 = Transformer.from_crs(crs, _WGS84, always_xy=True)
        corner_x = [west, east, west, east]
        corner_y = [south, south, north, north]
        lons, lats = to_wgs84.transform(corner_x, corner_y)
        # Generous padding: a projected bbox's corners don't map back to
        # a tight WGS84 bbox in general (rotation/distortion), so this
        # over-selects candidates rather than risk missing real land —
        # matplotlib's own axes limits (set below) do the actual clipping.
        pad_lon = (max(lons) - min(lons)) * 0.25 + 0.05
        pad_lat = (max(lats) - min(lats)) * 0.25 + 0.05
        candidates = _land_geometries_wgs84(
            (min(lons) - pad_lon, min(lats) - pad_lat, max(lons) + pad_lon, max(lats) + pad_lat)
        )
        from shapely.ops import transform as shp_transform

        to_crs = Transformer.from_crs(_WGS84, crs, always_xy=True)
        projected = [shp_transform(lambda x, y: to_crs.transform(x, y), geom) for geom in candidates]
        _draw_geometries(ax, projected, color=land_color, zorder=zorder)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    # Without this, whatever the caller plots on top next (pcolormesh,
    # ax.plot, add_collection) triggers matplotlib's own autoscale and
    # silently overrides the view back to that data's own tight bounds
    # -- which can be *larger* than requested (a big landmass extending
    # past bbox) or *smaller* (undoing the margin above). This background
    # is meant to set the view once, deliberately, not participate in
    # autoscaling.
    ax.set_autoscale_on(False)


def plot_locator_thumbnail(
    center_lon: float,
    center_lat: float,
    *,
    region_km: float = 50.0,
    mark_bbox_wgs84: Optional[Tuple[float, float, float, float]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (2.4, 2.4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Land/water context zoomed out to roughly ``region_km`` across,
    centered on ``(center_lon, center_lat)``, with a red marker showing
    where the "real" plot (a gridded segment, a survey track) sits
    within that region -- "where is this, relative to the coast."

    Draws into ``ax`` if given (e.g. an inset axes embedded in a larger
    figure via ``fig.add_axes([...])``/``ax.inset_axes([...])`` --
    matches this module's usual ``fig=None, ax=None`` convention), or
    creates its own small standalone figure otherwise.

    Parameters
    ----------
    center_lon, center_lat : float
        WGS84 degrees -- typically the survey/segment's own centroid.
    region_km : float, default 50.0
        Approximate width/height of the thumbnail's own view, in
        kilometers (converted to degrees locally via a flat-earth
        approximation -- fine at this zoomed-out, orientation-only
        scale, not meant for precise measurement).
    mark_bbox_wgs84 : (west, south, east, north), optional
        The real plot's own extent, drawn as a red box. If it would be
        too small to see at this thumbnail's scale, a red dot is drawn
        at its center instead (a box a fraction of a pixel wide is
        indistinguishable from nothing, not from "precisely located").
    fig, ax : optional
        Existing figure/axes to draw into.
    figsize : (float, float), default (2.4, 2.4)
        Figure size in inches, only used when creating a new figure.

    Returns
    -------
    fig, ax
    """
    _require_cartopy()
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    lat_rad = math.radians(center_lat)
    half_deg_lat = (region_km / 2.0) / 111.32
    half_deg_lon = (region_km / 2.0) / (111.32 * max(math.cos(lat_rad), 0.1))
    bbox = (center_lat - half_deg_lat, center_lon - half_deg_lon, center_lat + half_deg_lat, center_lon + half_deg_lon)

    add_land_background(ax, bbox, _WGS84, margin_frac=0.0)

    if mark_bbox_wgs84 is not None:
        west, south, east, north = mark_bbox_wgs84
        w_px = east - west
        h_px = north - south
        if w_px < half_deg_lon * 0.06 or h_px < half_deg_lat * 0.06:
            ax.plot([(west + east) / 2], [(south + north) / 2], marker="o", color="#c0392b", markersize=5, zorder=5)
        else:
            ax.add_patch(Rectangle((west, south), w_px, h_px, fill=False, edgecolor="#c0392b", linewidth=1.4, zorder=5))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
    ax.set_aspect("equal")
    return fig, ax
