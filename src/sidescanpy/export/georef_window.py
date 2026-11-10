"""
Georeferenced (northing x easting meters) windowed export.

Resamples a bounded square region directly at an explicit
``(resolution_m, size_px)``, centered on a given real-world lat/lon —
builds on the survey-wide georeferencing/gridding path already in
``sidescanpy.grids``. Produces a ``SideScanGriddedRaster`` (an in-memory
xarray-backed raster with northing/easting coordinates and CRS
metadata), not a file; :func:`sidescanpy.grids.to_geotiff` writes it as
a GeoTIFF (needs the ``geo`` extra).

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging

import numpy as np
import pyproj
import xarray as xr

from sidescanpy.core.history import STEP_TVG, has_step
from sidescanpy.core.history import processing_level as derived_processing_level
from sidescanpy.export.geometric_window import interpolate_scattered
from sidescanpy.grids.definition import GridDefinition, utm_crs_for_lonlat
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.grids.resample import ResamplingMethod, resample_channel_to_grid

logger = logging.getLogger(__name__)


def extract_georef_window(
    channel_ds: xr.Dataset,
    size_px: int,
    *,
    resolution_m: float,
    center_lat: float,
    center_lon: float,
    method: ResamplingMethod = ResamplingMethod.MEAN,
    backscatter_var: str = "backscatter_intensity",
    max_search_radius_m: float | None = None,
    interp_method: str | None = None,
) -> tuple[SideScanGriddedRaster, dict]:
    """Resample a ``size_px x size_px`` square region, centered on
    ``(center_lat, center_lon)``, at exactly ``resolution_m`` meters/pixel
    on both axes — built directly in the auto-selected UTM CRS (not via
    a WGS84 bbox roundtrip), so ``size_px`` and ``resolution_m`` alone
    determine the real-world extent, matching
    :func:`sidescanpy.export.geometric_window.extract_geometric_window`'s
    contract exactly. Port and starboard are pooled as independent
    observations by design (``resample_channel_to_grid``'s own
    convention) — this is the one mode where a channel side can't be
    selected.

    Parameters
    ----------
    channel_ds : xr.Dataset
        Already ping-sliced to a neighborhood comfortably containing the
        requested extent around the center.
    size_px : int
        Output side length in pixels.
    resolution_m : float
        Meters per pixel, both axes.
    center_lat, center_lon : float
        Real-world center of the requested window, WGS84 degrees (e.g. a
        target's own position, read directly from
        ``latitude_per_sample``/``longitude_per_sample`` at its
        ping/sample/channel).
    max_search_radius_m : float, optional
        Passed straight through to ``resample_channel_to_grid`` (defaults
        to its own ``1.5 * resolution_m``) -- see
        :func:`sidescanpy.export.geometric_window.extract_geometric_window`'s
        own ``search_radius_m`` docstring for when a wider radius is
        needed (a survey whose along-track position updates far less
        often than its ping rate).
    interp_method : {"linear", "cubic", "nearest"}, optional
        Interpolate with ``scipy.interpolate.griddata``
        (:func:`~sidescanpy.export.geometric_window.interpolate_scattered`)
        instead of aggregating within a search radius (``method``). Cells
        farther than ``max_search_radius_m`` (default ``1.5 *
        resolution_m``) from every sample are NaN. Radius aggregation
        spreads each sample over a disk of that radius, which blurs a
        window finer than the ping spacing and turns isolated dark samples
        into dark disks; interpolation keeps the detail the samples have.

    Returns
    -------
    (raster, meta)
        ``meta`` has ``center_row``/``center_col`` (the requested
        center's own pixel position — always the exact middle, by
        construction) for immediate use, plus enough to locate a
        *different* real-world point in this same raster later via
        :func:`locate_point`.
    """
    crs = utm_crs_for_lonlat(center_lon, center_lat)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_easting, center_northing = transformer.transform(center_lon, center_lat)

    half_extent = size_px * resolution_m / 2.0
    bbox = (
        center_northing - half_extent,
        center_easting - half_extent,
        center_northing + half_extent,
        center_easting + half_extent,
    )
    grid_def = GridDefinition(bbox, resolution_m, crs=crs)
    if interp_method is None:
        raster = resample_channel_to_grid(
            channel_ds, grid_def, method=method, backscatter_var=backscatter_var, max_search_radius_m=max_search_radius_m
        )
    else:
        fill_m = 1.5 * resolution_m if max_search_radius_m is None else max_search_radius_m
        raster = _georef_griddata(channel_ds, grid_def, transformer, backscatter_var, interp_method, fill_m)

    meta = {
        "crs": crs,
        "easting_min": bbox[1],
        "northing_min": bbox[0],
        "resolution_m": resolution_m,
        "center_row": (center_northing - bbox[0]) / resolution_m,
        "center_col": (center_easting - bbox[1]) / resolution_m,
    }
    return raster, meta


def locate_point(lat: float, lon: float, meta: dict) -> tuple[float, float]:
    """Translate a real-world ``(lat, lon)`` point into pixel
    ``(row, col)`` coordinates within a raster produced by
    :func:`extract_georef_window`, using that call's own returned
    ``meta`` — for marking a *different* point than the one the window
    was centered on.
    """
    # TODO(lib): this is the inverse of the window's affine transform --
    # rasterio.transform.rowcol / ~affine.Affine on a transform carried in
    # meta; parity test: identical fractional (row, col).
    transformer = pyproj.Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return (
        (northing - meta["northing_min"]) / meta["resolution_m"],
        (easting - meta["easting_min"]) / meta["resolution_m"],
    )


def _georef_griddata(
    channel_ds: xr.Dataset, grid_def: GridDefinition, transformer: pyproj.Transformer,
    backscatter_var: str, interp_method: str, max_fill_distance_m: float,
) -> SideScanGriddedRaster:
    """``extract_georef_window``'s griddata path: interpolate in the grid's projected frame."""
    values = channel_ds[backscatter_var].values.astype(np.float64).ravel()
    lat = channel_ds["latitude_per_sample"].values.astype(np.float64).ravel()
    lon = channel_ds["longitude_per_sample"].values.astype(np.float64).ravel()
    ok = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise ValueError("no finite georeferenced samples in this window")
    easting, northing = transformer.transform(lon[ok], lat[ok])
    projected = np.isfinite(easting) & np.isfinite(northing)
    points = np.column_stack([northing[projected], easting[projected]])
    grid_ys, grid_xs = grid_def.get_grid_centers()
    query = np.column_stack([grid_ys.ravel(), grid_xs.ravel()])
    grid = interpolate_scattered(points, values[ok][projected], query, interp_method=interp_method,
                                 max_fill_distance_m=max_fill_distance_m).reshape(grid_ys.shape)
    freq_hz = channel_ds.attrs.get("freq_hz_start", 0)
    units = "dB" if backscatter_var == "backscatter_intensity" and has_step(channel_ds, STEP_TVG) else "linear"
    raster = SideScanGriddedRaster(
        grid_def=grid_def, backscatter=grid, channel_name=f"{freq_hz / 1000:.0f} kHz" if freq_hz else "unknown",
        processing_level=derived_processing_level(channel_ds), units=units,
    )
    raster.dataset.attrs["resampling_method"] = f"griddata-{interp_method}"
    raster.dataset.attrs["backscatter_variable"] = backscatter_var
    return raster
