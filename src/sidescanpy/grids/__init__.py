"""
Gridding: resample georeferenced backscatter onto a regular raster.

Needs per-sample positions (the ``georef`` correction step).
:func:`grid_channel` grids one channel on a UTM grid fitted to the data;
:func:`resample_channel_to_grid` takes any :class:`GridDefinition`.
The result is a :class:`SideScanGriddedRaster`, which :func:`to_geotiff`
writes as a GeoTIFF (needs the ``geo`` extra).
"""

from sidescanpy.grids.definition import (
    GridDefinition,
    bbox_from_channel_dataset,
    utm_crs_for_lonlat,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.grids.geotiff import to_geotiff
from sidescanpy.grids.raster import SideScanGriddedRaster
from sidescanpy.grids.resample import ResamplingMethod, grid_channel, resample_channel_to_grid
from sidescanpy.grids.resolution import recommend_resolution

__all__ = [
    "GridDefinition",
    "bbox_from_channel_dataset",
    "utm_crs_for_lonlat",
    "utm_grid_from_wgs84_bbox",
    "SideScanGriddedRaster",
    "ResamplingMethod",
    "resample_channel_to_grid",
    "grid_channel",
    "recommend_resolution",
    "to_geotiff",
]
