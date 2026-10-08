# Gridding

`sidescanpy.grids` resamples georeferenced backscatter onto a regular
raster. The input needs `latitude_per_sample` / `longitude_per_sample`
(the `georef` correction step).

```python
from sidescanpy.grids import grid_channel, recommend_resolution, to_geotiff

res = recommend_resolution(ds)            # half the finer of ping spacing and sample interval
raster = grid_channel(ds, resolution_m=res, method="median")
to_geotiff(raster, "survey.tif")          # needs the [geo] extra
```

## Pieces

| Object | Purpose |
|---|---|
| `GridDefinition` | bounding box, resolution and CRS of a grid |
| `utm_grid_from_wgs84_bbox(bbox, resolution_m)` | a grid in the UTM zone containing the bbox centre |
| `bbox_from_channel_dataset(ds)` | a padded (south, west, north, east) box around the data |
| `resample_channel_to_grid(ds, grid_def, method=...)` | the resampler |
| `grid_channel(ds, resolution_m, method)` | the two steps above in one call, on a UTM grid fit to the data |
| `SideScanGriddedRaster` | the result: an `xr.Dataset` wrapper holding `backscatter`, with its CRS, units, cell counts and `data_coverage` |
| `to_geotiff(raster, path)` | single-band GeoTIFF via rioxarray |

## Methods

`ResamplingMethod`: `mean`, `median`, `max`, `nearest`, `idw`. Each sample
is an independent observation; port and starboard are never averaged
together before gridding. Samples within a search radius of each cell
centre contribute (radius derived from the resolution unless
`max_search_radius_m` is given); empty cells are NaN.

Projected grids name their coordinates `northing` / `easting` (meters);
geographic grids use `latitude` / `longitude`.

## Notes

- A UTM grid uses the zone containing the bbox centre. A survey wider
  than a zone puts its edges in a neighbouring zone's frame.
- `bbox_from_channel_dataset` pads by a fixed amount in degrees, which is
  negligible for a survey but large for a very small test dataset.
