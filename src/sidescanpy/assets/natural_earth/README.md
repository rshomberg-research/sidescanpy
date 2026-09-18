# Natural Earth land polygons (vendored)

`ne_50m_land.*` — 1:50m-scale world land vector data from
[Natural Earth](https://www.naturalearthdata.com/), used by
`sidescanpy.viz.basemap` to draw a land/water background behind a
georeferenced plot and a zoomed-out locator thumbnail.

**Public domain, generic world cartography — no survey data.** Natural
Earth states "No permission is needed to use Natural Earth."

## Why vendored instead of downloaded at runtime

`cartopy.io.shapereader.natural_earth()` downloads on first use — vendoring
this ~1MB shapefile directly in the package means `viz.basemap` never
needs network access at all, consistent with this library being used in
network-constrained/offline deployments.

## Resolution

1:50,000,000 — coarse (a coastline resolves as a smooth simplified
outline, not a survey-scale shoreline). Good enough for "is this segment
near land" orientation and a regional locator thumbnail; not meant to
substitute for real chart data at sidescan survey scales (tens to
thousands of meters).

## Regenerating / updating resolution

```python
import cartopy.io.shapereader as shpreader
print(shpreader.natural_earth(resolution="50m", category="physical", name="land"))
```
Copy the returned `.shp`/`.shx`/`.dbf`/`.prj`/`.cpg` files here.
