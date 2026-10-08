"""
GeoTIFF output for gridded rasters.

:func:`to_geotiff` writes one variable of a :class:`SideScanGriddedRaster`
(or of the plain dataset from its ``to_dataset()``) as a single-band,
north-up GeoTIFF with CRS, affine transform and nodata, via rioxarray.
rioxarray is the optional ``geo`` extra (``pip install sidescanpy[geo]``)
and is imported only here, only when called.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from sidescanpy.grids.raster import SideScanGriddedRaster

logger = logging.getLogger(__name__)

_MISSING_RIOXARRAY = (
    "sidescanpy.grids.to_geotiff needs rioxarray, which is not installed. "
    "Install the 'geo' extra: pip install 'sidescanpy[geo]' (or: uv sync --extra geo)."
)


def _axis_step(coords: np.ndarray, extent: float) -> float:
    """Cell spacing along one axis: from the (evenly spaced) centers, or
    the bbox extent for a single-cell axis."""
    if coords.size > 1:
        return float((coords[-1] - coords[0]) / (coords.size - 1))
    return float(extent)


def to_geotiff(
    raster: SideScanGriddedRaster | xr.Dataset,
    path: str | Path,
    *,
    variable: str = "backscatter",
    dtype: str = "float32",
    nodata: float = float("nan"),
    compress: str | None = "deflate",
) -> Path:
    """Write ``variable`` of a gridded raster as a single-band GeoTIFF.

    Parameters
    ----------
    raster : SideScanGriddedRaster or xr.Dataset
        A raster, or the dataset from ``SideScanGriddedRaster.to_dataset()``
        (1-D ``y``/``x`` cell-center coordinates plus a CF ``spatial_ref``
        coordinate holding the CRS).
    path : str or Path
        Output ``.tif`` path; parent directories must exist.
    variable : str, default "backscatter"
        Data variable to write (e.g. ``"n_contributing_samples"``).
    dtype : str, default "float32"
        Output band dtype.
    nodata : float, default NaN
        Nodata value recorded in the file; NaN cells are written as this.
    compress : str or None, default "deflate"
        GDAL compression creation option; None for uncompressed.

    Returns
    -------
    Path
        ``path``.

    Notes
    -----
    Rows are flipped to north-up. The affine transform is built from the
    real cell centers (pixel size = center spacing), which is
    approximately, not exactly, ``resolution_m`` -- see
    ``SideScanGriddedRaster.to_dataset``. The variable's attrs (units,
    long_name, ...) and the raster's channel/processing attrs are written
    as GeoTIFF tags.

    Raises
    ------
    ImportError
        If rioxarray is not installed (``geo`` extra).
    KeyError
        If ``variable`` is not in the dataset.
    """
    try:
        import rioxarray  # noqa: F401  (registers the .rio accessor)
        from affine import Affine
    except ImportError as exc:
        raise ImportError(_MISSING_RIOXARRAY) from exc

    ds = raster.to_dataset() if isinstance(raster, SideScanGriddedRaster) else raster
    if variable not in ds.data_vars:
        raise KeyError(f"{variable!r} not in raster dataset (has: {sorted(ds.data_vars)})")
    if "spatial_ref" not in ds.coords:
        raise ValueError("dataset has no 'spatial_ref' CRS coordinate; build it with SideScanGriddedRaster.to_dataset()")

    da = ds[variable].sortby("y", ascending=False)  # north-up
    ys, xs = da["y"].values, da["x"].values
    dx = _axis_step(xs, float(ds.attrs.get("bbox_east", 0.0)) - float(ds.attrs.get("bbox_west", 0.0)))
    dy = _axis_step(ys[::-1], float(ds.attrs.get("bbox_north", 0.0)) - float(ds.attrs.get("bbox_south", 0.0)))
    transform = Affine(dx, 0.0, float(xs[0]) - dx / 2.0, 0.0, -dy, float(ys[0]) + dy / 2.0)

    out = da.astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.integer) and np.isnan(nodata):
        raise ValueError(f"nodata=NaN is not representable in {dtype}; pass an integer nodata")
    if not np.isnan(nodata):
        out = out.where(np.isfinite(da), nodata).astype(dtype)
    tags = {k: str(v) for k, v in ds.attrs.items()}
    tags.update({k: str(v) for k, v in da.attrs.items() if k != "grid_mapping"})
    out.attrs = {}
    crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs["spatial_ref"]
    out = (
        out.drop_vars("spatial_ref")
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(crs_wkt)
        .rio.write_transform(transform)
        .rio.write_nodata(nodata, encoded=False)
    )
    path = Path(path)
    kwargs = {"tags": tags}
    if compress:
        kwargs["compress"] = compress
    out.rio.to_raster(path, **kwargs)
    logger.debug(f"wrote {variable} {out.shape} GeoTIFF to {path}")
    return path
