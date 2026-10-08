"""
Grid a corrected .ss.h5 channel onto a UTM raster and write it as a
GeoTIFF (and a PNG preview).

The GeoTIFF needs ``rioxarray`` (``pip install rioxarray``); without it
the raster is still gridded and previewed, and the GeoTIFF is skipped
with a warning.

Usage:
    python examples/gridding_and_geotiff.py survey.ss.h5 --output-dir out/ --resolution-m 0.5

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from sidescanpy.grids import (
    ResamplingMethod,
    SideScanGriddedRaster,
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct
from sidescanpy.viz import plot_gridded_raster

logger = logging.getLogger("gridding_and_geotiff")


def raster_to_dataarray(raster: SideScanGriddedRaster) -> xr.DataArray:
    """The raster's backscatter as a north-up ``(y, x)`` DataArray with 1-D
    cell-center coordinates in the raster's CRS."""
    ys = raster.y_values[:, 0]
    xs = raster.x_values[0, :]
    da = xr.DataArray(raster.backscatter, dims=("y", "x"), coords={"y": ys, "x": xs}, name="backscatter")
    da.attrs["units"] = raster.dataset["backscatter"].attrs.get("units", "")
    return da.sortby("y", ascending=False)


def write_geotiff(raster: SideScanGriddedRaster, path: Path) -> bool:
    """Write ``raster`` as a single-band float32 GeoTIFF. False if
    rioxarray is not installed."""
    try:
        import rioxarray  # noqa: F401  (registers the .rio accessor)
    except ImportError:
        logger.warning("rioxarray is not installed; skipping %s (pip install rioxarray)", path.name)
        return False
    da = raster_to_dataarray(raster).astype("float32")
    da = da.rio.write_crs(raster.crs).rio.write_nodata(float("nan"))
    da.rio.to_raster(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", help="freq_key, e.g. hz_1495000 (default: the first one)")
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--method", choices=[m.value for m in ResamplingMethod], default="mean")
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load one channel. Gridding needs per-sample lat/lon, so make sure
    #    tvg -> geometric -> georef have run (already-applied steps are skipped).
    survey = read_survey(args.input)
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = correct(survey.get_frequency(freq_key).dataset.load(), ["tvg", "geometric", "georef"])

    # 2. Fit a UTM grid to the swath and resample onto it.
    grid_def = utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(ds), resolution_m=args.resolution_m)
    raster = resample_channel_to_grid(ds, grid_def, method=ResamplingMethod(args.method))
    logger.info(
        "%s / %s: %d x %d cells at %g m (%s), %.0f%% covered, CRS %s",
        args.input.name, freq_key, raster.num_cells_ns, raster.num_cells_ew, args.resolution_m, args.method,
        100 * raster.data_coverage, raster.crs.to_epsg(),
    )

    # 3. Preview PNG and GeoTIFF.
    stem = f"{args.input.name.removesuffix('.ss.h5')}_{freq_key}"
    fig, _ax = plot_gridded_raster(raster, cmap="copper")
    fig.savefig(args.output_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    if write_geotiff(raster, args.output_dir / f"{stem}.tif"):
        logger.info("wrote %s.tif", stem)
    logger.info("done -> %s", args.output_dir)


if __name__ == "__main__":
    main()
