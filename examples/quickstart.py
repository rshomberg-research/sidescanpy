"""
Quickstart: load a .ss.h5 file, correct it, slice it, grid it and save a
waterfall and a gridded-raster PNG.

Usage:
    python examples/quickstart.py survey.ss.h5 --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy.core.history import processing_level
from sidescanpy.grids import (
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_dataset,
    slice_by_pings,
)
from sidescanpy.viz import plot_gridded_raster, plot_waterfall

logger = logging.getLogger("quickstart")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", help="freq_key, e.g. hz_1495000 (default: the first one)")
    parser.add_argument("--pings", type=int, default=2000, help="Use the first N pings")
    parser.add_argument("--resolution-m", type=float, default=1.0, help="Grid cell size")
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load one frequency channel (lazily) and slice the pings we want.
    survey = read_survey(args.input, chunks={"ping": 512})
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = slice_by_pings(survey.get_frequency(freq_key).dataset, stop=args.pings).load()
    logger.info("%s / %s: %d pings x %d samples", args.input.name, freq_key, ds.sizes["ping"], ds.sizes["sample"])

    # 2. Correct: navigation onto the pings, TVG, slant -> ground range,
    #    then per-sample lat/lon. Each call records itself in
    #    processing_history; a file sidescan-correct already processed is
    #    used as it is.
    if processing_level(ds) == "raw":
        ds = interpolate_aux_onto_dataset(ds, survey)
        ds = apply_tvg(ds)
        ds = compute_geometric_ranges(ds)
        ds = compute_georeferencing(ds)
    logger.info("processing level: %s", processing_level(ds))

    # 3. Grid onto a UTM raster fitted to the data.
    grid_def = utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(ds), resolution_m=args.resolution_m)
    raster = resample_channel_to_grid(ds, grid_def)
    logger.info("grid: %d x %d cells, %.0f%% covered", raster.num_cells_ns, raster.num_cells_ew, 100 * raster.data_coverage)

    # 4. Plot a waterfall in real-world distance and the gridded raster.
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_waterfall(ds, x_axis="across_track", y_axis="along_track", ax=ax, show_colorbar=True)
    fig.savefig(args.output_dir / "waterfall.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_gridded_raster(raster, cmap="copper")
    fig.savefig(args.output_dir / "grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote waterfall.png and grid.png to %s", args.output_dir)


if __name__ == "__main__":
    main()
