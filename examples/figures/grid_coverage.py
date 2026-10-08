"""
Figure recipe: a gridded raster with each cell's opacity set by how many
samples contributed to it (``n_contributing_samples``).

This is a diagnostic view, not a data product. It typically shows thin
"streaks" fanning out from the track, especially at resolutions finer
than the ping spacing: that is normal per-cell sample-density variation,
not misplaced data. Judge the raster itself with
``sidescanpy.viz.plot_gridded_raster``.

``plot_grid_coverage`` is the reusable part; ``main`` grids a corrected
.ss.h5 channel and saves the view next to the opaque raster.

Usage:
    python examples/figures/grid_coverage.py survey.ss.h5 --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy.core.units import mask_floor_clip
from sidescanpy.grids import (
    SideScanGriddedRaster,
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct, slice_by_pings
from sidescanpy.viz import plot_gridded_raster, resolve_cmap

logger = logging.getLogger("grid_coverage")


def plot_grid_coverage(
    raster: SideScanGriddedRaster,
    *,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    title: str | None = None,
):
    """Raster colored by backscatter, alpha = contributing-sample count
    relative to the busiest cell (finite/NaN when the raster carries no
    counts). Returns ``(fig, ax)``."""
    fig, ax = plt.subplots(figsize=(10, 8)) if ax is None else (ax.figure, ax)
    backscatter = mask_floor_clip(raster.backscatter)
    if "n_contributing_samples" in raster.dataset.data_vars:
        counts = raster.dataset["n_contributing_samples"].values
    else:
        counts = np.isfinite(backscatter).astype(int)

    finite = backscatter[np.isfinite(backscatter)]
    if vmin is None:
        vmin = np.percentile(finite, vmin_pct) if finite.size else 0
    if vmax is None:
        vmax = np.percentile(finite, vmax_pct) if finite.size else 1

    max_count = np.max(counts)
    colors = resolve_cmap(cmap)(mcolors.Normalize(vmin=vmin, vmax=vmax)(backscatter))
    colors[..., 3] = counts / max_count if max_count > 0 else 1.0

    xs, ys = raster.x_values, raster.y_values
    ax.imshow(colors, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin="lower", aspect="auto",
              interpolation="nearest")
    geographic = raster.is_geographic
    ax.set_xlabel("Longitude (°E)" if geographic else "Easting (m)")
    ax.set_ylabel("Latitude (°N)" if geographic else "Northing (m)")
    ax.set_title(title or f"{raster.channel_name} (coverage: {raster.data_coverage:.1%})")
    ax.grid(True, alpha=0.2, linestyle="--")
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", help="freq_key (default: the first one)")
    parser.add_argument("--pings", type=int, default=2000, help="Use the first N pings")
    parser.add_argument("--resolution-m", type=float, default=1.0)
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    survey = read_survey(args.input, chunks={"ping": 512})
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = slice_by_pings(survey.get_frequency(freq_key).dataset, stop=args.pings).load()
    ds = correct(ds, ["tvg", "geometric", "georef"])
    grid_def = utm_grid_from_wgs84_bbox(bbox_from_channel_dataset(ds), resolution_m=args.resolution_m)
    raster = resample_channel_to_grid(ds, grid_def)
    logger.info("%s / %s: %.0f%% of cells covered", args.input.name, freq_key, 100 * raster.data_coverage)

    fig, (ax_raster, ax_coverage) = plt.subplots(1, 2, figsize=(18, 8))
    plot_gridded_raster(raster, fig=fig, ax=ax_raster, cmap="copper", title="Raster")
    plot_grid_coverage(raster, ax=ax_coverage, cmap="copper", title="Opacity = contributing samples")
    fig.tight_layout()
    fig.savefig(args.output_dir / "grid_coverage.png", dpi=120)
    plt.close(fig)
    logger.info("wrote grid_coverage.png to %s", args.output_dir)


if __name__ == "__main__":
    main()
