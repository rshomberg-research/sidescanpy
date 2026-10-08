"""
Figure recipe: one gridded raster per resampling method, side by side on
a shared color scale.

``plot_method_comparison`` is the reusable part (copy it into your own
project); ``main`` grids a corrected .ss.h5 channel with every method in
:class:`sidescanpy.grids.ResamplingMethod` and saves the comparison.

Usage:
    python examples/figures/compare_grid_methods.py survey.ss.h5 --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy.core.units import mask_floor_clip
from sidescanpy.grids import (
    ResamplingMethod,
    SideScanGriddedRaster,
    bbox_from_channel_dataset,
    resample_channel_to_grid,
    utm_grid_from_wgs84_bbox,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct, slice_by_pings
from sidescanpy.viz import resolve_cmap

logger = logging.getLogger("compare_grid_methods")


def plot_method_comparison(
    rasters: dict[str, SideScanGriddedRaster],
    *,
    figsize: tuple[float, float] = (14, 10),
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
):
    """Grid of panels (up to 3 per row), one per ``{label: raster}``, all
    on one color scale. ``vmin``/``vmax`` default to the
    ``vmin_pct``/``vmax_pct`` percentiles of every raster's finite,
    non-floor-clipped values. Returns ``(fig, axes)`` with ``axes`` 2-D."""
    if vmin is None or vmax is None:
        pooled = np.concatenate([mask_floor_clip(r.backscatter).ravel() for r in rasters.values()])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            vmin = np.percentile(pooled, vmin_pct) if vmin is None else vmin
            vmax = np.percentile(pooled, vmax_pct) if vmax is None else vmax
        else:
            vmin, vmax = 0, 1

    n = len(rasters)
    n_cols = min(3, n)
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for ax, (label, raster) in zip(axes.flat, rasters.items()):
        im = ax.pcolormesh(
            raster.x_values, raster.y_values, np.ma.masked_invalid(mask_floor_clip(raster.backscatter)),
            cmap=resolve_cmap(cmap), vmin=vmin, vmax=vmax, shading="auto",
        )
        geographic = raster.is_geographic
        ax.set_xlabel("Longitude (°E)" if geographic else "Easting (m)", fontsize=9)
        ax.set_ylabel("Latitude (°N)" if geographic else "Northing (m)", fontsize=9)
        ax.set_title(f"{label} (cov: {raster.data_coverage:.1%})", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_aspect("equal")
        units = raster.dataset["backscatter"].attrs.get("units", "unknown units")
        fig.colorbar(im, ax=ax, label=f"Backscatter ({units})")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


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

    rasters = {}
    for method in ResamplingMethod:
        rasters[method.value] = resample_channel_to_grid(ds, grid_def, method=method)
        logger.info("%s: %.0f%% covered", method.value, 100 * rasters[method.value].data_coverage)

    fig, _axes = plot_method_comparison(rasters, cmap="copper")
    fig.savefig(args.output_dir / "grid_methods.png", dpi=120)
    plt.close(fig)
    logger.info("wrote grid_methods.png to %s", args.output_dir)


if __name__ == "__main__":
    main()
