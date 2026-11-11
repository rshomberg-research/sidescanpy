"""Make the sample figures on the Public datasets page (docs/public_datasets.md).

Every dataset goes through the same chain and the same figure, so the
panels can be compared: only the file, the format and the navigation CRS
an archive leaves unstated change. The chain:

1. read, navigation onto the pings, 20 s navigation smoothing
2. ``bottom_track`` (altitude from the first bottom return), ``tvg``
   (library defaults), ``geometric`` (flat seabed)
3. ``range_correction`` fitted on the same line, which flattens whatever
   level-against-range the TVG leaves (recorder TVG, beam pattern)
4. ``georef``

The figure: left, a waterfall in ground range, one swath width long,
from the stretch of the line with the steadiest positions; right, three
swath widths around it gridded on UTM (about 300 cells across the swath,
inverse-distance weighting within 4 cells), north up. Both in dB, copper,
2 to 98 % stretch.

Files registered in :mod:`sidescanpy.datasets` are fetched (cached after
the first time). BenthiCat-Raw is not registered (its XTF comes in a
110 GB split archive), so pass a local file with ``--benthicat``.

Usage::

    PYTHONPATH="" uv run --all-extras python -I docs/_scripts/public_datasets_figures.py \\
        [--benthicat PATH/N01_05_220731124500_xtf-CH12.xtf] [--only NAME ...] [--out DIR]

Writes ``<out>/<name>.png`` (default ``docs/assets/datasets``), reduced to
a 256-colour palette to keep each file small.
"""

# 2026 rshomberg@uri.edu

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sidescanpy import datasets  # noqa: E402
from sidescanpy.core.units import FLOOR_DB  # noqa: E402
from sidescanpy.grids import grid_channel  # noqa: E402
from sidescanpy.io import read_and_convert  # noqa: E402
from sidescanpy.log import configure_logging  # noqa: E402
from sidescanpy.processing import (  # noqa: E402
    correct,
    interpolate_aux_onto_pings,
    smooth_navigation,
)
from sidescanpy.processing.range_correction import (  # noqa: E402
    RangeCorrectionTable,
    fit_range_correction,
)
from sidescanpy.viz import plot_gridded_raster, plot_waterfall  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]

# name: (files, format, read kwargs, label). The highest frequency in the file is shown.
LINES = {
    "bgs_london_gateway": (["bgs_london_gateway_dynamo_78.xtf"], "xtf", {"nav_crs": "EPSG:32631"},
                           "BGS London Gateway, Klein, XTF (GeoDAS)"),
    "usgs_grand_bay": (["usgs_grand_bay_150602205800.xtf"], "xtf", {},
                       "USGS Grand Bay, Klein, XTF (SonarPro)"),
    "jncc_wight_barfleur": (["jncc_wight_barfleur_20130321174522.jsf"], "jsf", {},
                            "JNCC Wight-Barfleur, EdgeTech, JSF (Discover)"),
    "jncc_mid_irish_sea": (["jncc_mid_irish_sea_MIS11_0.xtf"], "xtf", {},
                           "JNCC mid-Irish Sea, Benthos SIS-1624, XTF (Isis)"),
    "jncc_solan_bank": (["jncc_solan_bank_SB04_10.xtf"], "xtf", {},
                        "JNCC Solan Bank, Benthos SIS-1500, XTF (Isis)"),
    "jncc_east_rockall": (["jncc_east_rockall_ER_C1_SS_001_6.xtf"], "xtf", {"nav_crs": "EPSG:32629"},
                          "JNCC East Rockall, EdgeTech 4200, XTF (Isis)"),
    "awi_ofobs_ps124": (["awi_ofobs_ps124_20210217004957.jsf"], "jsf", {},
                        "AWI OFOBS PS124, EdgeTech 2205, JSF (Discover)"),
    "benthicat": (None, "xtf", {}, "BenthiCat-Raw, Klein 3000H, XTF (SonarWiz)"),
}

STRETCH = dict(vmin_pct=2, vmax_pct=98)


def process(paths, fmt, read_kwargs):
    """The one chain: returns a georeferenced dataset of the highest frequency."""
    survey = read_and_convert([str(p) for p in paths], fmt=fmt, **read_kwargs)
    freq = max(survey.frequencies, key=lambda k: survey.get_frequency(k).dataset.attrs.get("freq_hz_start", 0))
    survey = smooth_navigation(interpolate_aux_onto_pings(survey), window_s=20.0)
    ds = survey.get_frequency(freq).dataset.load()
    ds = correct(ds, ["bottom_track", "tvg", "geometric"],
                 {"bottom_track": {"replace": "all", "smooth_pings": 5}})
    ranges, corrections, meta = fit_range_correction(ds, bin_width_m=1.0)
    table = RangeCorrectionTable()
    table.add_curve(ds.attrs["freq_hz_start"], ranges, corrections, meta)
    ds = correct(ds, ["range_correction", "georef"], {"range_correction": {"table": table}})
    # Exact-zero samples (dropped pings, blanked water column) sit at the
    # TVG floor. Leave them out of the grid and the colour stretch, or a
    # file with many of them averages them into the map and washes it out.
    bs = ds["backscatter_intensity"]
    ds["backscatter_intensity"] = bs.where(bs > FLOOR_DB)
    return ds, freq


def along_track(ds):
    """Cumulative distance between consecutive ping positions (m), and the steps."""
    lat = np.radians(ds["latitude"].values)
    lon = np.radians(ds["longitude"].values)
    step = 6371000.0 * np.hypot(np.diff(lat), np.diff(lon) * np.cos(lat[1:]))
    return np.concatenate([[0.0], np.nancumsum(step)]), step


def pick_stretch(ds, swath_m):
    """Ping slices for the waterfall (one swath width long) and the map (three).

    The waterfall stretch is the one with the smallest largest step
    between consecutive positions, of nine candidates spread along the
    line, nearest the middle on a tie; the map stretch is centred on it.
    """
    along, step = along_track(ds)
    total = along[-1]
    best = None
    for centre in np.linspace(0.1, 0.9, 9) * total:
        sel = np.flatnonzero(np.abs(along - centre) <= swath_m / 2)
        if sel.size < 2:
            continue
        worst = float(np.nanmax(step[sel[0]:sel[-1]])) if sel[-1] > sel[0] else 0.0
        score = (round(worst, 0), abs(centre - total / 2))
        if best is None or score < best[0]:
            best = (score, sel, centre)
    _, sel, centre = best
    wide = np.flatnonzero(np.abs(along - centre) <= 1.5 * swath_m)
    return slice(int(sel[0]), int(sel[-1]) + 1), slice(int(wide[0]), int(wide[-1]) + 1)


def cell_size(swath_m):
    """About 300 cells across the swath, rounded to 0.25, 0.5, 1 or 2 m."""
    return min((0.25, 0.5, 1.0, 2.0), key=lambda c: abs(np.log(swath_m / c / 300)))


def figure(name, ds, freq, label, out):
    swath = 2 * float(np.nanmax(ds["ground_range_m"].values))
    narrow, wide = pick_stretch(ds, swath)
    part = ds.isel(ping=narrow)
    # Average down to about 400 by 400 pixels, as the panel is drawn.
    factors = {"ping": max(1, part.sizes["ping"] // 400), "sample": max(1, part.sizes["sample"] // 200)}
    cell = cell_size(swath)
    fig, (ax_w, ax_g) = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        part = part.coarsen(factors, boundary="trim").mean()
        raster = grid_channel(ds.isel(ping=wide), resolution_m=cell, method="idw",
                              max_search_radius_m=4 * cell)
    plot_waterfall(part, x_axis="across_track", y_axis="along_track", ax=ax_w, **STRETCH,
                   title="Waterfall, ground range")
    plot_gridded_raster(raster, fig=fig, ax=ax_g, cmap="copper", **STRETCH, show_colorbar=False,
                        title=f"Around it, {cell:g} m UTM grid, north up")
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.set_xlabel("")
    ax_g.set_ylabel("")
    khz = ds.attrs.get("freq_hz_start", 0) / 1000
    freq_text = f"{khz:.0f} kHz" if khz else "frequency not recorded"
    fig.suptitle(f"{label}, {freq_text}", fontsize=11)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=80)
    plt.close(fig)
    Image.open(png).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(png, optimize=True)
    print(f"{png}  {png.stat().st_size / 1024:.0f} KB  {freq}  {ds.sizes['ping']} pings, "
          f"waterfall pings {narrow.start}-{narrow.stop - 1}, swath {swath:.0f} m, cell {cell:g} m")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benthicat", type=Path, help="one BenthiCat-Raw XTF file")
    parser.add_argument("--only", nargs="*", choices=sorted(LINES), help="make only these figures")
    parser.add_argument("--out", type=Path, default=DOCS / "assets" / "datasets")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    configure_logging("ERROR")
    for name in args.only or LINES:
        files, fmt, read_kwargs, label = LINES[name]
        if files is None:
            if args.benthicat is None:
                print(f"{name}: skipped (pass --benthicat)")
                continue
            paths = [args.benthicat]
        else:
            paths = [datasets.fetch(f) for f in files]
        ds, freq = process(paths, fmt, read_kwargs)
        figure(name, ds, freq, label, args.out)


if __name__ == "__main__":
    sys.exit(main())
