"""Make the sample figures on the Public datasets page (docs/public_datasets.md).

Every dataset goes through the same chain and the same three-panel
figure, so they can be compared; only the file, the format, a navigation
CRS an archive leaves unstated and (optionally) a known target change.

The chain (sidescanpy library calls):

1. read; navigation onto the pings; 20 s navigation smoothing
2. ``bottom_track`` (altitude from the first bottom return)
3. ``tvg`` (library defaults: 30 log R spreading, absorption for the
   recorded frequency)
4. ``geometric`` (flat seabed: slant to ground range)
5. ``range_correction`` fitted on the same line (flattens the level left
   against range: recorder TVG, beam pattern)
6. ``georef`` (WGS84), gridded on UTM

The section: one swath width of pings. If the dataset names a target
(a catalogued wreck), the section is centred on the ping nearest it;
otherwise it is the window with the most seabed texture (band-passed
corrected dB, outer 90 % of the swath) among straight windows (heading
within 20 degrees) with no navigation jumps or runs of dropped pings.

The figure, same pings in every panel:

- Raw: the samples as the file stores them (units inferred by
  :func:`sidescanpy.analysis.characterize.infer_units`), against slant
  range, linear 2-98 % stretch.
- Corrected: after steps 2-5, dB, against across-track ground range.
- Georeferenced: the section gridded on UTM at
  :func:`sidescanpy.grids.recommend_resolution` (half the finer native
  spacing, at least 5 cm), north up.

For each dataset it also writes ``<name>_caption.md``, the compact list
of corrections and parameters taken from the processing history, which
the page includes under the figure.

Files registered in :mod:`sidescanpy.datasets` are fetched (cached after
the first time). BenthiCat-Raw is not registered (its XTF comes in a
110 GB split archive), so pass a local file with ``--benthicat``.

Usage::

    PYTHONPATH="" uv run --all-extras python -I docs/_scripts/public_datasets_figures.py \\
        [--benthicat PATH/N02_SS_04_220807105300_xtf-CH12.xtf] [--only NAME ...] [--out DIR]
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
import pyproj  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from sidescanpy import datasets  # noqa: E402
from sidescanpy.analysis import measure_ping_geometry  # noqa: E402
from sidescanpy.analysis.characterize import characterize_values, infer_units  # noqa: E402
from sidescanpy.core.history import read_history  # noqa: E402
from sidescanpy.core.units import FLOOR_DB  # noqa: E402
from sidescanpy.grids import grid_channel, recommend_resolution  # noqa: E402
from sidescanpy.io import read_and_convert  # noqa: E402
from sidescanpy.log import configure_logging  # noqa: E402
from sidescanpy.processing import (  # noqa: E402
    correct,
    course_over_ground,
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
                           "BGS London Gateway, Klein, XTF (GeoDAS)",
                           ("EPSG:32631", *datasets.BGS_LONDON_GATEWAY_WRECKS["Dynamo"], "Dynamo wreck")),
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
CODE_URL = ("https://github.com/rshomberg-research/sidescanpy/blob/main/"
            "docs/_scripts/public_datasets_figures.py")


def process(paths, fmt, read_kwargs):
    """The one chain. Returns (raw, corrected, freq) for the highest frequency.

    ``raw`` holds the backscatter exactly as read (same pings and samples
    as ``corrected``).
    """
    survey = read_and_convert([str(p) for p in paths], fmt=fmt, **read_kwargs)
    freq = max(survey.frequencies, key=lambda k: survey.get_frequency(k).dataset.attrs.get("freq_hz_start", 0))
    survey = smooth_navigation(interpolate_aux_onto_pings(survey), window_s=20.0)
    ds = survey.get_frequency(freq).dataset.load()
    # A file with no real heading (all NaN or one constant value) would
    # place every swath at that bearing: use the course over ground.
    h = ds["heading_deg"].values
    heading_from_cog = bool(np.all(~np.isfinite(h)) or np.nanstd(h) < 0.01)
    if heading_from_cog:
        ds["heading_deg"] = ("ping", course_over_ground(ds).astype(np.float32))
    ds.attrs["_heading_from_cog"] = int(heading_from_cog)
    raw = ds["backscatter_intensity"].copy()
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
    assert raw.shape == ds["backscatter_intensity"].shape
    return raw, ds, freq


def along_track(ds):
    """Cumulative distance between consecutive ping positions (m), and the steps."""
    lat = np.radians(ds["latitude"].values)
    lon = np.radians(ds["longitude"].values)
    step = 6371000.0 * np.hypot(np.diff(lat), np.diff(lon) * np.cos(lat[1:]))
    return np.concatenate([[0.0], np.nancumsum(step)]), step


def texture_score(part):
    """Seabed texture of a section: std of band-passed corrected dB, outer swath."""
    db = part["backscatter_intensity"].values
    gr = part["ground_range_m"].values
    outer = gr > 0.1 * np.nanmax(gr)
    v = np.where(outer & np.isfinite(db), db, np.nan)
    scores = []
    for c in range(v.shape[1]):
        a = v[:, c, :]
        m = np.isfinite(a)
        if m.mean() < 0.3:
            continue
        f = np.where(m, a, np.nanmedian(a))
        band = gaussian_filter(f, 1.5) - gaussian_filter(f, 12)
        scores.append(np.nanpercentile(np.abs(band[m]), 95))
    return float(np.mean(scores)) if scores else -np.inf


def pick_section(ds, swath_m, target=None):
    """Pings of one swath width: centred on ``target`` or on the most textured steady window."""
    along, step = along_track(ds)
    total = along[-1]
    how = ""
    if target is not None:
        crs, te, tn, tname = target
        e, n = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            ds["longitude"].values, ds["latitude"].values)
        k = int(np.nanargmin(np.hypot(np.asarray(e) - te, np.asarray(n) - tn)))
        centre, how = along[k], f"centred on the {tname}"
    else:
        med = float(np.nanmedian(step[step > 0])) if np.any(step > 0) else 0.0
        seafloor_frac = float(np.isfinite(ds["backscatter_intensity"].values).mean())
        best = None
        for centre in np.arange(swath_m / 2, max(total - swath_m / 2, swath_m / 2) + 1e-6, swath_m / 4):
            sel = np.flatnonzero(np.abs(along - centre) <= swath_m / 2)
            if sel.size < 20:
                continue
            worst = float(np.nanmax(step[sel[0]:sel[-1]])) if sel[-1] > sel[0] else 0.0
            if worst > max(5 * med, 2.0):
                continue                                   # navigation jump inside
            win = ds.isel(ping=slice(int(sel[0]), int(sel[-1]) + 1))
            hd = np.unwrap(np.deg2rad(win["heading_deg"].values.astype(float)))
            if np.ptp(np.rad2deg(hd[np.isfinite(hd)])) > 20:
                continue                                   # a turn: pings fan out
            if np.isfinite(win["backscatter_intensity"].values).mean() < 0.85 * seafloor_frac:
                continue                                   # many dropped pings
            score = texture_score(win)
            if best is None or score > best[0]:
                best = (score, centre)
        if best is None:                                   # every window has a jump
            centre = total / 2
            how = "middle of the line (no straight, jump-free window)"
        else:
            centre, how = best[1], "most textured straight, jump-free window"
    sel = np.flatnonzero(np.abs(along - centre) <= swath_m / 2)
    return slice(int(sel[0]), int(sel[-1]) + 1), how


def cell_size(part):
    """Finest useful cell: half the finer native spacing, 5 cm to 1 m."""
    r = recommend_resolution(part)
    return float(np.clip(r if r else 0.5, 0.05, 1.0))


def caption(ds, section, raw_units, cell, utm, how, name):
    """Compact corrections line from the processing history."""
    hist = {r["step"]: r["params"] for r in read_history(ds)}
    tvg = hist.get("tvg", {})
    alpha = tvg.get("absorption_db_per_m")
    steps = []
    if "smooth_navigation" in hist:
        steps.append(f"navigation smoothed ({hist['smooth_navigation'].get('window_s', 20):g} s)")
    if ds.attrs.get("_heading_from_cog"):
        steps.append("heading from course over ground (file records none)")
    if "bottom_track" in hist:
        steps.append("altitude from bottom track")
    if tvg:
        a = f"{alpha * 1000:.0f} dB/km absorption" if alpha else "no absorption (frequency not recorded)"
        steps.append(f"TVG {tvg.get('spreading_loss_db', 30):g} log R + {a}")
    if "geometric" in hist:
        steps.append("flat-seabed slant to ground range")
    if "range_correction" in hist:
        steps.append("range correction fitted on this line")
    pings = f"pings {section.start}-{section.stop - 1} of {ds.sizes['ping']}"
    return (f"**Raw:** {raw_units}, as stored, slant range. "
            f"**Corrected:** {'; '.join(steps)}. "
            f"**Georeferenced:** {utm}, {cell * 100:.0f} cm cells. "
            f"**Section:** {pings}, {how}. "
            f"**Code:** [`public_datasets_figures.py`]({CODE_URL}) `--only {name}`.\n")


def figure(name, raw, ds, freq, label, out, target=None):
    swath = 2 * float(np.nanmax(ds["ground_range_m"].values))
    narrow, how = pick_section(ds, swath, target)
    part = ds.isel(ping=narrow)
    units = infer_units(characterize_values(raw.isel(ping=narrow).to_dataset(name="backscatter_intensity"))).units
    raw_units = {"raw_counts": "raw counts", "dB": "dB", "normalized": "normalized amplitude"}.get(units, "amplitude, units unknown")
    rawview = part.copy()
    rawview["backscatter_intensity"] = raw.isel(ping=narrow)
    cell = cell_size(part)
    # The search radius has to bridge the widest ordinary gap between pings
    # (they fan out on turns and with range), so use the 95th-percentile
    # ping step, not the median.
    _, steps = along_track(part)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    spacing = float(np.percentile(steps, 95)) if steps.size else (
        measure_ping_geometry(part).ping_spacing_m_median or cell)
    factors = {"ping": max(1, part.sizes["ping"] // 500), "sample": max(1, part.sizes["sample"] // 500)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raster = grid_channel(part, resolution_m=cell, method="idw",
                              max_search_radius_m=max(spacing, 2 * cell))
    # Map footprint aspect (north-up). A wide footprint gets its own row.
    rows, cols = np.nonzero(np.isfinite(np.asarray(raster.backscatter)))
    aspect = (np.ptp(cols) + 1) / max(np.ptp(rows) + 1, 1) if rows.size else 1.0
    if aspect > 1.6:
        fig = plt.figure(figsize=(12, 5.2 + 12 / 2 / aspect + 0.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[5.2, 12 / aspect * 0.55])
        ax_r, ax_c, ax_g = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    else:
        fig, (ax_r, ax_c, ax_g) = plt.subplots(
            1, 3, figsize=(15, 5.2), constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 1, float(np.clip(aspect, 0.5, 1.6))]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plot_waterfall(rawview.coarsen(factors, boundary="trim").mean(), x_axis="slant_range",
                       y_axis="along_track", ax=ax_r, cmap="gray", **STRETCH,
                       title=f"As stored: {raw_units}, slant range")
        plot_waterfall(part.coarsen(factors, boundary="trim").mean(), x_axis="across_track",
                       y_axis="along_track", ax=ax_c, **STRETCH, title="Corrected (dB), ground range")
    plot_gridded_raster(raster, fig=fig, ax=ax_g, cmap="copper", **STRETCH, show_colorbar=False,
                        title=f"Georeferenced, {cell * 100:.0f} cm UTM grid, north up")
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.set_xlabel("")
    ax_g.set_ylabel("")
    khz = ds.attrs.get("freq_hz_start", 0) / 1000
    freq_text = f"{khz:.0f} kHz" if khz else "frequency not recorded"
    fig.suptitle(f"{label}, {freq_text}", fontsize=12)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=90)
    plt.close(fig)
    Image.open(png).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(png, optimize=True)
    utm = raster.grid_def.crs.name if hasattr(raster.grid_def.crs, "name") else str(raster.grid_def.crs)
    utm = utm.replace("WGS 84 / ", "")
    (out / f"{name}_caption.md").write_text(caption(ds, narrow, raw_units, cell, utm, how, name))
    print(f"{png}  {png.stat().st_size / 1024:.0f} KB  {freq}  {ds.sizes['ping']} pings, "
          f"section {narrow.start}-{narrow.stop - 1} ({how}), swath {swath:.0f} m, cell {cell:g} m, raw {raw_units}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benthicat", type=Path, help="one BenthiCat-Raw XTF file")
    parser.add_argument("--only", nargs="*", choices=sorted(LINES), help="make only these figures")
    parser.add_argument("--out", type=Path, default=DOCS / "assets" / "datasets")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    configure_logging("ERROR")
    for name in args.only or LINES:
        files, fmt, read_kwargs, label, *rest = LINES[name]
        target = rest[0] if rest else None
        if files is None:
            if args.benthicat is None:
                print(f"{name}: skipped (pass --benthicat)")
                continue
            paths = [args.benthicat]
        else:
            paths = [datasets.fetch(f) for f in files]
        raw, ds, freq = process(paths, fmt, read_kwargs)
        figure(name, raw, ds, freq, label, args.out, target)


if __name__ == "__main__":
    sys.exit(main())
