"""
Step 4 -- exports from the corrected survey: track map (with targets, if any),
whole-survey waterfalls (slant range and ground range) with matched targets
boxed, and gridded rasters (PNG + GeoTIFF).

Writes:
  <output_root>/exports/track/<freq>/track.png
  <output_root>/exports/waterfall/<freq>/{overview.png, segment_NNN.png}
  <output_root>/exports/waterfall_ground/<freq>/segment_NNN.png
  <output_root>/exports/gridded/<freq>/<file>/segment_NNN.{png,tif}
  <output_root>/reports/04_export.{json,md}

Usage:
    python 04_export.py config.yaml

2026 rshomberg@uri.edu
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from steplib import logger, save_figure, ss_h5_files, start_step

from sidescanpy.grids import grid_channel, to_geotiff
from sidescanpy.io.hdf5 import concat_survey_frequency, read_survey
from sidescanpy.processing import build_survey_track, ping_windows
from sidescanpy.targets import read_targets_json
from sidescanpy.targets.correlate import build_position_index, correlate_catalog
from sidescanpy.targets.waterfall import targets_by_global_ping, waterfall_boxes
from sidescanpy.viz import (
    plot_gridded_raster,
    plot_survey_overview,
    plot_track_and_scanlines,
    plot_track_with_targets,
    plot_waterfall,
    resolve_waterfall_axes,
)


def export_track(cfg, paths, files, targets, report) -> None:
    opts = cfg["export"].get("track", {})
    maps = cfg["export"].get("basemap", {})
    surveys = {p: read_survey(p, chunks={"ping": 2048}) for p in files}
    n = 0
    for freq_key in cfg["frequencies"]:
        track = build_survey_track(surveys, freq_key, scanline_stride=opts.get("scanline_stride", 50))
        if track is None:
            report.note(f"track: no file has {freq_key}")
            continue
        show = dict(show_land=maps.get("show_land", False), show_locator=maps.get("show_locator", False))
        if targets:
            fig, _ax = plot_track_with_targets(track, targets, **show)
        else:
            fig, _ax = plot_track_and_scanlines(track.track_lat, track.track_lon, track.scan_port_lat,
                                                track.scan_port_lon, track.scan_stbd_lat, track.scan_stbd_lon, **show)
        out = save_figure(fig, paths.step_dir("exports") / "track" / freq_key / "track.png", dpi=opts.get("dpi", 150), tight=False)
        report.figure(out, paths.output_root)
        n += 1
    report.results["track_pngs"] = n


def save_waterfall(ds, out, *, x_axis, y_axis, ping_offset, boxes, wf) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_waterfall(ds, x_axis=x_axis, y_axis=y_axis, ping_offset=ping_offset, cmap=wf.get("cmap", "copper"),
                   vmin_pct=wf.get("vmin_pct", 2.0), vmax_pct=wf.get("vmax_pct", 98.0), gamma=wf.get("gamma", 2.5), ax=ax)
    for label, x0, y0, width, height in boxes:
        ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="red", linewidth=1.2))
        ax.annotate(label, (x0, y0), color="red", fontsize=11, fontweight="bold", xytext=(2, 2), textcoords="offset points")
    fig.tight_layout()
    save_figure(fig, out, dpi=wf.get("dpi", 100), tight=False)


def export_waterfalls(cfg, paths, files, matched, report, *, range_mode: str) -> None:
    exp = cfg["export"]
    wf = exp.get("waterfall", {})
    if range_mode == "ground":
        x_axis, y_axis = resolve_waterfall_axes(frozenset({"geometric"}), "sample", "ping")
        root = paths.step_dir("exports") / "waterfall_ground"
    else:
        x_axis, y_axis = "sample", "ping"
        root = paths.step_dir("exports") / "waterfall"
    n = 0
    for freq_key in cfg["frequencies"]:
        ds_all, file_start = concat_survey_frequency(files, freq_key, chunks={"ping": 512})
        if ds_all is None:
            continue
        if range_mode == "slant" and wf.get("overview_rows"):
            stride = max(1, ds_all.sizes["ping"] // wf["overview_rows"])
            fig, ax = plt.subplots(figsize=(14, 8))
            plot_survey_overview(ds_all, ax=ax, ping_stride=stride,
                                 break_ping_indices=sorted(p for p in file_start.values() if p > 0),
                                 cmap=wf.get("cmap", "copper"), vmin_pct=wf.get("vmin_pct", 2.0),
                                 vmax_pct=wf.get("vmax_pct", 98.0), gamma=wf.get("gamma", 2.5))
            fig.tight_layout()
            report.figure(save_figure(fig, root / freq_key / "overview.png", dpi=wf.get("dpi", 100), tight=False),
                          paths.output_root)
        by_ping = targets_by_global_ping(matched, freq_key, file_start)
        windows = ping_windows(ds_all.sizes["ping"], wf.get("window_pings", 3000), wf.get("overlap_pings", 500))
        for i, (start, stop) in enumerate(windows, 1):
            window = ds_all.isel(ping=slice(start, stop)).load()
            boxes = [(t, p) for p in range(start, stop) for t in by_ping.get(p, [])]
            rects = waterfall_boxes(window, boxes, start, x_axis, y_axis, exp.get("box_size_px", 50))
            save_waterfall(window, root / freq_key / f"segment_{i:03d}.png", x_axis=x_axis, y_axis=y_axis,
                           ping_offset=start, boxes=rects, wf=wf)
            n += 1
        logger.info("waterfall %s %s: %d window(s)", range_mode, freq_key, len(windows))
    report.results[f"waterfall_{range_mode}_pngs"] = n


def export_grids(cfg, paths, files, report) -> None:
    opts = cfg["export"].get("grid", {})
    maps = cfg["export"].get("basemap", {})
    n_png = n_tif = n_unpositioned = 0
    geotiff = opts.get("geotiff", False)
    if geotiff:
        try:
            import rioxarray  # noqa: F401
        except ImportError:
            report.note("grid.geotiff is set but rioxarray is not installed (pip install 'sidescanpy[geo]'); PNGs only")
            geotiff = False
    for path in files:
        survey = read_survey(path, chunks={"ping": opts.get("segment_pings", 2048)})
        for freq_key in cfg["frequencies"]:
            if freq_key not in survey.frequencies:
                continue
            ds = survey.get_frequency(freq_key).dataset
            for i, (start, stop) in enumerate(ping_windows(ds.sizes["ping"], opts.get("segment_pings", 2048)), 1):
                out = paths.step_dir("exports") / "gridded" / freq_key / path.stem / f"segment_{i:03d}.png"
                segment = ds.isel(ping=slice(start, stop)).load()
                if not np.isfinite(segment["latitude_per_sample"].values).any():
                    # e.g. a short-range frequency whose whole swath lies inside the water column
                    n_unpositioned += 1
                    continue
                raster = grid_channel(segment, opts.get("resolution_m", 1.0), opts.get("method", "mean"))
                fig, _ax = plot_gridded_raster(raster, cmap=opts.get("cmap", "copper"), vmin_pct=opts.get("vmin_pct", 2.0),
                                               vmax_pct=opts.get("vmax_pct", 98.0), gamma=opts.get("gamma", 2.5),
                                               show_land=maps.get("show_land", False),
                                               show_locator=maps.get("show_locator", False))
                save_figure(fig, out, dpi=opts.get("dpi", 100))
                n_png += 1
                if geotiff:
                    to_geotiff(raster, out.with_suffix(".tif"))
                    n_tif += 1
        logger.info("gridded %s", path.name)
    report.results["grid_pngs"] = n_png
    report.results["grid_geotiffs"] = n_tif
    if n_unpositioned:
        report.note(f"{n_unpositioned} grid segment(s) skipped: no georeferenced samples "
                    "(swath inside the water column, or georef not applied)")


def main(argv=None) -> None:
    cfg, paths, report = start_step("04_export", __doc__.split("\n\n")[0], argv)
    files = ss_h5_files(paths.ss_h5)
    if not files:
        raise SystemExit(f"no .ss.h5 files in {paths.ss_h5} -- run 01-03 first")
    targets, matched = [], []
    if paths.targets_json.exists():
        targets = read_targets_json(paths.targets_json)
        matched = correlate_catalog(targets, build_position_index(paths.ss_h5), cfg["export"].get("max_match_dist_m", 25.0))
        report.results["targets_matched"] = f"{len(matched)} placement(s) for {len(targets)} target(s)"
        if not matched:
            report.note("no target matched a ping within max_match_dist_m")

    export_track(cfg, paths, files, targets, report)
    export_waterfalls(cfg, paths, files, matched, report, range_mode="slant")
    export_waterfalls(cfg, paths, files, matched, report, range_mode="ground")
    export_grids(cfg, paths, files, report)
    report.write(paths.reports, next_step="05_dataset.py")


if __name__ == "__main__":
    main()
