"""
Step 1 -- convert raw .jsf/.sdf files to .ss.h5, and the target catalog to
targets.json.

Writes:
  <output_root>/ss_h5/<stem>.ss.h5      one per raw file (far-range noise cropped)
  <output_root>/targets/targets.json    + targets.csv (when a catalog is configured)
  <output_root>/convert/                QA: target positions, uncorrected overview per frequency
  <output_root>/reports/01_convert.{md,json}

Usage:
    python 01_convert.py config.yaml

2026 rshomberg@uri.edu
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steplib import logger, save_figure, ss_h5_files, start_step

from sidescanpy.io import (
    CATALOG_SDF_KLEIN,
    find_sonar_files,
    read_and_convert_segments,
    read_target_catalog,
)
from sidescanpy.io.hdf5 import concat_survey_frequency, write_survey
from sidescanpy.processing import crop_far_range_noise
from sidescanpy.targets import write_targets_csv, write_targets_json
from sidescanpy.targets.correlate import (
    calibrate_clock_offset,
    resolve_targets_by_catalog_ping,
    resolve_targets_nearest_ping,
    resolve_targets_offset_corrected,
)
from sidescanpy.viz import plot_survey_overview, plot_target_position_scatter


def convert_files(cfg, paths, report) -> None:
    if paths.raw_dir is None:
        logger.info("no raw_dir configured; using the existing files in %s", paths.ss_h5)
        return
    opts = cfg.get("convert", {})
    files = find_sonar_files(paths.raw_dir, "auto")
    if not files:
        raise SystemExit(f"no .jsf/.sdf files in {paths.raw_dir}")
    paths.ss_h5.mkdir(parents=True, exist_ok=True)
    per_file = {}
    for i, raw in enumerate(files, 1):
        existing = sorted(paths.ss_h5.glob(f"{raw.stem}.ss.h5")) + sorted(paths.ss_h5.glob(f"{raw.stem}_seg*.ss.h5"))
        if existing and not opts.get("overwrite", False):
            logger.info("[%d/%d] %s: already converted, skipping", i, len(files), raw.name)
            per_file[raw.name] = {"status": "skipped"}
            continue
        for old in existing:
            old.unlink()
        t0 = time.perf_counter()
        logger.info("[%d/%d] %s: converting", i, len(files), raw.name)
        # One survey per run of constant range setting (a file whose range,
        # and so sample interval, changes mid-file is split into <stem>_segNN).
        segments = read_and_convert_segments(raw, survey_name=raw.stem,
                                              min_pings=opts.get("min_segment_pings", 50))
        if len(segments) > 1:
            report.note(f"{raw.name}: range setting changes mid-file; split into {len(segments)} segments")
        for stem, survey in segments:
            out = paths.ss_h5 / f"{stem}.ss.h5"
            cutoffs = {}
            if opts.get("crop_farrange", True):
                survey, cutoffs = crop_far_range_noise(survey)
            write_survey(survey, out, overwrite=True)
            per_file[out.name] = {
                "status": "converted",
                "source": raw.name,
                "frequencies": {fk: dict(survey.get_frequency(fk).dataset.sizes) for fk in survey.frequencies},
                "sample_rate_hz": {fk: survey.get_frequency(fk).dataset.attrs["sample_rate_hz"]
                                   for fk in survey.frequencies},
                "far_range_cutoff_sample": cutoffs,
                "size_mb": round(out.stat().st_size / 1e6, 1),
                "seconds": round(time.perf_counter() - t0, 1),
            }
    report.results["files"] = per_file


def convert_targets(cfg, paths, report) -> None:
    if paths.targets_source_dir is None:
        logger.info("no targets_source_dir configured; no catalog to convert")
        return
    fmt, targets = read_target_catalog(paths.targets_source_dir)
    opts = cfg.get("targets", {})
    if fmt == CATALOG_SDF_KLEIN and opts.get("resolve", "catalog_ping") == "catalog_ping":
        # The catalog names the target's own ping (time + number on the
        # sonar clock), so no clock offset is needed.
        targets = resolve_targets_by_catalog_ping(targets, paths.ss_h5)
        sources = {}
        for t in targets:
            src = t.extra_attributes.get("position_source", "unplaced")
            sources[src] = sources.get(src, 0) + 1
        report.results["target_position_source"] = sources
        if sources.get("unplaced"):
            report.note(f"{sources['unplaced']} target(s) have a catalog time in no converted file "
                        "(picked while not logging, or that file was not converted)")
    elif fmt == CATALOG_SDF_KLEIN:
        # Legacy: match the pick (save) time, 30-90 s after the target ping.
        offset_s = opts.get("time_offset_s")
        if opts.get("calibrate_offset"):
            if opts.get("local_utc_offset_hours") is None:
                raise SystemExit("targets.calibrate_offset needs targets.local_utc_offset_hours")
            offset_s = calibrate_clock_offset(paths.ss_h5, local_utc_offset_hours=opts["local_utc_offset_hours"])
            report.results["clock_offset_s"] = offset_s
        max_dt = opts.get("max_time_delta_s", 300.0)
        if offset_s is not None:
            targets = resolve_targets_offset_corrected(targets, paths.ss_h5, offset_s, max_time_delta_s=max_dt)
        else:
            targets = resolve_targets_nearest_ping(targets, paths.ss_h5, max_time_delta_s=max_dt)
            report.note("Klein target positions use the nearest ping in time with no clock-offset correction")
    out_dir = paths.targets_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_targets_json(targets, paths.targets_json)
    write_targets_csv(targets, out_dir / "targets.csv")
    n_positioned = sum(t.latitude is not None for t in targets)
    confidence = {c: sum(t.position_confidence == c for t in targets) for c in ("verified", "approximate", "unknown")}
    report.results["targets"] = {"format": fmt, "n": len(targets), "with_position": n_positioned,
                                 "position_confidence": confidence}
    if n_positioned < len(targets):
        report.note(f"{len(targets) - n_positioned} target(s) have no position")
    fig = plot_target_position_scatter(targets)
    if fig is not None:
        report.figure(save_figure(fig, paths.step_dir("convert") / "target_positions.png", dpi=150), paths.output_root)


def overview_figures(cfg, paths, report) -> None:
    """Whole-survey overview of the converted (uncorrected) signal, per frequency."""
    wf = cfg.get("export", {}).get("waterfall", {})
    files = ss_h5_files(paths.ss_h5)
    for freq_key in cfg["frequencies"]:
        ds, file_start = concat_survey_frequency(files, freq_key, chunks={"ping": 512})
        if ds is None:
            report.note(f"{freq_key}: no file has this frequency")
            continue
        report.results.setdefault("pings", {})[freq_key] = int(ds.sizes["ping"])
        stride = max(1, ds.sizes["ping"] // wf.get("overview_rows", 60))
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_survey_overview(
            ds, ax=ax, ping_stride=stride, break_ping_indices=sorted(p for p in file_start.values() if p > 0),
            cmap=wf.get("cmap", "copper"), vmin_pct=wf.get("vmin_pct", 2.0), vmax_pct=wf.get("vmax_pct", 98.0),
            gamma=wf.get("gamma", 2.5), title=f"{cfg.get('survey', '')} {freq_key}: converted, uncorrected",
        )
        out = paths.step_dir("convert") / f"{freq_key}_overview_uncorrected.png"
        report.figure(save_figure(fig, out, dpi=100), paths.output_root)


def main(argv=None) -> None:
    cfg, paths, report = start_step("01_convert", __doc__.split("\n\n")[0], argv)
    convert_files(cfg, paths, report)
    convert_targets(cfg, paths, report)
    overview_figures(cfg, paths, report)
    report.write(paths.reports, next_step="02_analyze.py")


if __name__ == "__main__":
    main()
