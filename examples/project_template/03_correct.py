"""
Step 3 -- correct every .ss.h5 in place: TVG, geometric ranges,
georeferencing and any other configured steps, with the TVG recipe
(per frequency where it differs) from the config.

Already-applied steps are skipped (recorded in processing_history), so a
re-run only does what is missing.

Writes:
  <output_root>/ss_h5/*.ss.h5               corrected in place (atomic rewrite)
  <output_root>/correct/<freq>_vs_range.png corrected mean vs. slant range (should be flat)
  <output_root>/correct/<freq>_overview.png corrected whole-survey overview
  <output_root>/reports/03_correct.{json,md}

Usage:
    python 03_correct.py config.yaml

2026 rshomberg@uri.edu
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steplib import logger, save_figure, ss_h5_files, start_step

from sidescanpy.core.history import applied_steps
from sidescanpy.io.hdf5 import concat_survey_frequency, load_channel_dataset
from sidescanpy.processing import backscatter_vs_range, correct_file, correction_steps
from sidescanpy.viz import plot_mean_vs_range, plot_survey_overview


def resolve_params(params: dict | None, paths) -> dict | None:
    """Config params with relative file paths (a range-correction table) made absolute."""
    if not params:
        return params
    out = {step: dict(p or {}) for step, p in params.items()}
    table = out.get("range_correction", {}).get("table")
    if isinstance(table, str):
        out["range_correction"]["table"] = str(paths.resolve(table))
    return out


def correct_files(cfg, paths, report) -> None:
    opts = cfg.get("correct", {})
    steps = correction_steps(opts.get("steps", ["tvg", "geometric", "georef"]))
    params = resolve_params(opts.get("params"), paths)
    frequency_params = {fk: resolve_params(p, paths) for fk, p in (opts.get("frequency_params") or {}).items()}
    files = ss_h5_files(paths.ss_h5)
    if not files:
        raise SystemExit(f"no .ss.h5 files in {paths.ss_h5} -- run 01_convert.py first")
    report.results["steps"] = steps
    per_file = {}
    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        mtime_before = path.stat().st_mtime_ns
        logger.info("[%d/%d] %s: %s", i, len(files), path.name, " -> ".join(steps))
        correct_file(
            path, steps, params, frequency_params=frequency_params,
            chunk_pings=opts.get("chunk_pings"), progress=lambda msg: logger.debug("  %s", msg),
        )
        per_file[path.name] = {
            "rewritten": path.stat().st_mtime_ns != mtime_before,  # False: every step was already applied
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "seconds": round(time.perf_counter() - t0, 1),
        }
    report.results["files"] = per_file


def qa_figures(cfg, paths, report) -> None:
    files = ss_h5_files(paths.ss_h5)
    wf = cfg.get("export", {}).get("waterfall", {})
    start, stop = cfg.get("analyze", {}).get("pings", [0, 5000])
    for freq_key in cfg["frequencies"]:
        # Applied steps, as recorded in the first file.
        ds = load_channel_dataset(files[0], freq_key, {"ping": 512}).isel(ping=slice(start, stop)).load()
        report.results.setdefault("applied", {})[freq_key] = applied_steps(ds)
        binned = backscatter_vs_range(ds, "backscatter_intensity", range_var="slant_range_m", bin_width_m=2.0,
                                      exclude_below_db=-110.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_mean_vs_range(binned, ax=ax, title=f"{files[0].name} {freq_key}: corrected, pings {start}:{stop}")
        report.figure(save_figure(fig, paths.step_dir("correct") / f"{freq_key}_vs_range.png", dpi=120), paths.output_root)

        full, file_start = concat_survey_frequency(files, freq_key, chunks={"ping": 512})
        stride = max(1, full.sizes["ping"] // wf.get("overview_rows", 60))
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_survey_overview(
            full, ax=ax, ping_stride=stride, break_ping_indices=sorted(p for p in file_start.values() if p > 0),
            cmap=wf.get("cmap", "copper"), vmin_pct=wf.get("vmin_pct", 2.0), vmax_pct=wf.get("vmax_pct", 98.0),
            gamma=wf.get("gamma", 2.5), title=f"{cfg.get('survey', '')} {freq_key}: corrected",
        )
        report.figure(save_figure(fig, paths.step_dir("correct") / f"{freq_key}_overview.png", dpi=100), paths.output_root)


def main(argv=None) -> None:
    cfg, paths, report = start_step("03_correct", __doc__.split("\n\n")[0], argv)
    correct_files(cfg, paths, report)
    qa_figures(cfg, paths, report)
    report.write(paths.reports, next_step="04_export.py")


if __name__ == "__main__":
    main()
