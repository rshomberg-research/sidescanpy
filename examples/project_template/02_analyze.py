"""
Step 2 -- measure the converted (uncorrected) data before correcting it:
the TVG model fit per frequency, on one file, and how flat the data are
after the best model and after the config's recipe.

The fit (fit_tvg_models) compares no TVG, absorption only and 20/30/40
log R with and without absorption over a window from past the profile
peak to the noise knee; check_tvg applies a recipe and measures the
corrected level. The report flags a config recipe that leaves the data
more than 0.5 dB less flat than the best model, so the recipe is checked
before step 3 applies it.

Writes:
  <output_root>/analyze/<freq>_tvg_fit.png      fit + level after TVG
  <output_root>/analyze/<freq>_metrics.json     fit and both checks
  <output_root>/reports/02_analyze.{json,md}

Usage:
    python 02_analyze.py config.yaml

2026 rshomberg@uri.edu
"""

import json

import matplotlib

matplotlib.use("Agg")
from steplib import logger, save_figure, ss_h5_files, start_step

from sidescanpy.analysis import check_tvg, fit_tvg_models
from sidescanpy.core.history import has_step
from sidescanpy.io.hdf5 import load_channel_dataset
from sidescanpy.processing import compute_geometric_ranges, invert
from sidescanpy.processing.radiometric import DEFAULT_SPREADING_LOSS_DB, TVGParams
from sidescanpy.viz import plot_tvg_fit

# The config's recipe is "less flat" than the best model by more than this.
_FLATNESS_TOLERANCE_DB = 0.5


def configured_params(cfg: dict, freq_key: str) -> TVGParams:
    """The TVG the config's correct: section applies to ``freq_key``
    (absorption None: Francois-Garrison, as apply_tvg does)."""
    correct = cfg.get("correct", {})
    tvg = dict((correct.get("params") or {}).get("tvg") or {})
    tvg.update(((correct.get("frequency_params") or {}).get(freq_key) or {}).get("tvg") or {})
    return TVGParams(spreading_loss_db=tvg.get("spreading_loss_db", DEFAULT_SPREADING_LOSS_DB),
                     absorption_db_per_m=tvg.get("absorption_db_per_m"))


def fit_figure(fit, check_best, check_config, title: str):
    """plot_tvg_fit's column, with the config recipe's level added to the
    after-TVG panel (orange dashed)."""
    fig, (ax, ax_check) = plot_tvg_fit(fit, check_best)
    fig.set_size_inches(6.5, 8.5)
    if check_config is not None and check_config.relative_db is not None:
        shown = check_config.range_m >= fit.window_m[0] - 3.0
        ax_check.plot(check_config.range_m[shown], check_config.relative_db[shown], color="C1", ls="--", lw=1.5,
                      label=f"config: {check_config.model_name}, spread {check_config.spread_db:.2f} dB")
        ax_check.legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=9)
    return fig


def _round(value, digits=2):
    return None if value is None else round(float(value), digits)


def main(argv=None) -> None:
    cfg, paths, report = start_step("02_analyze", __doc__.split("\n\n")[0], argv)
    opts = cfg.get("analyze", {})
    files = ss_h5_files(paths.ss_h5)
    if not files:
        raise SystemExit(f"no .ss.h5 files in {paths.ss_h5} -- run 01_convert.py first")
    path = files[0] if opts.get("file", "first") == "first" else paths.ss_h5 / opts["file"]
    start, stop = opts.get("pings", [0, 5000])
    range_var = opts.get("range_var", "slant_range_m")
    bin_width_m = opts.get("bin_width_m", 1.0)
    window_m = opts.get("window_m")
    report.results["file"] = path.name
    report.results["pings"] = [start, stop]
    out_dir = paths.step_dir("analyze")
    out_dir.mkdir(parents=True, exist_ok=True)

    for freq_key in cfg["frequencies"]:
        ds = load_channel_dataset(path, freq_key, {"ping": 512}).isel(ping=slice(start, stop)).load()
        if has_step(ds, "tvg"):
            ds = invert(ds)  # analyze the uncorrected signal even on a corrected file
        if "slant_range_m" not in ds:
            ds = compute_geometric_ranges(ds)
        # analyze.window_m overrides the automatic window (5 m past the peak to
        # the noise knee) when the figure shows it on the wrong stretch.
        if window_m is not None:
            report.note(f"{freq_key}: fit window set in the config: {window_m} m")
        fit = fit_tvg_models(
            ds, range_var=range_var, bin_width_m=bin_width_m, window_m=window_m,
            start_margin_m=opts.get("start_margin_m", 5.0), knee_db=opts.get("knee_db", 3.0),
        )
        if fit.best is None:
            (out_dir / f"{freq_key}_metrics.json").write_text(json.dumps({"fit": fit.to_dict()}, indent=2, default=str))
            report.note(f"{freq_key}: no TVG model could be fit: {'; '.join(fit.notes)}")
            continue
        configured = configured_params(cfg, freq_key)
        check_best = check_tvg(ds, fit.tvg_params, fit.window_m, range_var=range_var, bin_width_m=bin_width_m)
        check_config = check_tvg(ds, configured, fit.window_m, range_var=range_var, bin_width_m=bin_width_m)
        (out_dir / f"{freq_key}_metrics.json").write_text(json.dumps(
            {"fit": fit.to_dict(), "check_best": check_best.to_dict(), "check_config": check_config.to_dict()},
            indent=2, default=str,
        ))
        best = fit.best_model
        report.results[freq_key] = {
            "best_model": best.name, "best_rms_db": _round(best.rms_db), "runner_up": fit.runner_up,
            "free_n": _round(fit.free_n, 1), "fit_window_m": [_round(w) for w in fit.window_m],
            "window_source": fit.window_source, "low_contrast": fit.low_contrast,
            "tvg_params": {"spreading_loss_db": best.spreading_loss_db,
                           "absorption_db_per_m": _round(best.absorption_db_per_m, 4)},
            "tvg_status": fit.tvg_status,
            "spread_after_best_db": _round(check_best.spread_db),
            "trend_after_best_db_per_10m": _round(check_best.trend_db_per_10m),
            "configured": check_config.params, "configured_model": check_config.model_name,
            "spread_after_config_db": _round(check_config.spread_db),
            "trend_after_config_db_per_10m": _round(check_config.trend_db_per_10m),
        }
        logger.info("%s: best %s (%.2f dB RMS, free n %.1f); after TVG %.2f dB; config %s leaves %s dB",
                    freq_key, best.name, best.rms_db, fit.free_n, check_best.spread_db, check_config.model_name,
                    _round(check_config.spread_db))
        for note in fit.notes:
            if not note.startswith("Best "):
                report.note(f"{freq_key}: {note}")
        if (check_config.spread_db is not None and check_best.spread_db is not None
                and check_config.spread_db - check_best.spread_db > _FLATNESS_TOLERANCE_DB):
            report.note(
                f"{freq_key}: the config's TVG ({check_config.model_name}) leaves {check_config.spread_db:.2f} dB "
                f"against {check_best.spread_db:.2f} dB for {best.name} -- check the figure and update "
                "correct: before step 3"
            )
        fig = fit_figure(fit, check_best, check_config, f"{path.name}  {freq_key}  pings {start}:{stop}")
        report.figure(save_figure(fig, out_dir / f"{freq_key}_tvg_fit.png", dpi=140, tight=False), paths.output_root)

    report.write(paths.reports, next_step="03_correct.py")


if __name__ == "__main__":
    main()
