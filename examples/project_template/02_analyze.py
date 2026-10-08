"""
Step 2 -- measure the converted (uncorrected) data before correcting it:
usable range and the TVG range-decay fit, per frequency, on one file.

Compares the fit with the TVG recipe in the config's `correct:` section and
flags a disagreement in the report, so the recipe is checked before step 3
applies it.

Writes:
  <output_root>/analyze/<freq>_tvg_fit.png      fit + residuals
  <output_root>/analyze/<freq>_metrics.json     fit_range_decay result
  <output_root>/reports/02_analyze.{json,md}

Usage:
    python 02_analyze.py config.yaml

2026 rshomberg@uri.edu
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from steplib import logger, save_figure, ss_h5_files, start_step

from sidescanpy.analysis import estimate_usable_range, fit_range_decay
from sidescanpy.core.history import has_step
from sidescanpy.io.hdf5 import load_channel_dataset
from sidescanpy.processing import compute_geometric_ranges, invert
from sidescanpy.viz import plot_candidate_fit, plot_range_decay_residuals, set_fit_window_limits

# fit_range_decay's tvg_status names the loss terms already compensated in
# the data; with none compensated (status "none"), both terms belong in the
# software TVG -- the "full physics" recipe.
_RECIPE_FOR_STATUS = {"none": "full physics", "spreading_only": "absorption only",
                      "absorption_only": "spreading only", "full": "no TVG"}


def configured_recipe(cfg: dict, freq_key: str) -> str:
    correct = cfg.get("correct", {})
    tvg = dict((correct.get("params") or {}).get("tvg") or {})
    tvg.update(((correct.get("frequency_params") or {}).get(freq_key) or {}).get("tvg") or {})
    spreading = tvg.get("spreading_loss_db", 20.0) != 0
    absorption = tvg.get("absorption_db_per_m", None) != 0
    return {(True, True): "full physics", (True, False): "spreading only",
            (False, True): "absorption only", (False, False): "no TVG"}[(spreading, absorption)]


def fit_figure(decay, title: str):
    fit = decay.fit
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for i, c in enumerate(fit.candidates):
        plot_candidate_fit(c, fit.observed_range_m, fit.observed_loss_db, ax=ax_fit, show_observed=(i == 0))
    ax_fit.set_title(f"Decay fit -- best: {decay.best_candidate}, RMSE={decay.best_rmse_db:.2f} dB")
    ax_fit.set_xlabel("")
    plot_range_decay_residuals(decay, ax=ax_res, relative_x=False)
    ax_res.set_title("Residuals")
    ax_res.set_xlabel("Slant range (m)")
    set_fit_window_limits(ax_fit, decay)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


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
        usable = estimate_usable_range(ds, range_var=range_var, bin_width_m=bin_width_m).usable_range_m
        if usable is None:
            report.note(f"{freq_key}: usable range could not be determined; no TVG fit")
            continue
        decay = fit_range_decay(ds, range_var=range_var, bin_width_m=bin_width_m, usable_range_m=usable,
                                r_ref_m="auto", r_fit_min_m="auto")
        (out_dir / f"{freq_key}_metrics.json").write_text(json.dumps(decay.to_dict(), indent=2, default=str))
        if decay.fit is None:
            report.note(f"{freq_key}: no decay model could be fit ({decay.tvg_status}): {'; '.join(decay.notes)}")
            continue
        suggested = _RECIPE_FOR_STATUS.get(decay.tvg_status, decay.tvg_status)
        configured = configured_recipe(cfg, freq_key)
        report.results[freq_key] = {
            "usable_range_m": round(usable, 2), "tvg_status": decay.tvg_status,
            "best_candidate": decay.best_candidate, "best_rmse_db": round(decay.best_rmse_db, 2),
            "fit_window_m": [round(decay.r_fit_min_m, 2), round(decay.r_fit_max_m, 2)],
            "r_ref_m": round(decay.r_ref_m, 2), "suggested_recipe": suggested, "configured_recipe": configured,
        }
        logger.info("%s: %s (best %s, RMSE %.2f dB); config uses %s", freq_key, decay.tvg_status,
                    decay.best_candidate, decay.best_rmse_db, configured)
        if suggested != configured:
            report.note(f"{freq_key}: the fit suggests {suggested} but the config applies {configured} -- "
                        "check the figure and update correct: before step 3")
        fig = fit_figure(decay, f"{path.name}  {freq_key}  pings {start}:{stop}")
        report.figure(save_figure(fig, out_dir / f"{freq_key}_tvg_fit.png", dpi=140, tight=False), paths.output_root)

    report.write(paths.reports, next_step="03_correct.py")


if __name__ == "__main__":
    main()
